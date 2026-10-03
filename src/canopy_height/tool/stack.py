"""Training stack, study-area mosaics and ALS reference of the canopy-height tool.

Every used cell of plan/cells.csv has one whole 256 x 256 raster per layer, i.e. it is one training
chip, encoded exactly as in the paper's stacks:

    X         uint16 [76, 256, 256] = 64 x (Embedding+1)*1e4, DEM, 2 x (S1 dB+50)*100, 9 x S2 as exported
              (float64 -> NaN->0 -> clip 0..65535 -> truncate)
    seasonal  uint16 [44, 256, 256] = S1_asc_0..3 (VV, VH: (dB+50)*100), then S2_0..3 (B2 ... B12: 0-1 reflectance
              *1e4), same NaN->0 / clip / truncate, as the paper's seasonal stacks; band order
              channels.SEASONAL_BANDS; only for the inputs T / TE
    GEDI      float32 [256, 256], NaN / -9999 -> -999 (= no label)

The input representation project['input'] (project.INPUTS) decides which rasters are read: the annual layers it
does not use are not read (not downloaded) and their channels of X are 0 (= no data, not read by its models); the
seasonal layers (project.SEASONAL_LAYERS) are stacked when it needs them. With the default AE every layer is used
and the stacks are those of the paper.

Chips are stacked in cells.csv order, because that order defines the training / validation split.
stacks/hashes.json (written last) records
the data hashes, cells.csv, the settings the rasters depend on (and the input representation) and the size / mtime
of every input raster; the stack is rebuilt when any of them changes. `stack_id` identifies a complete stack,
`model_is_current` tells whether a model was trained on it.

The mosaics put the same X (and seasonal) encoding on the bounding box of the study-area cells (input of the
prediction stage; surrounding cells inside the box are filled in as context for the UNet), next to GEDI, the
published benchmarks (HRCH <- ETH, GFCH <- UMD, GMTCH), the study-area mask and, optionally, an ALS reference for
the accuracy check. They are written cell by cell into tiled sparse GeoTIFFs (cells without data are not stored),
so building them needs the memory of a few cells whatever the size of the box.
"""
import hashlib
import json
import math
import os
import shutil
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from .. import channels, labels
from .project import (BENCHMARKS, CELL_M, CELL_PX, INPUTS, LABEL_LAYER, PART, RES, SEASONAL_LAYERS, X_LAYERS,
                      canonical_input, input_layers, replace_retry)

WORKERS = 8                                            # raster-reading threads
TOL = 0.01                                             # m, tolerance of cell origins
X_BANDS = {"Embedding": 64, "DEM": 1, "S1": 2, "S2": 9}
# bands of the seasonal layers S1_asc_k / S2_k (by prefix), in the order of the paper's rasters
SEASONAL_BANDS = {"S1": ["VV", "VH"], "S2": ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12"]}
INDEX_COLUMNS = ["chip", "part", "row_in_part", "cell", "role", "dist_m", "x0", "y1", "epsg", "gedi_px"]
GTIFF = dict(driver="GTiff", tiled=True, blockxsize=CELL_PX, blockysize=CELL_PX, compress="deflate",
             BIGTIFF="IF_SAFER", num_threads="ALL_CPUS",   # threaded deflate: writing is the bottleneck
             SPARSE_OK="TRUE")                             # blocks never written (or all nodata) are not stored
ALS_FINE_M = 2.0                                       # ALS rasters <= this -> 10 m p90, coarser -> nearest neighbour
ALS_MAX_M = 100.0                                      # 99th percentile of the ALS reference above this: not metres


# ------------------------------------------------------------------------------------------------ encodings
def encode_x(emb, dem, s1, s2):
    """76-band uint16 X: 64 x (Embedding+1)*1e4, DEM, 2 x (S1 dB+50)*100, 9 x S2 raw (reflectance*1e4 as exported);
    float64 -> NaN->0 -> clip 0..65535 -> truncate (identical to the paper's stacks)."""
    x = np.concatenate([(emb + 1) * 1e4, dem, (s1 + 50) * 100, s2])
    return np.clip(np.nan_to_num(x, nan=0.0), 0, 65535).astype(np.uint16)


def encode_seasonal(arrays):
    """44-band uint16 seasonal input from {layer: [bands, H, W]} of the SEASONAL_LAYERS (or a sequence in that
    order): S1_asc_0..3 (VV, VH) then S2_0..3 (B2 ... B12) = channels.SEASONAL_BANDS; float64 -> S1 (dB+50)*100,
    S2 (0-1 reflectance)*1e4 -> NaN->0 -> clip 0..65535 -> truncate (as the paper's seasonal
    stacks)."""
    if not isinstance(arrays, dict):
        arrays = dict(zip(SEASONAL_LAYERS, arrays))
    out = []
    for layer in SEASONAL_LAYERS:
        a = np.asarray(arrays[layer], dtype=np.float64)
        if a.shape[0] != len(SEASONAL_BANDS[layer[:2]]):
            raise ValueError(f"{layer}: {a.shape[0]} bands, expected {len(SEASONAL_BANDS[layer[:2]])}")
        a = (a + 50.0) * 100.0 if layer.startswith("S1") else a * 10000.0
        out.append(np.clip(np.nan_to_num(a, nan=0.0), 0, 65535).astype(np.uint16))
    return np.concatenate(out)


def enc999(a):
    """GEDI labels: float32, NaN (and the -9999 export sentinel) -> -999."""
    c = np.nan_to_num(a.astype(np.float32), nan=-999)
    c[c == -9999] = -999
    return c


def data_sha256(arr):
    """sha256 of the array data (C order), as recorded in hashes.json."""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ cells
def read_cells(project):
    """plan/cells.csv in file (= stack) order, `use` as bool."""
    f = project.cells_file
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: run the plan stage first")
    df = pd.read_csv(f)
    truthy = ("true", "1", "1.0", "yes")
    df["use"] = df["use"].map(lambda v: str(v).strip().lower() in truthy) if "use" in df else True
    if "role" not in df:
        df["role"] = "ring"
    return df


def read_grid(project):
    """plan/grid.json as a dict, None before the plan stage."""
    f = project.grid_file
    if not f.exists():
        return None
    g = json.loads(Path(f).read_text(encoding="utf-8"))
    return g


def _epsg_of(row, grid):
    """Expected EPSG of a cell: the grid's, else unknown (None)."""
    return int(grid["epsg"]) if grid else None


def read_cell_layer(path, row, bands=None, epsg=None, names=None):
    """Whole-cell raster as float64 and its EPSG, after checking
    that it is 256 x 256 at 10 m with the upper-left corner at the cell's (x0, y1) (and in `epsg` if given; with
    `names`: that it has these bands, band descriptions - when set - in this order)."""
    with rasterio.open(path) as s:
        t, e = s.transform, (s.crs.to_epsg() if s.crs else None)
        if (s.height, s.width) != (CELL_PX, CELL_PX):
            raise ValueError(f"{path}: {s.width} x {s.height} px, expected one whole {CELL_PX} x {CELL_PX} cell")
        if (abs(t.c - float(row["x0"])) > TOL or abs(t.f - float(row["y1"])) > TOL or abs(t.a - RES) > 1e-9
                or abs(t.e + RES) > 1e-9 or t.b or t.d):
            raise ValueError(f"{path}: grid ({t.c}, {t.f}, {t.a} m) is not that of cell {row['cell']} "
                             f"({row['x0']}, {row['y1']}, {RES} m)")
        if epsg is not None and e != int(epsg):
            raise ValueError(f"{path}: EPSG {e}, expected {int(epsg)}")
        if names is not None:
            if s.count != len(names):
                raise ValueError(f"{path}: {s.count} bands, expected {len(names)} ({', '.join(names)})")
            if all(s.descriptions) and list(s.descriptions) != list(names):
                raise ValueError(f"{path}: bands {', '.join(s.descriptions)}, expected {', '.join(names)}")
        return s.read(bands).astype("float64"), e


def _needed(project, row, layer, input_name):
    """Raster of a layer the input representation needs (FileNotFoundError when it is missing)."""
    p = project.cell_raster(row, layer)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: input {input_name} needs the {layer} layer of cell {row['cell']} "
                                f"(run the download stage)")
    return p


def cell_x(project, row, epsg=None, input_name=None):
    """76-band uint16 annual model input of a cell (encode_x of its Embedding, DEM, S1, S2 rasters) and its EPSG.
    Only the annual layers of the input representation (input_name, default project['input']) are read; the
    channels of the others are 0 (= no data). With AE (every layer) this is the paper's encoding."""
    inp = input_name or project["input"]
    use = INPUTS[inp]["annual"]
    arrs = []
    for layer in X_LAYERS:
        if layer not in use:                                        # not downloaded, not read by the models
            arrs.append(np.full((X_BANDS[layer], CELL_PX, CELL_PX), np.nan))
            continue
        p = _needed(project, row, layer, inp)
        a, epsg = read_cell_layer(p, row, epsg=epsg)
        if a.shape[0] != X_BANDS[layer]:
            raise ValueError(f"{p}: {a.shape[0]} bands, expected {X_BANDS[layer]}")
        arrs.append(a)
    return encode_x(*arrs), epsg


def cell_seasonal(project, row, epsg=None):
    """44-band uint16 seasonal model input of a cell (encode_seasonal of its S1_asc_0..3 and S2_0..3 rasters: S1
    VV, VH in dB, S2 B2 ... B12 as 0-1 reflectance, float64) and its EPSG."""
    arrs = {}
    for layer in SEASONAL_LAYERS:
        p = _needed(project, row, layer, project["input"])
        arrs[layer], epsg = read_cell_layer(p, row, epsg=epsg, names=SEASONAL_BANDS[layer[:2]])
    return encode_seasonal(arrs), epsg


def _imap(fn, items, workers=WORKERS, ahead=4 * WORKERS):
    """fn over items with a thread pool; results in order, at most `ahead` in flight."""
    ex = ThreadPoolExecutor(workers)
    try:
        it = iter(items)
        q = deque(ex.submit(fn, x) for x in islice(it, ahead))
        while q:
            f = q.popleft()
            for x in islice(it, 1):
                q.append(ex.submit(fn, x))
            yield f.result()
    finally:
        ex.shutdown(wait=True, cancel_futures=True)


# ------------------------------------------------------------------------------------------------ training stack
def _settings(project):
    """Settings the downloaded rasters depend on and the input representation (recorded in hashes.json)."""
    return dict(year=int(project["year"]), gedi_window=[str(d) for d in project["gedi_window"]],
                built_up_mask=bool(project["built_up_mask"]), input=str(project["input"]))


def _layers(project):
    """Raster layers of a training chip: those of the input representation, then the GEDI labels."""
    return input_layers(project["input"]) + [LABEL_LAYER]


def _inputs(project, rows):
    """{raster: [size, mtime_ns]} of the input (annual and seasonal layers of the input representation) and label
    rasters of `rows` (paths relative to the project folder when inside it); None if one is missing."""
    out = {}
    layers = _layers(project)
    for r in rows:
        for layer in layers:
            p = project.cell_raster(r, layer)
            try:
                st = p.stat()
            except OSError:
                return None
            try:
                key = p.relative_to(project.root).as_posix()
            except ValueError:
                key = str(p)
            out[key] = [st.st_size, st.st_mtime_ns]
    return out


def _read_hashes(d):
    try:
        with open(d / "hashes.json", encoding="utf-8") as fh:
            h = json.load(fh)
    except (OSError, ValueError):
        return None
    return h if isinstance(h, dict) else None


def _write_hashes(d, h):
    tmp = d / "_hashes.tmp.json"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(h, fh, indent=1)
    replace_retry(tmp, d / "hashes.json")


def _complete(d, h):
    """hashes.json `h` (written last) lists exactly the stack parts in d, and train_index.csv exists."""
    files = (h or {}).get("files")
    return (bool(files) and (d / "train_index.csv").exists()
            and {p.name for p in d.glob("train_*part*.npy")} == set(files))


def _stack_state(project, d, rows, cells_sha):
    """Why the training stack in d must be (re)built; '' when it is up to date."""
    h = _read_hashes(d)
    if h is None:
        return "no stacks/hashes.json"
    if h.get("cells_sha256") != cells_sha:
        return "plan/cells.csv changed"
    if not _complete(d, h):
        return "stack parts missing or left over"
    inputs = _inputs(project, rows)
    if inputs is None:
        return "input rasters missing"
    if "inputs" not in h:
        return "no input fingerprint in stacks/hashes.json"
    now = _settings(project)
    old = dict(h.get("settings") or {})
    if old != now:
        return "settings changed: " + ", ".join(f"{k} {old.get(k)} -> {v}" for k, v in now.items() if old.get(k) != v)
    if h["inputs"] != inputs:
        changed = [k for k, v in inputs.items() if h["inputs"].get(k) != v]
        changed += [k for k in h["inputs"] if k not in inputs]
        return f"{len(changed)} input rasters changed since the stack was built, e.g. {changed[0]}"
    return ""


def stack_id(project):
    """Identity of the training stack: sha256 of json.dumps(hashes.json 'files', sort_keys=True) - the input, label
    and (inputs T / TE) seasonal parts; None unless the stack is complete (hashes.json, written last, lists exactly
    the parts on disk)."""
    d = project.path("stacks")
    h = _read_hashes(d)
    if not _complete(d, h):
        return None
    return hashlib.sha256(json.dumps(h["files"], sort_keys=True).encode("utf-8")).hexdigest()


def _labelled_pixels(project, block=64):
    """GEDI labels > 0 in the training stack (the rows of RF-SLS before its hold-out split)."""
    n = 0
    for f in project.stack_files(LABEL_LAYER):
        y = np.load(f, mmap_mode="r")
        for i in range(0, len(y), block):
            b = np.array(y[i:i + block])
            n += int(((b > 0) & np.isfinite(b)).sum())
        del y
    return n


def model_is_current(project, model):
    """(True, '') if models/<model> was trained on the current training stack, else (False, reason).

    Runs record the stack in stack_id.txt (== stack_id(project)); earlier runs without it count as current when
    their result.json fits the current stack: the input representation if recorded (UNets: model.input, RF-SLS:
    'input') and the part files and chip count (UNets: 'annual', 'seasonal', 'labels', 'n_chips') or the number
    of labelled pixels (RF-SLS: 'n_rows')."""
    sid = stack_id(project)
    if sid is None:
        return False, "the training stack is incomplete (run the stack stage)"
    d = project.model_dir(model)
    f = d / "stack_id.txt"
    if f.exists():
        if f.read_text(encoding="utf-8").strip() == sid:
            return True, ""
        return False, "trained on another training stack (stack_id.txt differs)"
    try:
        with open(d / "result.json", encoding="utf-8") as fh:
            res = json.load(fh)
    except (OSError, ValueError):
        return False, "no finished run (result.json)"
    inp = res.get("input") if model == "rf-sls" else (res.get("model") or {}).get("input")
    inp = canonical_input(inp) if inp else inp
    if inp and inp != project["input"]:
        return False, f"trained with input {inp}, the project's input is {project['input']}"
    if model == "rf-sls":
        n = _labelled_pixels(project)
        if res.get("n_rows") == n:
            return True, ""
        return False, f"trained on {res.get('n_rows')} labelled pixels, the current stack has {n}"
    chips = _read_hashes(project.path("stacks"))["chips"]

    def names(files):
        return [Path(p).name for p in files or []]
    was = (names(res.get("annual")), names(res.get("labels")), res.get("n_chips"), names(res.get("seasonal")))
    now = (names(project.stack_files("x")), names(project.stack_files(LABEL_LAYER)), chips,
           names(project.stack_files("seasonal")))
    if was == now:
        return True, ""
    if was[:3] == now[:3]:
        return False, (f"trained on {len(was[3])} seasonal part(s), the current stack has {len(now[3])} seasonal "
                       f"part(s)")
    return False, (f"trained on {was[2]} chips in {len(was[0])} part(s), the current stack has {chips} chips in "
                   f"{len(now[0])} part(s)")


def _chip(project, row, grid):
    """(X, seasonal or None, GEDI, EPSG) of a cell."""
    x, epsg = cell_x(project, row, _epsg_of(row, grid))
    s = cell_seasonal(project, row, epsg)[0] if project.seasonal else None
    g, _ = read_cell_layer(project.cell_raster(row, LABEL_LAYER), row, 1, epsg)
    return x, s, enc999(g), epsg


SCRATCH_ENV = "CHM_STACK_SCRATCH"   # local folder for the memmaps (colab.run: memmaps on the Drive mount are unreliable)


def _write_part(project, pi, rows, grid, index, scratch=None):
    """One part: X, GEDI (and seasonal) written through _<name>.tmp.npy memmaps (in `scratch` if given, else next to
    the part files); returns ({tmp file: part file}, hashes.json entries). The caller moves the tmp files into place
    once every part is written."""
    from numpy.lib.format import open_memmap
    n = len(rows)
    out = {"x": (project.stack_part("x", pi), np.uint16, (n, len(channels.ANNUAL_BANDS), CELL_PX, CELL_PX)),
           LABEL_LAYER: (project.stack_part(LABEL_LAYER, pi), np.float32, (n, CELL_PX, CELL_PX))}
    if project.seasonal:
        out["seasonal"] = (project.stack_part("seasonal", pi), np.uint16,
                           (n, len(channels.SEASONAL_BANDS), CELL_PX, CELL_PX))
    # "_" prefix: never matched by Project.stack_files ("train_part*.npy") while being written
    tmp = {k: (Path(scratch) if scratch else p.parent) / f"_{p.stem}.tmp.npy" for k, (p, _, _) in out.items()}
    mm = {k: open_memmap(tmp[k], mode="w+", dtype=dt, shape=sh) for k, (_, dt, sh) in out.items()}
    sha = {k: hashlib.sha256() for k in out}
    try:
        for j, (x, s, g, epsg) in enumerate(_imap(lambda r: _chip(project, r, grid), rows)):
            for k, a in (("x", x), (LABEL_LAYER, g), ("seasonal", s)):
                if k not in mm:
                    continue
                mm[k][j] = a
                sha[k].update(a.tobytes())
            r = rows[j]
            index.append(dict(chip=len(index), part=pi, row_in_part=j, cell=r["cell"], role=r.get("role", "ring"),
                              dist_m=r.get("dist_m", np.nan), x0=r["x0"], y1=r["y1"], epsg=epsg,
                              gedi_px=int((g > -999).sum())))
            if (j + 1) % 100 == 0 and j + 1 < n:
                project.log(f"stack: part {pi}: {j + 1}/{n} chips")
        for k in mm:
            mm[k].flush()
    except BaseException:
        mm.clear()
        for t in tmp.values():
            t.unlink(missing_ok=True)
        raise
    mm.clear()
    files = {p.name: dict(shape=list(sh), dtype=np.dtype(dt).name, data_sha256=sha[k].hexdigest())
             for k, (p, dt, sh) in out.items()}
    return {tmp[k]: p for k, (p, _, _) in out.items()}, files


def build_train_stack(project, overwrite=False):
    """Training stack of the used cells of plan/cells.csv, in file order: stacks/train_partNNN.npy (uint16
    [n, 76, 256, 256]; channels of annual layers the input representation does not use are 0), with the inputs
    T / TE stacks/train_seasonal_partNNN.npy (uint16 [n, 44, 256, 256]), and stacks/train_GEDI_partNNN.npy
    (float32 [n, 256, 256], -999 = no label) in parts of PART chips, stacks/train_index.csv and stacks/hashes.json
    (sha256 of the array data and of cells.csv, the settings year / gedi_window / built_up_mask / input and size +
    mtime of every input raster). The parts appear only once all are written, hashes.json last. Skipped when
    hashes.json matches cells.csv, the settings and the input rasters and lists exactly the parts on disk
    (overwrite=True rebuilds). Returns the index as a DataFrame."""
    d = project.path("stacks")
    df = read_cells(project)
    rows = df[df["use"]].to_dict("records")
    cells_sha = file_sha256(project.cells_file)
    why = "rebuild requested" if overwrite else _stack_state(project, d, rows, cells_sha)
    if not why:
        project.log(f"stack: training stack up to date ({d})")
        return pd.read_csv(d / "train_index.csv")
    if not rows:
        raise ValueError(f"{project.cells_file}: no used cells")
    grid = read_grid(project)
    missing = [str(p) for r in rows for p in (project.cell_raster(r, lay) for lay in _layers(project))
               if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} rasters of used cells missing (run the download stage), "
                                f"e.g. {missing[:3]}")
    if (d / "hashes.json").exists():
        project.log(f"stack: rebuilding the training stack ({why})")
    inputs = _inputs(project, rows)
    d.mkdir(parents=True, exist_ok=True)
    scratch = os.environ.get(SCRATCH_ENV) or None
    if scratch:
        os.makedirs(scratch, exist_ok=True)
    for p in [d / "hashes.json", d / "train_index.csv", *d.glob("train_*part*.npy"), *d.glob("_train_*.tmp.npy"),
              *(Path(scratch).glob("_train_*.tmp.npy") if scratch else [])]:
        p.unlink(missing_ok=True)
    parts = [rows[i:i + PART] for i in range(0, len(rows), PART)]
    project.log(f"stack: {len(rows)} training chips from {project.cells_file.name}, {len(parts)} part(s), input "
                f"{project['input']}" + (" (annual + seasonal parts)" if project.seasonal else ""))
    index, files, moves, t0 = [], {}, {}, time.time()
    try:
        for pi, sub in enumerate(parts, 1):
            m, f = _write_part(project, pi, sub, grid, index, scratch)
            moves.update(m)
            files.update(f)
            project.log(f"stack: part {pi}/{len(parts)} written, {len(sub)} chips ({time.time() - t0:.0f} s)")
    except BaseException:
        for t in moves:
            t.unlink(missing_ok=True)
        raise
    for t, p in moves.items():
        if t.parent != p.parent:                   # built in the scratch folder: copy in, check, then rename
            q = p.with_name(t.name)
            shutil.copyfile(t, q)
            if q.stat().st_size != t.stat().st_size:
                raise OSError(f"copying {t} to {q}: {q.stat().st_size} of {t.stat().st_size} bytes")
            t.unlink()
            t = q
        replace_retry(t, p)
    for p in moves.values():                       # every part must be in place before hashes.json is written
        if not p.exists():
            raise OSError(f"stack part {p} missing after it was written (storage did not keep it)")
    ix = pd.DataFrame(index, columns=INDEX_COLUMNS)
    tmp = d / "_train_index.tmp.csv"
    ix.to_csv(tmp, index=False)
    replace_retry(tmp, d / "train_index.csv")
    _write_hashes(d, dict(cells_sha256=cells_sha, chips=len(rows), part_size=PART, settings=_settings(project),
                          files=files, inputs=inputs))
    project.log(f"stack: done, {len(rows)} chips, {int((ix.gedi_px > 0).sum())} with GEDI labels, "
                f"{int(ix.gedi_px.sum())} label pixels")
    return ix


# ------------------------------------------------------------------------------------------------ mosaics
def _grid_pos(grid, row):
    """(row, col) of a cell on the grid, None if it is off the grid."""
    c = (float(row["x0"]) - float(grid["x0"])) / CELL_M
    r = (float(grid["y1"]) - float(row["y1"])) / CELL_M
    ic, ir = round(c), round(r)
    if abs(c - ic) * CELL_M > TOL or abs(r - ir) * CELL_M > TOL:
        return None
    return ir, ic


def mosaic_layout(project):
    """Grid of the study-area mosaics: the bounding box of the aoi cells on plan/grid.json. None without a grid or
    aoi cells; else dict(epsg, x0, y1, n_rows, n_cols, transform, cells={(row, col) in the box: cells.csv row})."""
    grid = read_grid(project)
    if grid is None:
        return None
    if abs(float(grid.get("res", RES)) - RES) > 1e-9 or int(grid.get("cell_px", CELL_PX)) != CELL_PX:
        raise ValueError(f"{project.grid_file}: expected res {RES} m and {CELL_PX} px cells")
    recs = read_cells(project).to_dict("records")
    pos = [_grid_pos(grid, r) for r in recs]
    aoi = [p for p, r in zip(pos, recs) if r["role"] == "aoi"]
    if not aoi:
        return None
    if any(p is None for p in aoi):
        raise ValueError(f"study-area cells of {project.cells_file} are not on the grid of {project.grid_file}")
    r0, r1 = min(p[0] for p in aoi), max(p[0] for p in aoi)
    c0, c1 = min(p[1] for p in aoi), max(p[1] for p in aoi)
    cells = {}
    for p, r in zip(pos, recs):
        if p is not None and r0 <= p[0] <= r1 and c0 <= p[1] <= c1:
            cells.setdefault((p[0] - r0, p[1] - c0), r)
    x0, y1 = float(grid["x0"]) + c0 * CELL_M, float(grid["y1"]) - r0 * CELL_M
    return dict(epsg=int(grid["epsg"]), x0=x0, y1=y1, n_rows=r1 - r0 + 1, n_cols=c1 - c0 + 1,
                transform=from_origin(x0, y1, RES, RES), cells=cells)


def _write_cells(path, profile, positions, get, descriptions, tags=None):
    """GeoTIFF written cell by cell, each into its 256 x 256 block; get((r, c)) -> array or None (block left
    unwritten: not stored, read as nodata). Written as <name>.partial.tif, renamed when complete. Returns the
    number of cells with data."""
    tmp = path.with_name(f"{path.stem}.partial.tif")
    n = 0
    try:
        with rasterio.open(tmp, "w", **profile) as dst:
            for (r, c), a in zip(positions, _imap(get, positions)):
                if a is None:
                    continue
                n += 1
                dst.write(a if a.ndim == 3 else a[None], window=Window(c * CELL_PX, r * CELL_PX, CELL_PX, CELL_PX))
            for i, desc in enumerate(descriptions, 1):
                dst.set_band_description(i, desc)
            if tags:
                dst.update_tags(**tags)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    replace_retry(tmp, path)
    return n


def aoi_shapes(project, epsg):
    """Study area (aoi.geojson, EPSG:4326) as rasterize shapes in EPSG:epsg."""
    from rasterio.warp import transform_geom
    with open(project.aoi_file, encoding="utf-8") as fh:
        gj = json.load(fh)
    items = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
    geoms = [f.get("geometry") if f.get("type") == "Feature" else f for f in items]
    return [(transform_geom("EPSG:4326", CRS.from_epsg(epsg), g), 1) for g in geoms if g]


def aoi_mask(project, transform, shape, epsg, shapes=None):
    """uint8 [H, W] on `transform`: 1 where the pixel centre lies inside the study area (aoi.geojson, EPSG:4326;
    or `shapes` from aoi_shapes)."""
    from rasterio.features import rasterize
    shapes = aoi_shapes(project, epsg) if shapes is None else shapes
    if not shapes:
        return np.zeros(shape, np.uint8)
    return rasterize(shapes, out_shape=shape, transform=transform, fill=0, all_touched=False, dtype="uint8")


def _write_aoi_mask(project, lay, profile, path):
    """Study-area mask of the mosaic box written one row of cells at a time; returns the pixels inside."""
    shapes = aoi_shapes(project, lay["epsg"])
    W, n = lay["n_cols"] * CELL_PX, 0
    try:
        with rasterio.open(path, "w", **dict(profile, count=1, dtype="uint8", nodata=None)) as dst:
            for r in range(lay["n_rows"]):
                win = Window(0, r * CELL_PX, W, CELL_PX)
                m = aoi_mask(project, window_transform(win, lay["transform"]), (CELL_PX, W), lay["epsg"], shapes)
                n += int(m.sum())
                dst.write(m, 1, window=win)
            dst.set_band_description(1, "study area (1 = inside)")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return n


def mosaics_match_input(project):
    """mosaic/annual.tif (and, for the inputs T / TE, mosaic/seasonal.tif) exist and were built for the project's
    input representation (tag 'input')."""
    files = [project.mosaic("annual")] + ([project.mosaic("seasonal")] if project.seasonal else [])
    for f in files:
        if not f.exists():
            return False
        with rasterio.open(f) as s:
            if s.tags().get("input") != project["input"]:
                return False
    return True


def build_mosaics(project):
    """Study-area mosaics on the bounding box of the aoi cells (plan/grid.json): mosaic/annual.tif (76-band uint16,
    0 = no data; every aoi or ring cell in the box with the inputs of the input representation; channels of the
    annual layers it does not use are 0), for the inputs T / TE mosaic/seasonal.tif (44-band uint16, 0 = no data,
    bands channels.SEASONAL_BANDS; same cells), mosaic/GEDI.tif, the benchmarks of project['benchmarks']
    downloaded for the aoi cells (HRCH <- ETH, GFCH <- UMD, GMTCH; float32, values as downloaded, NaN = no data)
    and mosaic/aoi_mask.tif (uint8, 1 inside the study area). annual.tif and seasonal.tif carry the input
    representation as tag 'input' (mosaics_match_input); a seasonal.tif left from another input is removed. Cells
    without data are not stored. Fails when the study area covers no pixel centre of the box. Returns the written
    paths."""
    lay = mosaic_layout(project)
    if lay is None:
        project.log("mosaic: no plan/grid.json with study-area cells: no study-area mosaics")
        return []
    nr, nc, cells, epsg = lay["n_rows"], lay["n_cols"], lay["cells"], lay["epsg"]
    need, seasonal, tags = input_layers(project["input"]), project.seasonal, dict(input=project["input"])
    prof = dict(GTIFF, width=nc * CELL_PX, height=nr * CELL_PX, crs=CRS.from_epsg(epsg), transform=lay["transform"])
    project.path("mosaic").mkdir(parents=True, exist_ok=True)
    positions = sorted(cells)
    n_aoi = sum(r["role"] == "aoi" for r in cells.values())
    if nr * nc > 4 * len(positions):
        project.log(f"mosaic: WARNING the study-area parts are far apart: {len(positions)} planned cells in a box of "
                    f"{nr} x {nc} cells (consider one project per part)")
    written = []

    mask_tmp = None
    if project.aoi_file.exists():
        mask_tmp = project.mosaic("aoi_mask").with_name("aoi_mask.partial.tif")
        n_in = _write_aoi_mask(project, lay, prof, mask_tmp)
        if n_in == 0:
            mask_tmp.unlink(missing_ok=True)
            raise ValueError(f"{project.aoi_file.name} covers no 10 m pixel centre of the {n_aoi} study-area cells "
                             f"(a study area narrower than a 10 m pixel): every map "
                             f"would be empty")
    else:
        project.log(f"mosaic: {project.aoi_file} missing, aoi_mask.tif not written")

    def has(rc, layers, roles=("aoi", "ring")):
        row = cells.get(rc)
        return row is not None and row["role"] in roles and all(project.cell_raster(row, x).exists() for x in layers)

    def get_x(rc):
        return cell_x(project, cells[rc], epsg)[0] if has(rc, need) else None

    def get_s(rc):
        return cell_seasonal(project, cells[rc], epsg)[0] if has(rc, need) else None

    try:
        f = project.mosaic("annual")
        n = _write_cells(f, dict(prof, count=len(channels.ANNUAL_BANDS), dtype="uint16", nodata=0, predictor=2,
                                 interleave="band"), positions, get_x, channels.ANNUAL_BANDS, tags)
        written.append(f)
        project.log(f"mosaic: annual.tif {nr} x {nc} cells ({nr * CELL_PX} x {nc * CELL_PX} px), {n} with inputs "
                    f"(input {project['input']})")
        f = project.mosaic("seasonal")
        if seasonal:
            n = _write_cells(f, dict(prof, count=len(channels.SEASONAL_BANDS), dtype="uint16", nodata=0,
                                     predictor=2, interleave="band"), positions, get_s, channels.SEASONAL_BANDS, tags)
            written.append(f)
            project.log(f"mosaic: seasonal.tif, {n} cells with inputs")
        elif f.exists():
            f.unlink()
            project.log(f"mosaic: seasonal.tif removed (input {project['input']} does not use the seasonal composites)")
        lack = [r["cell"] for rc, r in cells.items() if r["role"] == "aoi" and not has(rc, need)]
        if lack:
            project.log(f"mosaic: WARNING {len(lack)} study-area cells without inputs (no prediction there): "
                        f"{lack[:5]}")

        f32 = dict(prof, count=1, dtype="float32", nodata=np.nan, predictor=3)
        layers = [("GEDI", LABEL_LAYER, ("aoi", "ring"), "GEDI rh95 (m)")]
        for product in project["benchmarks"] or []:
            if product not in BENCHMARKS:
                project.log(f"mosaic: unknown benchmark {product!r} skipped (known: {', '.join(BENCHMARKS)})")
                continue
            layers.append((product, BENCHMARKS[product], ("aoi",), f"{product} canopy height (m)"))
        for name, layer, roles, desc in layers:
            if not any(has(rc, [layer], roles) for rc in positions):
                project.log(f"mosaic: no {layer} rasters for the study-area cells, {name}.tif not written")
                continue

            def get(rc, layer=layer, roles=roles):
                if not has(rc, [layer], roles):
                    return None
                a = read_cell_layer(project.cell_raster(cells[rc], layer), cells[rc], 1, epsg)[0].astype(np.float32)
                if layer == LABEL_LAYER:
                    a[a == -9999] = np.nan
                return a

            f = project.mosaic(name)
            n = _write_cells(f, f32, positions, get, [desc])
            written.append(f)
            project.log(f"mosaic: {name}.tif from {layer}, {n} cells")

        if mask_tmp is not None:
            f = project.mosaic("aoi_mask")
            replace_retry(mask_tmp, f)
            written.append(f)
            project.log(f"mosaic: aoi_mask.tif, {n_in} pixels inside the study area")
    finally:
        if mask_tmp is not None:
            mask_tmp.unlink(missing_ok=True)
    return written


# ------------------------------------------------------------------------------------------------ ALS reference
def _warp_nearest(paths, grid_path, out_path, strip=4 * CELL_PX):
    """10 m ALS rasters -> nearest neighbour on the grid of grid_path (float32, NaN = no data; first raster wins)."""
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    with rasterio.open(grid_path) as g:
        H, W, T, crs = g.height, g.width, g.transform, g.crs
    srcs, vrts = [rasterio.open(p) for p in paths], []
    try:
        vrts = [WarpedVRT(s, crs=crs, transform=T, width=W, height=H, resampling=Resampling.nearest,
                          src_nodata=s.nodata, nodata=np.nan, dtype="float32") for s in srcs]
        prof = dict(GTIFF, count=1, dtype="float32", nodata=np.nan, predictor=3, width=W, height=H, crs=crs,
                    transform=T)
        with rasterio.open(out_path, "w", **prof) as dst:
            for r0 in range(0, H, strip):
                win = Window(0, r0, W, min(strip, H - r0))
                a = np.full((int(win.height), W), np.nan, np.float32)
                for v in vrts:
                    b = v.read(1, window=win)
                    fill = np.isnan(a) & ~np.isnan(b)
                    a[fill] = b[fill]
                dst.write(a, 1, window=win)
            dst.set_band_description(1, "ALS canopy height (m)")
    finally:
        for v in vrts:
            v.close()
        for s in srcs:
            s.close()


def _res_m(src):
    """Pixel size (m) of an open raster: the coarser of its two axes."""
    if src.crs is None:
        raise ValueError(f"ALS raster {src.name}: no coordinate reference system")
    rx, ry = abs(src.res[0]), abs(src.res[1])
    if src.crs.is_geographic:
        lat = math.radians((src.bounds.bottom + src.bounds.top) / 2)
        return max(rx * 111320.0 * math.cos(lat), ry * 110574.0)
    return max(rx, ry) * src.crs.linear_units_factor[1]


def _on_grid(src, crs, T):
    """An open raster lies on the 10 m analysis grid (CRS crs, transform T): same CRS, 10 m, corners on the grid."""
    t = src.transform
    same = src.crs == crs or (src.crs.to_epsg() is not None and src.crs.to_epsg() == crs.to_epsg())
    if not same or abs(t.a - RES) > 1e-6 or abs(t.e + RES) > 1e-6 or t.b or t.d:
        return False
    dx, dy = (t.c - T.c) / RES, (t.f - T.f) / RES
    return abs(dx - round(dx)) * RES <= TOL and abs(dy - round(dy)) * RES <= TOL


def _scale_raster(src_path, dst_path, scale, strip=4 * CELL_PX):
    """Copy of a float32 raster with its values multiplied by `scale`."""
    with rasterio.open(src_path) as s:
        prof = dict(GTIFF, count=s.count, dtype="float32", nodata=np.nan, predictor=3, width=s.width,
                    height=s.height, crs=s.crs, transform=s.transform)
        with rasterio.open(dst_path, "w", **prof) as dst:
            for r0 in range(0, s.height, strip):
                win = Window(0, r0, s.width, min(strip, s.height - r0))
                dst.write((s.read(window=win) * np.float32(scale)).astype(np.float32), window=win)
            for i, desc in enumerate(s.descriptions, 1):
                dst.set_band_description(i, desc or "")


def _finite_stats(path, strip=4 * CELL_PX, sample=2_000_000):
    """(number of finite values of band 1, their 99th percentile) - the percentile from a regular subsample of at
    most ~2 x `sample` values."""
    n, keep, step = 0, [], 1
    with rasterio.open(path) as s:
        for r0 in range(0, s.height, strip):
            a = s.read(1, window=Window(0, r0, s.width, min(strip, s.height - r0)))
            v = a[np.isfinite(a)]
            n += v.size
            keep.append(v[::step])
            if sum(k.size for k in keep) > 2 * sample:
                keep, step = [np.concatenate(keep)[::2]], step * 2
    v = np.concatenate(keep) if keep else np.empty(0, np.float32)
    return n, (float(np.percentile(v, 99)) if v.size else math.nan)


def als_reference(project):
    """mosaic/ALS.tif from project['als'] on the grid of mosaic/annual.tif, in metres (values x
    project['als_scale']). The method follows the rasters' own resolution: <= 2 m -> 10 m p90 as the paper's
    references (canopy_height.labels.prepare_als); coarser (10 m) -> nearest neighbour (a warning when they are not
    on the analysis grid). project['als_resolution'] is only checked against it. Fails when the 99th percentile
    of the reference is above ALS_MAX_M (not metres: e.g. centimetres need als_scale 0.01). Returns the path, None
    without ALS rasters."""
    als = project["als"]
    if not als:
        return None
    paths = [str(p) for p in ([als] if isinstance(als, (str, Path)) else als)]
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"ALS rasters not found: {missing}")
    scale = float(project["als_scale"])
    if not scale > 0:
        raise ValueError(f"als_scale {project['als_scale']!r}: a positive factor to metres (0.01 for centimetres)")
    grid = project.mosaic("annual")
    if not grid.exists():
        if mosaic_layout(project) is None:
            project.log("mosaic: no study-area mosaic (no common grid or no study-area cells), ALS reference skipped")
            return None
        raise FileNotFoundError(f"{grid} missing: the ALS reference is put on the study-area mosaic (build_mosaics)")
    with rasterio.open(grid) as g:
        crs, T = g.crs, g.transform
    res, off = {}, []
    for p in paths:
        with rasterio.open(p) as s:
            res[p] = _res_m(s)
            if res[p] > ALS_FINE_M and not _on_grid(s, crs, T):
                off.append(Path(p).name)
    fine = res[paths[0]] <= ALS_FINE_M
    if any((r <= ALS_FINE_M) != fine for r in res.values()):
        raise ValueError("ALS rasters of mixed resolution (" + ", ".join(f"{Path(p).name} {r:.3g} m" for p, r in
                         res.items()) + f"): give only rasters of <= {ALS_FINE_M:g} m (aggregated to the 10 m p90) "
                         f"or only 10 m rasters")
    lo, hi = min(res.values()), max(res.values())
    size = f"{lo:.3g}" + (f"-{hi:.3g}" if hi > lo * 1.001 else "")
    setting = float(project["als_resolution"])
    if not 0.9 * setting <= lo <= hi <= 1.1 * setting:
        project.log(f"mosaic: WARNING als_resolution is {setting:g} m but the ALS rasters are {size} m: their own "
                    f"resolution is used ({'10 m p90' if fine else 'nearest neighbour'})")
    if off:
        project.log(f"mosaic: WARNING {len(off)} ALS raster(s) not on the 10 m analysis grid, resampled by nearest "
                    f"neighbour (shifts of up to half a pixel): {off[:3]}")
    out = project.mosaic("ALS")
    tmp, tmp2 = out.with_name("ALS.partial.tif"), out.with_name("ALS.scaled.partial.tif")
    try:
        if fine:
            labels.prepare_als(paths, grid_path=grid, out_path=tmp, percentiles=(90,))
        else:
            _warp_nearest(paths, grid, tmp)
        if scale != 1.0:
            _scale_raster(tmp, tmp2, scale)
            replace_retry(tmp2, tmp)
        n, p99 = _finite_stats(tmp)
        if n and p99 > ALS_MAX_M:
            out.unlink(missing_ok=True)
            raise ValueError(f"ALS values do not look like metres: 99th percentile {p99:.4g} (als_scale {scale:g}); "
                             f"for canopy-height rasters in centimetres set als_scale: 0.01 in project.yaml (0.3048 "
                             f"for feet)")
    except BaseException:
        tmp.unlink(missing_ok=True)
        tmp2.unlink(missing_ok=True)
        raise
    replace_retry(tmp, out)
    project.log(f"mosaic: ALS.tif from {len(paths)} raster(s) at {size} m "
                f"({'10 m p90' if fine else 'nearest neighbour'}{f', x {scale:g}' if scale != 1.0 else ''}), "
                f"{n} pixels with ALS" + (f", 99th percentile {p99:.1f} m" if n else ""))
    return out
