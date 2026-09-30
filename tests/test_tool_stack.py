"""tool/stack.py on small synthetic cell rasters (no Earth Engine)."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

rasterio = pytest.importorskip("rasterio")
yaml = pytest.importorskip("yaml")
from rasterio.transform import from_origin  # noqa: E402
from rasterio.warp import transform_geom  # noqa: E402

from canopy_height import channels, labels  # noqa: E402
from canopy_height.tool import stack  # noqa: E402
from canopy_height.tool.project import INPUTS, Project  # noqa: E402

EPSG, GX0, GY1 = 32650, 584670.0, 550110.0
# the paper's pipeline code (optional: the comparisons with it are skipped without it)
PIPELINE = os.environ.get("CHM_PIPELINE_DIR", "no-pipeline")
SEASONAL_PIPELINE = os.environ.get("CHM_SEASONAL_DIR", "no-seasonal-pipeline")
SEASONAL = [f"S1_asc_{k}" for k in range(4)] + [f"S2_{k}" for k in range(4)]
BAND_NAMES = {"S1": ["VV", "VH"], "S2": ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12"]}


def _layer(name, seed):
    g = np.random.default_rng(seed)
    if name.startswith("S1_asc_"):                         # seasonal S1: dB, float64, NaN = no data
        a = g.uniform(-60, 10, (2, 256, 256))
        a[:, 30:33, :] = np.nan
        a[0, 0, :4] = [-50.0, -49.995, 605.35, -12.34567]
    elif name.startswith("S2_"):                           # seasonal S2: 0-1 reflectance (some beyond), float64
        a = g.uniform(-0.1, 7.0, (9, 256, 256))
        a[:, :, 40:42] = np.nan
        a[0, 0, :4] = [0.12345678, 6.5535, 6.55359, -0.00001]
    elif name == "Embedding":
        a = g.uniform(-1.2, 1.2, (64, 256, 256))
        a[:, :4, :4] = np.nan
    elif name == "DEM":
        a = g.integers(-5, 800, (1, 256, 256)).astype(np.int16)
    elif name == "S1":
        a = g.uniform(-60, 10, (2, 256, 256))
        a[:, 10:12, :] = np.nan
    elif name == "S2":
        a = g.uniform(0, 70000, (9, 256, 256))
        a[:, :, 100:103] = np.nan
    elif name == "GEDI":
        a = np.full((1, 256, 256), np.nan)
        k = g.integers(0, 256, (2, 300))
        a[0, k[0], k[1]] = g.uniform(0, 60, 300)
        a[0, 0, :5] = -9999
    elif name in ("ETH", "UMD"):
        a = g.integers(0, 50, (1, 256, 256)).astype(np.uint8)
    else:                                                  # GMTCH
        a = g.uniform(0, 40, (1, 256, 256)).astype(np.float32)
        a[0, :20] = np.nan
    return a


def _write(path, a, x0, y1, epsg=EPSG, res=10.0, descriptions=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nodata = np.nan if a.dtype.kind == "f" else None
    with rasterio.open(path, "w", driver="GTiff", width=a.shape[2], height=a.shape[1], count=a.shape[0],
                       dtype=a.dtype.name, crs=f"EPSG:{epsg}", transform=from_origin(x0, y1, res, res),
                       nodata=nodata) as d:
        d.write(a)
        for i, s in enumerate(descriptions or [], 1):
            d.set_band_description(i, s)


def _cell(r, c, role="ring", use=True, dist=0.0, **extra):
    x0, y1 = GX0 + c * 2560, GY1 - r * 2560
    return dict(cell=f"x{round(x0)}y{round(y1)}", x0=x0, y1=y1, col=c, row=r, role=role, dist_m=dist,
                water=np.nan, built=np.nan, tree=np.nan, use=use, **extra)


def _fill(project, cell, layers, seed, folder=None):
    """Write synthetic rasters of a cell (raw/ or `folder`); returns {layer: array as written}."""
    out = {}
    for i, name in enumerate(layers):
        a = _layer(name, seed * 100 + i)
        d = Path(folder) if folder else project.path("raw")
        _write(d / f"{cell['cell']}_{name}.tif", a, cell["x0"], cell["y1"],
               descriptions=BAND_NAMES[name[:2]] if name in SEASONAL else None)   # as the paper's seasonal rasters
        out[name] = a
    return out


def _project(root, cells, grid=True, aoi=None, **cfg):
    root.mkdir(parents=True, exist_ok=True)
    (root / "project.yaml").write_text(yaml.safe_dump(cfg) if cfg else "{}\n", encoding="utf-8")
    (root / "plan").mkdir(exist_ok=True)
    pd.DataFrame(cells).to_csv(root / "plan" / "cells.csv", index=False)
    if grid:
        json.dump(dict(epsg=EPSG, x0=GX0, y1=GY1, res=10.0, cell_px=256), open(root / "plan" / "grid.json", "w"))
    if aoi is not None:
        g = transform_geom(f"EPSG:{EPSG}", "EPSG:4326", aoi)
        json.dump(dict(type="FeatureCollection", features=[dict(type="Feature", properties={}, geometry=g)]),
                  open(root / "aoi.geojson", "w"))
    return Project(root)


def _expected_x(a):
    emb, dem, s1, s2 = (a[k].astype(np.float64) for k in stack.X_LAYERS)
    x = np.concatenate([(emb + 1) * 1e4, dem, (s1 + 50) * 100, s2])
    return np.clip(np.nan_to_num(x), 0, 65535).astype(np.uint16)


def _expected_gedi(a):
    g = a["GEDI"][0].astype(np.float32)
    return np.where(np.isnan(g) | (g == -9999), np.float32(-999), g)


X5 = ["Embedding", "DEM", "S1", "S2", "GEDI"]


# ------------------------------------------------------------------------------------------------ encodings
def test_encodings():
    emb = np.array([[[np.nan, -1.0, -0.99995, 0.5, 5.6]]])
    dem = np.array([[[-3.0, 0.0, 123.0, 7.0, 70000.0]]])
    s1 = np.array([[[-60.0, -12.345, np.nan, 0.0, 700.0]]])
    s2 = np.array([[[65535.9, 1234.9, np.nan, -1.0, 0.0]]])
    x = stack.encode_x(emb, dem, s1, s2)
    assert x.dtype == np.uint16 and x.shape == (4, 1, 5)
    assert x[0, 0].tolist() == [0, 0, 0, 15000, 65535]           # NaN -> 0, truncate, clip
    assert x[1, 0].tolist() == [0, 0, 123, 7, 65535]
    assert x[2, 0].tolist() == [0, 3765, 0, 5000, 65535]
    assert x[3, 0].tolist() == [65535, 1234, 0, 0, 0]
    g = stack.enc999(np.array([np.nan, -9999.0, 3.5, 0.0]))
    assert g.dtype == np.float32 and g.tolist() == [-999, -999, 3.5, 0]


@pytest.mark.skipif(not os.path.exists(os.path.join(PIPELINE, "common.py")), reason="pipeline code not available")
def test_matches_pipeline_common(tmp_path):
    sys.path.insert(0, PIPELINE)
    try:
        import common as U
    finally:
        sys.path.remove(PIPELINE)
    cell = _cell(0, 0, "aoi")
    p = _project(tmp_path / "p", [cell])
    a = _fill(p, cell, X5, 1)
    f = {k: str(p.raw(cell["cell"], k)) for k in X5}
    ref_x = U.encode_x(*(U.read_tile(f[k], 0, "reflect") for k in stack.X_LAYERS))
    ref_g = U.enc999(U.read_tile(f["GEDI"], 0, "reflect", bands=1)[0])
    x, _ = stack.cell_x(p, cell)
    g = stack.enc999(stack.read_cell_layer(f["GEDI"], cell, 1)[0])
    assert np.array_equal(x, ref_x) and np.array_equal(g, ref_g) and np.array_equal(x, _expected_x(a))


# ------------------------------------------------------------------------------------------------ training stack
def test_build_train_stack(tmp_path, monkeypatch):
    monkeypatch.setattr(stack, "PART", 2)
    ext = tmp_path / "pipeline_inputs"
    cells = [_cell(0, 0, "aoi"), _cell(0, 1, "aoi"), _cell(1, 1, "ring", use=False, dist=0.0),
             _cell(-1, 0, "ring", dist=0.0), _cell(2, 0, "ring", dist=1000.0)]
    cells[4].update(src_dir=str(ext / "x"), lab_dir=str(ext / "lab"), epsg=EPSG)   # an imported cell
    p = _project(tmp_path / "p", cells)
    data = {}
    for i, c in enumerate(cells):
        if not c["use"]:
            continue
        if "src_dir" in c:
            data[c["cell"]] = {**_fill(p, c, stack.X_LAYERS, i, c["src_dir"]),
                               **_fill(p, c, ["GEDI"], i + 50, c["lab_dir"])}
        else:
            data[c["cell"]] = _fill(p, c, X5, i)
    ix = stack.build_train_stack(p)
    used = [c for c in cells if c["use"]]
    assert ix.cell.tolist() == [c["cell"] for c in used]
    assert ix.part.tolist() == [1, 1, 2, 2] and ix.row_in_part.tolist() == [0, 1, 0, 1]
    assert ix.role.tolist() == ["aoi", "aoi", "ring", "ring"] and (ix.epsg == EPSG).all()
    xs, gs = p.stack_files("x"), p.stack_files("GEDI")
    assert [os.path.basename(f) for f in xs] == ["train_part001.npy", "train_part002.npy"]
    X = np.concatenate([np.load(f) for f in xs])
    G = np.concatenate([np.load(f) for f in gs])
    assert X.dtype == np.uint16 and X.shape == (4, 76, 256, 256) and G.dtype == np.float32 and G.shape == (4, 256, 256)
    for j, c in enumerate(used):
        assert np.array_equal(X[j], _expected_x(data[c["cell"]]))
        assert np.array_equal(G[j], _expected_gedi(data[c["cell"]]))
        assert ix.gedi_px[j] == int((G[j] > -999).sum())
    h = json.load(open(p.path("stacks", "hashes.json")))
    assert h["cells_sha256"] == stack.file_sha256(p.cells_file) and h["chips"] == 4
    for f in xs + gs:
        a = np.load(f)
        assert h["files"][os.path.basename(f)] == dict(shape=list(a.shape), dtype=str(a.dtype),
                                                       data_sha256=stack.data_sha256(a))
    assert not list(p.path("stacks").glob("*.tmp.*"))

    # unchanged cells.csv -> skipped
    t = {f: os.path.getmtime(f) for f in xs}
    stack.build_train_stack(p)
    assert {f: os.path.getmtime(f) for f in xs} == t
    assert "up to date" in p.log_file.read_text(encoding="utf-8")

    # changed cells.csv -> rebuilt, stale parts removed
    df = pd.read_csv(p.cells_file)
    df.loc[df.index[-2:], "use"] = False
    df.to_csv(p.cells_file, index=False)
    ix = stack.build_train_stack(p)
    assert len(ix) == 2 and [os.path.basename(f) for f in p.stack_files("x")] == ["train_part001.npy"]
    assert [os.path.basename(f) for f in p.stack_files("GEDI")] == ["train_GEDI_part001.npy"]
    assert np.array_equal(np.load(p.stack_files("x")[0]), X[:2])


def _log(p):
    return p.log_file.read_text(encoding="utf-8")


def _mtimes(p):
    return {f: os.stat(f).st_mtime_ns for f in p.stack_files("x") + p.stack_files("GEDI")}


def test_stack_rebuilt_on_changed_inputs_and_settings(tmp_path):
    cells = [_cell(0, 0, "aoi"), _cell(0, 1, "aoi")]
    p = _project(tmp_path / "p", cells)
    for i, c in enumerate(cells):
        _fill(p, c, X5, i)
    stack.build_train_stack(p)
    hf = p.path("stacks", "hashes.json")
    h = json.load(open(hf))
    assert h["settings"] == dict(year=2020, gedi_window=["2019-01-01", "2021-12-31"], built_up_mask=True, input="AE")
    assert list(h["inputs"]) == [f"raw/{c['cell']}_{k}.tif" for c in cells for k in X5]
    assert h["inputs"]["raw/" + cells[0]["cell"] + "_GEDI.tif"][0] == os.path.getsize(p.raw(cells[0]["cell"], "GEDI"))
    sid = stack.stack_id(p)
    assert sid == hashlib.sha256(json.dumps(h["files"], sort_keys=True).encode()).hexdigest()
    assert stack.stack_outdated(p) == ""

    # re-downloaded labels -> rebuilt with them
    g = _layer("GEDI", 77)
    _write(p.raw(cells[1]["cell"], "GEDI"), g, cells[1]["x0"], cells[1]["y1"])
    assert stack.stack_outdated(p).startswith("1 input rasters changed")
    stack.build_train_stack(p)
    assert np.array_equal(np.load(p.stack_files("GEDI")[0])[1], _expected_gedi({"GEDI": g}))
    assert stack.stack_id(p) != sid and "rebuilding the training stack (1 input rasters changed" in _log(p)

    # changed settings -> rebuilt
    p.cfg["gedi_window"] = ["2019-01-01", "2020-01-01"]
    assert "gedi_window" in stack.stack_outdated(p)
    t = _mtimes(p)
    stack.build_train_stack(p)
    assert _mtimes(p) != t and json.load(open(hf))["settings"]["gedi_window"] == ["2019-01-01", "2020-01-01"]
    assert stack.stack_outdated(p) == ""

    # hashes.json of an earlier version (no settings / inputs): adopted when no input raster is newer ...
    h = json.load(open(hf))
    old = {k: v for k, v in h.items() if k not in ("settings", "inputs")}
    json.dump(old, open(hf, "w"))
    t, sid = _mtimes(p), stack.stack_id(p)
    stack.build_train_stack(p)
    assert _mtimes(p) == t and stack.stack_id(p) == sid and json.load(open(hf)) == h
    assert "stack of an earlier version" in _log(p)
    # ... else rebuilt
    json.dump(old, open(hf, "w"))
    early = min(v for f in p.path("raw").iterdir() for v in [f.stat().st_mtime_ns]) - 10 ** 9
    os.utime(hf, ns=(early, early))
    assert stack.stack_outdated(p) == "input rasters newer than the stack"
    stack.build_train_stack(p)
    assert _mtimes(p) != t and "inputs" in json.load(open(hf))


def test_stack_built_in_a_scratch_folder(tmp_path, monkeypatch):
    """colab.run builds the memmaps on the local disk (CHM_STACK_SCRATCH) and copies the parts into stacks/: same
    files, nothing left behind in either folder."""
    monkeypatch.setattr(stack, "PART", 1)
    cells = [_cell(0, c, "aoi") for c in range(2)]
    ref = _project(tmp_path / "ref", cells)
    for i, c in enumerate(cells):
        _fill(ref, c, X5, i)
    stack.build_train_stack(ref)
    p = _project(tmp_path / "p", cells)
    for i, c in enumerate(cells):
        _fill(p, c, X5, i)
    scratch = tmp_path / "local_disk"
    monkeypatch.setenv(stack.SCRATCH_ENV, str(scratch))
    stack.build_train_stack(p)
    assert [os.path.basename(f) for f in p.stack_files("x")] == ["train_part001.npy", "train_part002.npy"]
    for a, b in zip(ref.stack_files("x") + ref.stack_files("GEDI"), p.stack_files("x") + p.stack_files("GEDI")):
        assert np.array_equal(np.load(a), np.load(b))
    assert json.load(open(p.path("stacks", "hashes.json")))["files"] == \
        json.load(open(ref.path("stacks", "hashes.json")))["files"]
    assert not list(scratch.iterdir()) and not list(p.path("stacks").glob("*.tmp.*"))
    assert stack.stack_id(p) == stack.stack_id(ref)


def test_failed_stack_leaves_no_parts(tmp_path, monkeypatch):
    monkeypatch.setattr(stack, "PART", 1)
    cells = [_cell(0, c, "aoi") for c in range(3)]
    p = _project(tmp_path / "p", cells)
    for i, c in enumerate(cells):
        _fill(p, c, X5, i)
    stack.build_train_stack(p)
    assert len(p.stack_files("x")) == 3 and stack.stack_id(p)
    _write(p.raw(cells[2]["cell"], "S2"), _layer("S2", 9)[:, :255], cells[2]["x0"], cells[2]["y1"])
    with pytest.raises(ValueError, match="whole"):
        stack.build_train_stack(p)                      # parts 1 and 2 were written before part 3 failed
    assert not list(p.path("stacks").iterdir()) and stack.stack_id(p) is None
    assert stack.model_is_current(p, "unet-sls") == (False, "the training stack is incomplete (run the stack stage)")


def test_stack_id_and_model_is_current(tmp_path):
    cells = [_cell(0, c, "aoi") for c in range(3)]
    p = _project(tmp_path / "p", cells)
    for i, c in enumerate(cells):
        _fill(p, c, X5, i)
    stack.build_train_stack(p)
    sid = stack.stack_id(p)
    d = p.model_dir("kg-unet1")
    d.mkdir(parents=True)
    ok, why = stack.model_is_current(p, "kg-unet1")
    assert not ok and "result.json" in why
    (d / "stack_id.txt").write_text(sid + "\n")
    assert stack.model_is_current(p, "kg-unet1") == (True, "")
    (d / "stack_id.txt").write_text("0" * 64)
    ok, why = stack.model_is_current(p, "kg-unet1")
    assert not ok and "another training stack" in why

    # runs without stack_id.txt: result.json against the current stack
    (d / "stack_id.txt").unlink()
    res = dict(annual=[str(Path(r"C:\elsewhere") / "train_part001.npy")], labels=p.stack_files("GEDI"), n_chips=3)
    json.dump(res, open(d / "result.json", "w"))
    assert stack.model_is_current(p, "kg-unet1") == (True, "")
    json.dump(dict(res, n_chips=2), open(d / "result.json", "w"))
    ok, why = stack.model_is_current(p, "kg-unet1")
    assert not ok and "trained on 2 chips in 1 part(s), the current stack has 3 chips" in why
    r = p.model_dir("rf-sls")
    r.mkdir(parents=True)
    n = int((np.load(p.stack_files("GEDI")[0]) > 0).sum())
    json.dump(dict(model="rf-sls", n_rows=n), open(r / "result.json", "w"))
    assert stack.model_is_current(p, "rf-sls") == (True, "")
    json.dump(dict(model="rf-sls", n_rows=n - 1), open(r / "result.json", "w"))
    assert stack.model_is_current(p, "rf-sls")[0] is False

    # a stray part or a missing hashes.json: incomplete
    np.save(p.stack_part("x", 9), np.zeros(1, np.uint16))
    assert stack.stack_id(p) is None and stack.stack_outdated(p) == "stack parts missing or left over"
    stack.build_train_stack(p)
    assert stack.stack_id(p) == sid and not p.stack_part("x", 9).exists()
    p.path("stacks", "hashes.json").unlink()
    assert stack.stack_id(p) is None and stack.model_is_current(p, "rf-sls")[0] is False


def test_stack_rejects_bad_rasters(tmp_path):
    c = _cell(0, 0, "aoi")
    p = _project(tmp_path / "p", [c])
    with pytest.raises(FileNotFoundError):
        stack.build_train_stack(p)
    _fill(p, c, X5, 3)
    _write(p.raw(c["cell"], "S2"), _layer("S2", 9), c["x0"] + 10, c["y1"])        # shifted by one pixel
    with pytest.raises(ValueError, match="grid"):
        stack.build_train_stack(p)
    _write(p.raw(c["cell"], "S2"), _layer("S2", 9)[:, :255], c["x0"], c["y1"])    # not a whole cell
    with pytest.raises(ValueError, match="whole"):
        stack.build_train_stack(p)
    _write(p.raw(c["cell"], "S2"), _layer("S2", 9), c["x0"], c["y1"], epsg=32649)   # other CRS
    with pytest.raises(ValueError, match="EPSG"):
        stack.build_train_stack(p)
    assert not p.stack_files("x") and not list(p.path("stacks").glob("*.npy"))


# ------------------------------------------------------------------------------------------------ mosaics
def _l_shape():
    """Study area: row 0 of three cells + column 0 of row 1 (vertices 3 m inside the cells, off pixel centres)."""
    x, y = GX0, GY1
    return dict(type="Polygon", coordinates=[[(x + 3, y - 3), (x + 7677, y - 3), (x + 7677, y - 2553),
                                              (x + 2553, y - 2553), (x + 2553, y - 5117), (x + 3, y - 5117),
                                              (x + 3, y - 3)]])


def _mosaic_project(root, **cfg):
    cells = [_cell(0, 0, "aoi"), _cell(0, 1, "aoi"), _cell(0, 2, "aoi"), _cell(1, 0, "aoi"),
             _cell(1, 1, "ring"),                        # inside the box, with inputs -> filled in
             _cell(1, 2, "ring", use=False),              # inside the box, not downloaded -> no data
             _cell(0, -1, "ring", dist=1.0)]              # outside the box
    p = _project(root, cells, aoi=_l_shape(), **cfg)
    data = {}
    for i, c in enumerate(cells):
        if c["use"]:
            data[c["cell"]] = _fill(p, c, X5, i)
    data[cells[0]["cell"]].update(_fill(p, cells[0], ["ETH", "UMD"], 20))
    _fill(p, cells[1], ["ETH"], 21)
    _fill(p, cells[4], ["ETH"], 22)                      # ring cell: never used for a benchmark
    return p, cells, data


def test_build_mosaics(tmp_path):
    p, cells, data = _mosaic_project(tmp_path / "p")
    ix = stack.build_train_stack(p)
    chips = np.load(p.stack_files("x")[0])
    gchips = np.load(p.stack_files("GEDI")[0])
    written = stack.build_mosaics(p)
    assert [f.name for f in written] == ["annual.tif", "GEDI.tif", "GFCH.tif", "HRCH.tif", "aoi_mask.tif"]
    assert not p.mosaic("GMTCH").exists() and not list(p.path("mosaic").glob("*.partial.tif"))
    with rasterio.open(p.mosaic("annual")) as s:
        assert (s.count, s.height, s.width, s.dtypes[0], s.nodata) == (76, 512, 768, "uint16", 0)
        assert s.crs.to_epsg() == EPSG and tuple(s.transform)[:6] == (10.0, 0.0, GX0, 0.0, -10.0, GY1)
        assert list(s.descriptions) == channels.ANNUAL_BANDS
        A = s.read()
    with rasterio.open(p.mosaic("GEDI")) as s:
        assert s.dtypes[0] == "float32" and np.isnan(s.nodata)
        Gm = s.read(1)
    for c in cells[:5]:
        win = np.s_[c["row"] * 256:(c["row"] + 1) * 256, c["col"] * 256:(c["col"] + 1) * 256]
        j = ix.index[ix.cell == c["cell"]][0]
        assert np.array_equal(A[(slice(None),) + win], chips[j])
        assert np.array_equal(np.where(np.isnan(Gm[win]), np.float32(-999), Gm[win]), gchips[j])
    assert (A[:, 256:, 512:] == 0).all() and np.isnan(Gm[256:, 512:]).all()
    with rasterio.open(p.mosaic("HRCH")) as s:
        H = s.read(1)
    eth0 = data[cells[0]["cell"]]["ETH"][0].astype(np.float32)
    assert np.array_equal(H[:256, :256], eth0) and (H[:256, :256] == 0).any()      # 0 kept as 0
    assert np.isfinite(H[:256, 256:512]).all()
    assert np.isnan(H[:256, 512:]).all() and np.isnan(H[256:, :]).all()           # no ETH / ring cell
    with rasterio.open(p.mosaic("GFCH")) as s:
        U = s.read(1)
    assert np.array_equal(U[:256, :256], data[cells[0]["cell"]]["UMD"][0].astype(np.float32))
    assert np.isnan(U[:, 256:]).all() and np.isnan(U[256:, :]).all()
    with rasterio.open(p.mosaic("aoi_mask")) as s:
        assert s.dtypes[0] == "uint8"
        m = s.read(1)
    exp = np.zeros((512, 768), np.uint8)
    exp[:255, :] = 1                                   # centres 5 + 10 i < 2553
    exp[:, :255] = 1
    assert np.array_equal(m, exp)


def _inner_box(r, c, lo=3, hi=2553):
    x, y = GX0 + c * 2560, GY1 - r * 2560
    return [(x + lo, y - lo), (x + hi, y - lo), (x + hi, y - hi), (x + lo, y - hi), (x + lo, y - lo)]


def test_mosaics_of_parts_far_apart_are_sparse(tmp_path):
    """Two study-area cells 77 x 102 km apart: a 31 x 41-cell box (12.7 GB as dense uint16) of which only the two
    cells are stored."""
    a, b = _cell(0, 0, "aoi"), _cell(30, 40, "aoi")
    aoi = dict(type="MultiPolygon", coordinates=[[_inner_box(0, 0)], [_inner_box(30, 40)]])
    p = _project(tmp_path / "p", [a, b], aoi=aoi, benchmarks=["HRCH"])
    data = {c["cell"]: {**_fill(p, c, X5, i), **_fill(p, c, ["ETH"], i + 10)} for i, c in enumerate([a, b])}
    stack.build_mosaics(p)
    assert "WARNING the study-area parts are far apart: 2 planned cells in a box of 31 x 41 cells" in _log(p)
    assert os.path.getsize(p.mosaic("annual")) < 40e6 and os.path.getsize(p.mosaic("aoi_mask")) < 1e6
    with rasterio.open(p.mosaic("annual")) as s, rasterio.open(p.mosaic("GEDI")) as g, \
            rasterio.open(p.mosaic("HRCH")) as e, rasterio.open(p.mosaic("aoi_mask")) as m:
        assert (s.height, s.width) == (31 * 256, 41 * 256) and m.dtypes[0] == "uint8"
        for c in (a, b):
            w = stack.Window(c["col"] * 256, c["row"] * 256, 256, 256)
            assert np.array_equal(s.read(window=w), _expected_x(data[c["cell"]]))
            gd = g.read(1, window=w)
            assert np.array_equal(np.where(np.isnan(gd), np.float32(-999), gd), _expected_gedi(data[c["cell"]]))
            assert np.array_equal(e.read(1, window=w), data[c["cell"]]["ETH"][0].astype(np.float32))
            mk = m.read(1, window=w)
            assert mk[:255, :255].all() and not mk[255:].any() and not mk[:, 255:].any()
        w = stack.Window(20 * 256, 10 * 256, 256, 256)                     # an empty cell: nodata
        assert not s.read(window=w).any() and np.isnan(g.read(1, window=w)).all()
        assert np.isnan(e.read(1, window=w)).all() and not m.read(1, window=w).any()
        assert int(sum(m.read(1, window=wd).sum() for _, wd in m.block_windows(1))) == 2 * 255 * 255


def test_mosaics_refuse_a_study_area_without_pixels(tmp_path):
    c = _cell(0, 0, "aoi")
    tiny = dict(type="Polygon", coordinates=[_inner_box(0, 0, 1, 4)])      # 3 m wide: no pixel centre inside
    p = _project(tmp_path / "p", [c], aoi=tiny)
    _fill(p, c, X5, 1)
    with pytest.raises(ValueError, match="covers no 10 m pixel centre of the 1 study-area cells"):
        stack.build_mosaics(p)
    assert not list(p.path("mosaic").iterdir())


def test_mosaics_need_grid_and_aoi_cells(tmp_path):
    chm = tmp_path / "chm.tif"
    _write(chm, np.ones((1, 10, 10), np.float32), GX0, GY1)
    c = _cell(0, 0, "ring")
    p = _project(tmp_path / "a", [c], grid=False, als=[str(chm)], als_resolution=10)
    assert stack.build_mosaics(p) == [] and stack.als_reference(p) is None
    p = _project(tmp_path / "b", [c], grid=True)
    assert stack.build_mosaics(p) == [] and stack.als_reference(p) is None
    a = _cell(0, 0, "aoi")
    p = _project(tmp_path / "c", [a], grid=False, als=[str(chm)], als_resolution=10)
    json.dump(dict(epsg=EPSG, x0=None, y1=None, res=10.0, cell_px=256, cells_on_common_grid=False),
              open(p.grid_file, "w"))                                   # grid.import_cells of scattered cells
    assert stack.read_grid(p) is None and stack.build_mosaics(p) == [] and stack.als_reference(p) is None
    p = _project(tmp_path / "d", [a], grid=True, als=[str(tmp_path / "missing.tif")])
    with pytest.raises(FileNotFoundError):
        stack.als_reference(p)
    p = _project(tmp_path / "e", [a], grid=True, als=[str(chm)], als_resolution=10)
    with pytest.raises(FileNotFoundError, match="build_mosaics"):
        stack.als_reference(p)


def test_als_reference_10m(tmp_path):
    chm1 = np.random.default_rng(5).uniform(0, 50, (1, 200, 300)).astype(np.float32)
    chm1[0, :10, :10] = np.nan
    chm2 = np.full((1, 100, 100), 77.0, np.float32)
    _write(tmp_path / "chm1.tif", chm1, GX0 + 100, GY1 - 50)
    _write(tmp_path / "chm2.tif", chm2, GX0 + 1000, GY1 - 1000)    # overlaps chm1: chm1 wins where it has data
    p, cells, _ = _mosaic_project(tmp_path / "p", als=[str(tmp_path / "chm1.tif"), str(tmp_path / "chm2.tif")],
                                  als_resolution=10)
    stack.build_mosaics(p)
    out = stack.als_reference(p)
    with rasterio.open(out) as s:
        assert (s.height, s.width, s.dtypes[0]) == (512, 768, "float32") and np.isnan(s.nodata)
        a = s.read(1)
    exp = np.full((512, 768), np.nan, np.float32)
    exp[100:200, 100:200] = 77.0
    fp = np.s_[5:205, 10:310]
    exp[fp] = np.where(np.isnan(chm1[0]) & ~np.isnan(exp[fp]), exp[fp], chm1[0])
    assert np.array_equal(a, exp, equal_nan=True)


def test_als_reference_1m(tmp_path):
    g = np.random.default_rng(6)
    chm = g.uniform(0, 40, (1, 60, 80)).astype(np.float32)
    chm[0, :10, :10] = np.nan                              # one empty 10 m cell
    chm[0, 20:25, 30:40] = np.nan                          # one half-empty cell
    _write(tmp_path / "chm1m.tif", chm, GX0 + 20, GY1 - 30, res=1.0)
    p, _, _ = _mosaic_project(tmp_path / "p", als=[str(tmp_path / "chm1m.tif")], als_resolution=1)
    stack.build_mosaics(p)
    with rasterio.open(stack.als_reference(p)) as s:
        a = s.read(1)
    exp = labels.block_percentile(chm[0], 10, 90)
    assert np.array_equal(a[3:9, 2:10], exp, equal_nan=True) and np.isnan(a[3, 2])
    a[3:9, 2:10] = np.nan
    assert np.isnan(a).all()


def test_als_method_follows_the_rasters(tmp_path):
    """The rasters' own resolution decides (p90 for <= 2 m, nearest for 10 m); als_resolution is only checked."""
    g = np.random.default_rng(7)
    chm1 = g.uniform(0, 40, (1, 60, 80)).astype(np.float32)
    _write(tmp_path / "chm1m.tif", chm1, GX0 + 20, GY1 - 30, res=1.0)
    chm10 = g.uniform(0, 40, (1, 20, 30)).astype(np.float32)
    _write(tmp_path / "chm10m.tif", chm10, GX0 + 100, GY1 - 50)
    _write(tmp_path / "shifted.tif", chm10, GX0 + 105, GY1 - 50)
    p, _, _ = _mosaic_project(tmp_path / "p", als=[str(tmp_path / "chm1m.tif")], als_resolution=10)
    stack.build_mosaics(p)
    with rasterio.open(stack.als_reference(p)) as s:
        assert np.array_equal(s.read(1)[3:9, 2:10], labels.block_percentile(chm1[0], 10, 90), equal_nan=True)
    assert "WARNING als_resolution is 10 m but the ALS rasters are 1 m: their own resolution is used (10 m p90)" \
        in _log(p)

    p.cfg.update(als=[str(tmp_path / "chm10m.tif")], als_resolution=1.0)
    with rasterio.open(stack.als_reference(p)) as s:
        assert np.array_equal(s.read(1)[5:25, 10:40], chm10[0])
    assert "(nearest neighbour)" in _log(p) and "not on the 10 m analysis grid" not in _log(p)
    p.cfg.update(als=[str(tmp_path / "shifted.tif")], als_resolution=10)
    stack.als_reference(p)
    assert "WARNING 1 ALS raster(s) not on the 10 m analysis grid" in _log(p)
    p.cfg.update(als=[str(tmp_path / "chm1m.tif"), str(tmp_path / "chm10m.tif")])
    with pytest.raises(ValueError, match="mixed resolution"):
        stack.als_reference(p)


def test_als_units(tmp_path):
    cm = np.random.default_rng(8).uniform(0, 4000, (1, 20, 30)).astype(np.float32)     # a CHM in centimetres
    _write(tmp_path / "chm_cm.tif", cm, GX0 + 100, GY1 - 50)
    p, _, _ = _mosaic_project(tmp_path / "p", als=[str(tmp_path / "chm_cm.tif")], als_resolution=10)
    stack.build_mosaics(p)
    with pytest.raises(ValueError, match="do not look like metres.*als_scale: 0.01"):
        stack.als_reference(p)
    assert not list(p.path("mosaic").glob("ALS*"))
    p.cfg["als_scale"] = 0.01
    with rasterio.open(stack.als_reference(p)) as s:
        a = s.read(1)
    assert np.array_equal(a[5:25, 10:40], cm[0] * np.float32(0.01)) and np.isnan(a[:5]).all()
    assert "x 0.01), 600 pixels with ALS" in _log(p)
    p.cfg["als_scale"] = 0
    with pytest.raises(ValueError, match="positive factor"):
        stack.als_reference(p)


# ------------------------------------------------------------------------------------------------ input representations
ANNUAL_CH = {"Embedding": slice(0, 64), "DEM": slice(64, 65), "S1": slice(65, 67), "S2": slice(67, 76)}


def _expected_seasonal(a):
    """44-band seasonal chip as make_outside_seasonal.encode: S1 (dB+50)*100, S2 *10000, NaN->0, clip, truncate."""
    out = []
    for k in SEASONAL:
        v = a[k].astype(np.float64)
        v = (v + 50.0) * 100.0 if k.startswith("S1") else v * 10000.0
        out.append(np.clip(np.nan_to_num(v, nan=0.0), 0, 65535).astype(np.uint16))
    return np.concatenate(out)


def _expected_x_of(a, inp):
    """76-band chip of an input representation: the paper's encoding of its annual layers, 0 elsewhere."""
    x = np.zeros((76, 256, 256), np.uint16)
    full = _expected_x({k: a.get(k, _layer(k, 0)) for k in stack.X_LAYERS})
    for k in INPUTS[inp]["annual"]:
        x[ANNUAL_CH[k]] = full[ANNUAL_CH[k]]
    return x


def test_encode_seasonal():
    s1 = np.array([[[np.nan, -50.0, -49.993, 700.0, -12.34567]], [[0.0, -60.0, 10.0, 1e9, -0.004]]])
    base = np.array([np.nan, 0.12345678, 6.55349, 0.00019, -0.00001])
    s2 = np.stack([base * (b + 1) for b in range(9)])[:, None, :]
    arr = {k: (s1 if k.startswith("S1") else s2) for k in SEASONAL}
    x = stack.encode_seasonal(arr)
    assert x.dtype == np.uint16 and x.shape == (44, 1, 5)
    assert x[0, 0].tolist() == [0, 0, 0, 65535, 3765]                     # NaN -> 0, (dB+50)*100 truncated, clip
    assert x[1, 0].tolist() == [5000, 0, 6000, 65535, 4999]               # 4999.6 -> 4999 (truncate, not round)
    assert x[8, 0].tolist() == [0, 1234, 65534, 1, 0]                     # S2 x 1e4
    assert x[9, 0].tolist() == [0, 2469, 65535, 3, 0]
    assert all(np.array_equal(x[k], x[k % 2] if k < 8 else x[8 + (k - 8) % 9]) for k in range(44))
    assert np.array_equal(x, stack.encode_seasonal([arr[k] for k in SEASONAL]))    # sequence in layer order
    assert len(channels.SEASONAL_BANDS) == 44 and channels.SEASONAL_BANDS[:2] == ["S1_VV_DJF", "S1_VH_DJF"]
    assert channels.SEASONAL_BANDS[8] == "S2_B2_DJF" and channels.SEASONAL_BANDS[-1] == "S2_B12_SON"
    with pytest.raises(ValueError, match="S2_0: 8 bands"):
        stack.encode_seasonal({**arr, "S2_0": s2[:8]})


@pytest.mark.skipif(not os.path.exists(os.path.join(SEASONAL_PIPELINE, "make_outside_seasonal.py")),
                    reason="seasonal pipeline code not available")
def test_encode_seasonal_matches_paper_pipeline():
    sys.path.insert(0, SEASONAL_PIPELINE)
    try:
        import make_outside_seasonal as M
    finally:
        sys.path.remove(SEASONAL_PIPELINE)
    assert M.LAYERS == stack.SEASONAL_LAYERS and M.BANDS == stack.SEASONAL_BANDS and M.NB == 44
    a = {k: _layer(k, 40 + i) for i, k in enumerate(SEASONAL)}
    ref = np.concatenate([M.encode(k, a[k]) for k in M.LAYERS])
    assert np.array_equal(stack.encode_seasonal(a), ref) and np.array_equal(ref, _expected_seasonal(a))


def _needed_layers(inp):
    return INPUTS[inp]["annual"] + (SEASONAL if INPUTS[inp]["seasonal"] else []) + ["GEDI"]


def test_seasonal_training_stack_T_and_TE(tmp_path, monkeypatch):
    """T: DEM + seasonal (no Embedding / S1 / S2 downloaded); then TE: the Embedding added. Unused annual channels
    are 0 and never read; the seasonal parts follow the same part split; a changed input rebuilds the stack."""
    monkeypatch.setattr(stack, "PART", 2)
    cells = [_cell(0, 0, "aoi"), _cell(0, 1, "aoi"), _cell(1, 0, "ring", dist=10.0)]
    p = _project(tmp_path / "p", cells, input="T")
    assert p.seasonal and p["input"] == "T"
    data = {c["cell"]: _fill(p, c, _needed_layers("T"), i) for i, c in enumerate(cells)}
    # an annual layer T does not use, present but unreadable as a cell: never read
    _write(p.raw(cells[0]["cell"], "S2"), _layer("S2", 9)[:, :255], cells[0]["x0"], cells[0]["y1"])
    ix = stack.build_train_stack(p)
    assert ix.cell.tolist() == [c["cell"] for c in cells] and ix.part.tolist() == [1, 1, 2]
    xs, ss, gs = p.stack_files("x"), p.stack_files("seasonal"), p.stack_files("GEDI")
    assert [os.path.basename(f) for f in ss] == ["train_seasonal_part001.npy", "train_seasonal_part002.npy"]
    assert [os.path.basename(f) for f in xs] == ["train_part001.npy", "train_part002.npy"]
    X, S = np.concatenate([np.load(f) for f in xs]), np.concatenate([np.load(f) for f in ss])
    G = np.concatenate([np.load(f) for f in gs])
    assert [np.load(f).shape for f in ss] == [(2, 44, 256, 256), (1, 44, 256, 256)] and S.dtype == np.uint16
    assert X.shape == (3, 76, 256, 256) and X.dtype == np.uint16
    for j, c in enumerate(cells):
        a = data[c["cell"]]
        assert np.array_equal(S[j], _expected_seasonal(a)) and np.array_equal(G[j], _expected_gedi(a))
        assert np.array_equal(X[j], _expected_x_of(a, "T"))
        assert not X[j, :64].any() and not X[j, 65:].any() and X[j, 64].any()   # only the DEM channel
    h = json.load(open(p.path("stacks", "hashes.json")))
    assert h["settings"]["input"] == "T" and h["chips"] == 3
    assert set(h["files"]) == {os.path.basename(f) for f in xs + ss + gs}
    for f in ss:
        a = np.load(f)
        assert h["files"][os.path.basename(f)] == dict(shape=list(a.shape), dtype="uint16",
                                                       data_sha256=stack.data_sha256(a))
    assert list(h["inputs"]) == [f"raw/{c['cell']}_{k}.tif" for c in cells for k in _needed_layers("T")]
    assert not list(p.path("stacks").glob("*.tmp.*")) and stack.stack_outdated(p) == ""
    sid_t = stack.stack_id(p)
    assert sid_t == hashlib.sha256(json.dumps(h["files"], sort_keys=True).encode()).hexdigest()

    # a re-downloaded seasonal raster -> rebuilt
    s = _layer("S1_asc_2", 77)
    _write(p.raw(cells[2]["cell"], "S1_asc_2"), s, cells[2]["x0"], cells[2]["y1"], descriptions=["VV", "VH"])
    assert stack.stack_outdated(p).startswith("1 input rasters changed")
    stack.build_train_stack(p)
    data[cells[2]["cell"]]["S1_asc_2"] = s
    assert np.array_equal(np.load(p.stack_files("seasonal")[1])[0], _expected_seasonal(data[cells[2]["cell"]]))
    sid_t = stack.stack_id(p)

    # models: stack_id.txt, and runs without it by their result.json (seasonal parts and input)
    d = p.model_dir("unet-sls")
    d.mkdir(parents=True)
    (d / "stack_id.txt").write_text(sid_t)
    assert stack.model_is_current(p, "unet-sls") == (True, "")
    (d / "stack_id.txt").unlink()
    res = dict(annual=xs, labels=gs, seasonal=ss, n_chips=3, model=dict(input="T"))
    json.dump(res, open(d / "result.json", "w"))
    assert stack.model_is_current(p, "unet-sls") == (True, "")
    json.dump(dict(res, seasonal=None), open(d / "result.json", "w"))
    assert stack.model_is_current(p, "unet-sls") == (False, "trained on 0 seasonal part(s), the current stack has 2 "
                                                            "seasonal part(s)")
    json.dump(dict(res, model=dict(input="AE")), open(d / "result.json", "w"))
    assert stack.model_is_current(p, "unet-sls") == (False, "trained with input AE, the project's input is T")

    # TE: the Embedding is needed -> missing rasters, then a settings change (the annual channels differ)
    p.cfg["input"] = "TE"
    assert stack.stack_outdated(p) == "input rasters missing"
    with pytest.raises(FileNotFoundError, match="rasters of used cells missing"):
        stack.build_train_stack(p)
    with pytest.raises(FileNotFoundError, match="input TE needs the Embedding layer"):
        stack.cell_x(p, cells[0])
    for i, c in enumerate(cells):
        data[c["cell"]].update(_fill(p, c, ["Embedding"], 30 + i))
    assert stack.stack_outdated(p) == "settings changed: input T -> TE"
    stack.build_train_stack(p)
    X2 = np.concatenate([np.load(f) for f in p.stack_files("x")])
    S2 = np.concatenate([np.load(f) for f in p.stack_files("seasonal")])
    assert np.array_equal(S2, np.stack([_expected_seasonal(data[c["cell"]]) for c in cells]))
    for j, c in enumerate(cells):
        assert np.array_equal(X2[j], _expected_x_of(data[c["cell"]], "TE")) and not X2[j, 65:].any()
    sid_te = stack.stack_id(p)
    assert sid_te != sid_t and json.load(open(p.path("stacks", "hashes.json")))["settings"]["input"] == "TE"
    assert "settings changed: input T -> TE" in _log(p)

    # back to AE is a rebuild without seasonal parts (S1 / S2 now needed and present)
    for i, c in enumerate(cells):
        data[c["cell"]].update(_fill(p, c, ["S1", "S2"], 60 + i))
    p.cfg["input"] = "AE"
    stack.build_train_stack(p)
    assert not p.stack_files("seasonal") and stack.stack_id(p) not in (sid_t, sid_te)
    X3 = np.concatenate([np.load(f) for f in p.stack_files("x")])
    assert all(np.array_equal(X3[j], _expected_x(data[c["cell"]])) for j, c in enumerate(cells))
    assert stack.model_is_current(p, "unet-sls")[0] is False


def test_cell_x_of_every_input(tmp_path):
    """cell_x reads only the annual layers of the representation; AE is encode_x of all four (the paper)."""
    c = _cell(0, 0, "aoi")
    p = _project(tmp_path / "p", [c])
    a = _fill(p, c, X5, 5)
    for inp in INPUTS:
        x, e = stack.cell_x(p, c, input_name=inp)
        assert e == EPSG and np.array_equal(x, _expected_x_of(a, inp)), inp
    assert np.array_equal(stack.cell_x(p, c)[0], _expected_x(a))
    p.raw(c["cell"], "S1").unlink()
    assert np.array_equal(stack.cell_x(p, c, input_name="E")[0], _expected_x_of(a, "E"))
    with pytest.raises(FileNotFoundError, match="input A needs the S1 layer"):
        stack.cell_x(p, c, input_name="A")


def test_seasonal_band_order_is_checked(tmp_path):
    c = _cell(0, 0, "aoi")
    p = _project(tmp_path / "p", [c], input="T")
    a = _fill(p, c, _needed_layers("T"), 2)
    assert np.array_equal(stack.cell_seasonal(p, c)[0], _expected_seasonal(a))
    _write(p.raw(c["cell"], "S1_asc_1"), a["S1_asc_1"][::-1].copy(), c["x0"], c["y1"], descriptions=["VH", "VV"])
    with pytest.raises(ValueError, match="bands VH, VV, expected VV, VH"):
        stack.cell_seasonal(p, c)
    _write(p.raw(c["cell"], "S1_asc_1"), a["S1_asc_1"], c["x0"], c["y1"])          # no descriptions: accepted
    assert np.array_equal(stack.cell_seasonal(p, c)[0], _expected_seasonal(a))
    _write(p.raw(c["cell"], "S2_3"), a["S2_3"][:8].copy(), c["x0"], c["y1"])
    with pytest.raises(ValueError, match="8 bands, expected 9"):
        stack.cell_seasonal(p, c)
    p.raw(c["cell"], "S2_3").unlink()
    with pytest.raises(FileNotFoundError, match="input T needs the S2_3 layer"):
        stack.cell_seasonal(p, c)


def test_hashes_without_input_are_ie(tmp_path):
    """hashes.json written before the input representation was a setting: an AE stack, kept as it is."""
    cells = [_cell(0, 0, "aoi"), _cell(0, 1, "aoi")]
    p = _project(tmp_path / "p", cells)
    for i, c in enumerate(cells):
        _fill(p, c, X5, i)
    stack.build_train_stack(p)
    hf = p.path("stacks", "hashes.json")
    h = json.load(open(hf))
    del h["settings"]["input"]
    json.dump(h, open(hf, "w"))
    t = _mtimes(p)
    assert stack.stack_outdated(p) == ""
    stack.build_train_stack(p)
    assert _mtimes(p) == t
    p.cfg["input"] = "E"
    assert stack.stack_outdated(p) == "settings changed: input AE -> E"
    # hashes.json of the first version (no settings, no inputs): adopted for AE only
    json.dump({k: v for k, v in h.items() if k not in ("settings", "inputs")}, open(hf, "w"))
    assert stack.stack_outdated(p) == "settings changed: input AE -> E"
    p.cfg["input"] = "AE"
    assert stack.stack_outdated(p) == "" and json.load(open(hf))["settings"]["input"] == "AE"


def test_seasonal_mosaics(tmp_path):
    p, cells, data = _mosaic_project(tmp_path / "p")               # AE first: every layer of the used cells
    for i, c in enumerate(cells):
        if c["use"]:
            data[c["cell"]].update(_fill(p, c, SEASONAL, 70 + i))
    stack.build_mosaics(p)
    assert not p.mosaic("seasonal").exists() and stack.mosaics_match_input(p)
    p.cfg["input"] = "T"
    assert not stack.mosaics_match_input(p)
    p.raw(cells[4]["cell"], "S2_1").unlink()                       # ring cell in the box without all T inputs
    written = stack.build_mosaics(p)
    assert [f.name for f in written] == ["annual.tif", "seasonal.tif", "GEDI.tif", "GFCH.tif", "HRCH.tif",
                                         "aoi_mask.tif"]
    assert stack.mosaics_match_input(p)
    with rasterio.open(p.mosaic("seasonal")) as s:
        assert (s.count, s.height, s.width, s.dtypes[0], s.nodata) == (44, 512, 768, "uint16", 0)
        assert s.crs.to_epsg() == EPSG and tuple(s.transform)[:6] == (10.0, 0.0, GX0, 0.0, -10.0, GY1)
        assert list(s.descriptions) == channels.SEASONAL_BANDS and s.tags()["input"] == "T"
        S = s.read()
    with rasterio.open(p.mosaic("annual")) as s:
        assert s.tags()["input"] == "T" and list(s.descriptions) == channels.ANNUAL_BANDS
        A = s.read()
    for c in cells[:4]:
        win = np.s_[c["row"] * 256:(c["row"] + 1) * 256, c["col"] * 256:(c["col"] + 1) * 256]
        assert np.array_equal(S[(slice(None),) + win], _expected_seasonal(data[c["cell"]]))
        assert np.array_equal(A[(slice(None),) + win], _expected_x_of(data[c["cell"]], "T"))
    assert not S[:, 256:, 256:].any() and not A[:, 256:, 256:].any()     # ring cell without S2_1, unused cell
    assert "study-area cells without inputs" not in _log(p)

    # back to AE: the seasonal mosaic is removed, annual.tif holds every layer again
    p.cfg["input"] = "AE"
    written = stack.build_mosaics(p)
    assert "seasonal.tif" not in [f.name for f in written] and not p.mosaic("seasonal").exists()
    assert "seasonal.tif removed" in _log(p) and stack.mosaics_match_input(p)
    with rasterio.open(p.mosaic("annual")) as s:
        A = s.read()
    c = cells[0]
    assert np.array_equal(A[:, :256, :256], _expected_x(data[c["cell"]]))


def test_earlier_input_names_keep_stacks_and_models(tmp_path):
    """Stacks and models recorded with the earlier names IE / I are current for the projects' AE / A."""
    cells = [_cell(0, 0, "aoi"), _cell(0, 1, "aoi")]
    p = _project(tmp_path / "p", cells)
    for i, c in enumerate(cells):
        _fill(p, c, X5, i)
    stack.build_train_stack(p)
    hf = p.path("stacks", "hashes.json")
    h = json.load(open(hf))
    h["settings"]["input"] = "IE"                                    # written by an earlier version
    json.dump(h, open(hf, "w"))
    t = _mtimes(p)
    assert p["input"] == "AE" and stack.stack_outdated(p) == ""
    stack.build_train_stack(p)
    assert _mtimes(p) == t                                           # not rebuilt
    d = p.model_dir("unet-sls")
    d.mkdir(parents=True)
    (d / "stack_id.txt").unlink(missing_ok=True)
    json.dump(dict(model=dict(input="IE"), annual=p.stack_files("x"), labels=p.stack_files("GEDI"),
                   n_chips=len(cells), seasonal=[]), open(d / "result.json", "w"))
    ok, why = stack.model_is_current(p, "unet-sls")
    assert ok, why
