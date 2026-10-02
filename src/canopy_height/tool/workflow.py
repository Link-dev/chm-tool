"""Stages of the canopy-height tool: plan -> download -> stack -> train -> predict -> report.

Every stage works on one project folder (project.py), holds the project's run lock (one process at a time),
records running / done / failed in status.json, logs start, end, duration and, on failure, the traceback to
logs/run.log, and is resumable: finished parts (plan, downloaded rasters, stack parts, models with a result.json
trained on the current stack, maps newer than their model) are kept, so a stopped or failed run continues where it
ended when the stage is started again.

The models are trained with the paper's settings (canopy_height.train.RECIPES, canopy_height.rf.RF_PARAMS) in
their dependency order RF-SLS, UNet-SLS, KG-UNet1, KG-UNet2, on the project's input representation
(project.INPUTS): KG-UNet1/2 start from the UNet-ALS checkpoint of that representation and KG-UNet2 uses the
project's own UNet-SLS as GEDI teacher. KG-UNet2 uses the paper's settings (at most 50 epochs, pooling 4, teacher
terms and model selection from epoch 11, zero labels missing), the same at every site; they can be changed
through train_overrides. The only automatic changes are the small-data guard (batch size <= number of training
chips, because the recipes assume the ~400-chip training regions of the paper) and the RF-SLS row cap
(rf_max_rows, above the size of the paper's training sets). PyTorch, scikit-learn and Earth Engine are imported
only by the stages that need them.
"""
import csv
import functools
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

from .project import MODEL_NAMES, MODELS, STAGES, Project, canonical_input, replace_retry, run_lock, validate

UNETS = ("unet-sls", "kg-unet1", "kg-unet2")
RECIPE_FILES = {"rf-sls": ("rf_sls.joblib",), **{m: ("model.pth", "history.csv", "split.npz") for m in UNETS}}
MIN_CHIPS = 5                      # fewer training cells than this: no model is trained
MIN_LABELS = 1000                  # fewer GEDI label pixels than this in the training region: no model is trained
BLOCK, HALO = 2048, 256            # windowed prediction of large study areas (core block, context on each side)


def _as_project(project):
    return project if isinstance(project, Project) else Project(project)


def _fmt(sec):
    if sec < 60:
        return f"{sec:.1f} s"
    if sec < 3600:
        return f"{int(sec // 60)} min {int(sec % 60):02d} s"
    return f"{int(sec // 3600)} h {int(sec % 3600 // 60):02d} min"


def _stage(name):
    """Stage wrapper: settings check, run lock, status.json (running / done / failed), log lines with durations,
    traceback on failure. The wrapped function returns (result, status message); the stage returns the result."""
    def deco(fn):
        @functools.wraps(fn)
        def run_stage(project, *args, **kwargs):
            project = _as_project(project)
            validate(project.cfg)
            with run_lock(project, f"stage {name}"):
                t0 = time.time()
                project.set_status(name, "running", "", pid=os.getpid(), seconds=None)
                project.log(f"== {name}: start")
                try:
                    result, message = fn(project, *args, **kwargs)
                except BaseException as e:
                    dt = time.time() - t0
                    msg = "interrupted" if isinstance(e, KeyboardInterrupt) else f"{type(e).__name__}: {e}"
                    project.log(f"== {name}: FAILED after {_fmt(dt)}: {msg}\n{traceback.format_exc()}")
                    project.set_status(name, "failed", msg, seconds=round(dt, 1))
                    raise
                dt = time.time() - t0
                project.set_status(name, "done", message, seconds=round(dt, 1))
                project.log(f"== {name}: done in {_fmt(dt)}" + (f" ({message})" if message else ""))
                return result
        return run_stage
    return deco


def torch_device(project):
    """project['device'] as a torch.device (None = first GPU if available, else CPU)."""
    import torch
    if project["device"]:
        return torch.device(project["device"])
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def read_cells(project):
    """plan/cells.csv as a list of dicts (use parsed to bool)."""
    if not project.cells_file.exists():
        raise FileNotFoundError(f"{project.cells_file} missing: run the plan stage first")
    with open(project.cells_file, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["use"] = str(r.get("use", "True")).strip().lower() in ("true", "1", "yes")
    return rows


def _cells_message(rows):
    aoi = sum(r["role"] == "aoi" for r in rows)
    ring = sum(r["role"] != "aoi" and r["use"] for r in rows)
    dropped = sum(not r["use"] for r in rows)
    msg = f"{aoi} study-area cells, {ring} surrounding cells"
    return msg + (f" ({dropped} water cells dropped)" if dropped else "")


def _rel(project, path):
    try:
        return Path(path).relative_to(project.root).as_posix()
    except ValueError:
        return str(path)


def _newer(path, *others):
    """path exists and is newer than every existing file in others."""
    if not Path(path).exists():
        return False
    t = Path(path).stat().st_mtime
    return all(t >= Path(o).stat().st_mtime for o in others if Path(o).exists())


# ---------------------------------------------------------------------------------------------- plan
@_stage("plan")
def plan(project):
    """Analysis grid and training cells (grid.plan_cells); kept if plan/cells.csv exists."""
    if project.cells_file.exists():
        project.log("plan: plan/cells.csv exists, kept (delete the plan folder to plan again)")
    else:
        from . import grid
        grid.plan_cells(project)
    rows = read_cells(project)
    return rows, _cells_message(rows)


# ---------------------------------------------------------------------------------------------- download
def _job(f):
    return f"{f.get('cell')} {f.get('layers')}" if isinstance(f, dict) else str(f)


@_stage("download")
def download(project, workers=None):
    """Model inputs, GEDI labels and benchmarks of every used cell (download.run_download); existing files are
    skipped. Fails if any job failed (errors in the log); the next run retries only the missing files."""
    from . import download as dl
    read_cells(project)
    result = dl.run_download(project, workers=workers if workers is not None else project["workers"])
    failed = list((result or {}).get("failed") or [])
    if failed:
        shown = ", ".join(_job(f) for f in failed[:20]) + (f", ... ({len(failed)} in total)" if len(failed) > 20 else "")
        raise RuntimeError(f"{len(failed)} download jobs failed (see the log; run the stage again to retry): {shown}")
    result = result or {}
    return result, f"{result.get('cells', '?')} cells, {result.get('jobs', 0)} downloads, " \
                   f"{result.get('gmtch', 0)} GMTCH derived"


# ---------------------------------------------------------------------------------------------- stack
def _newest_raw(project):
    """Newest file of raw/ (None if empty): mosaics older than a new download are rebuilt."""
    d = project.path("raw")
    if not d.exists():
        return None
    newest = max((e for e in os.scandir(d) if e.is_file()), key=lambda e: e.stat().st_mtime, default=None)
    return newest.path if newest else None


def _mosaics_current(project):
    """The study-area mosaics exist and are newer than the plan, the settings, the study area and raw/."""
    need = ["annual", "GEDI", "aoi_mask"] + (["seasonal"] if project.seasonal else [])
    inputs = [project.cells_file, project.path("project.yaml"), project.aoi_file] + \
        list(filter(None, [_newest_raw(project)]))
    return all(_newer(project.mosaic(k), *inputs) for k in need)


@_stage("stack")
def stack(project):
    """Training stack (stack.build_train_stack), study-area mosaics (stack.build_mosaics) and, with ALS rasters
    in the settings, the ALS reference (stack.als_reference); mosaics and ALS reference newer than their inputs are
    kept."""
    from ..data import Stack
    from . import stack as st
    rows = read_cells(project)
    st.build_train_stack(project)
    parts = project.stack_files("x")
    if not any(r["role"] == "aoi" and r["use"] for r in rows):
        project.log("stack: no study-area cells, mosaics skipped")
    elif _mosaics_current(project):
        project.log("stack: study-area mosaics are up to date, kept")
    else:
        st.build_mosaics(project)
    if project["als"]:
        als = [project["als"]] if isinstance(project["als"], str) else list(project["als"])
        if _newer(project.mosaic("ALS"), project.mosaic("annual"), project.path("project.yaml"), *als):
            project.log("stack: mosaic/ALS.tif is up to date, kept")
        else:
            st.als_reference(project)
    msg = f"{len(Stack(parts)) if parts else 0} training chips in {len(parts)} part(s)"
    return None, msg + (", study-area mosaics" if project.mosaic("annual").exists() else ", no study-area mosaic") \
        + (", ALS reference" if project.mosaic("ALS").exists() else "")


# ---------------------------------------------------------------------------------------------- train
def model_order(models):
    """Models in dependency order (rf-sls, unet-sls, kg-unet1, kg-unet2); unknown names are an error."""
    bad = [m for m in models if m not in MODELS]
    if bad:
        raise ValueError(f"unknown models {bad}; choose from {MODELS}")
    return [m for m in MODELS if m in models]


def trained_models(project):
    """Models with a finished run (result.json and the model file), in MODELS order."""
    return [m for m in MODELS
            if (project.model_dir(m) / "result.json").exists() and project.model_file(m).exists()]


def _clean_partial(project, model):
    d = project.model_dir(model)
    removed = []
    for f in RECIPE_FILES[model]:
        p = d / f
        if p.exists():
            p.unlink()
            removed.append(f)
    if removed:
        project.log(f"{MODEL_NAMES[model]}: removed files of an unfinished run: {', '.join(removed)}")


def _run_input(res):
    """Input representation a finished run was trained with (result.json of rf.train_rf / train.train)."""
    inp = res.get("input") or (res.get("model") or {}).get("input") or (res.get("recipe") or {}).get("input")
    return canonical_input(inp) if inp else inp     # aliases IE, I -> AE, A


def _file_id(path):
    """sha256 of a model file (identifies the GEDI teacher a KG-UNet2 run was trained with)."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def stale_reason(project, model, retrained=()):
    """Why a finished model run no longer fits the project (None if it does)."""
    from . import stack as st
    d = project.model_dir(model)
    if not project.model_file(model).exists():
        return f"{project.model_file(model).name} is missing"
    ok, reason = st.model_is_current(project, model)
    if not ok:
        return reason or "trained on another training stack"
    try:
        res = json.load(open(d / "result.json", encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unreadable result.json"
    inp = _run_input(res)
    if inp and inp != project["input"]:
        return f"trained with input {inp}, the project uses {project['input']}"
    if model == "kg-unet2":
        teacher = project.model_file("unet-sls")
        if "unet-sls" in retrained:
            return "its GEDI teacher UNet-SLS was retrained"
        tid = d / "teacher_id.txt"
        if teacher.exists() and tid.exists():
            if tid.read_text(encoding="utf-8").strip() != _file_id(teacher):
                return "its GEDI teacher UNet-SLS has changed since"
        elif teacher.exists() and teacher.stat().st_mtime > (d / "result.json").stat().st_mtime:
            return "its GEDI teacher UNet-SLS is newer than the run"
    return None


def _retire(project, model, reason):
    """Keep a stale run as models/<model>.stale-<time> (nothing is deleted) so the model is trained again."""
    d = project.model_dir(model)
    dst = d.with_name(f"{d.name}.stale-{time.strftime('%Y%m%d-%H%M%S')}")
    os.replace(d, dst)
    project.log(f"{MODEL_NAMES[model]}: {reason}; previous run kept as {_rel(project, dst)}, training again")


def _small_data_guard(project, recipe, n_chips, overrides):
    """Batch size = number of training chips when the recipe's split leaves fewer chips than one batch."""
    from ..train import resolve, split_chips
    cfg = resolve(recipe, **overrides)
    n_train = len(split_chips(n_chips, cfg)[0])
    if n_train < 1:
        raise ValueError(f"{MODEL_NAMES[recipe]}: no training chips left after the validation split")
    if n_train < cfg["batch_size"]:
        project.log(f"WARNING: {MODEL_NAMES[recipe]}: {n_train} training chips, fewer than the batch size "
                    f"{cfg['batch_size']}; batch size set to {n_train}. Results of such a small area are not "
                    f"meaningful.")
        overrides = dict(overrides, batch_size=n_train)
    return overrides


def _teacher_sls(project):
    t = project.model_file("unet-sls")
    if not (t.exists() and (project.model_dir("unet-sls") / "result.json").exists()):
        raise FileNotFoundError(f"KG-UNet2 needs a trained UNet-SLS as GEDI teacher ({t}): add unet-sls to the "
                                f"models")
    reason = stale_reason(project, "unet-sls")
    if reason:
        raise ValueError(f"the GEDI teacher UNet-SLS does not fit the project ({reason}): train unet-sls again "
                         f"first (add it to the models)")
    return t


def _base_for(project):
    """UNet-ALS checkpoint of the project's representation; its stored input layout must match."""
    from ..models import load_checkpoint
    base = project.base_model()
    _, cfg = load_checkpoint(base)
    if cfg.get("input") != project["input"]:
        raise ValueError(f"{base.name} has input {cfg.get('input')}, the project uses {project['input']}")
    return base


def _count_labels(files, block=50):
    """GEDI label pixels (> 0) of the label stack, the pixels RF-SLS fits and the UNets learn from."""
    n = 0
    for f in files:
        a = np.load(f, mmap_mode="r")
        for i in range(0, a.shape[0], block):
            n += int((np.asarray(a[i:i + block]) > 0).sum())
    return n


@_stage("train")
def train(project, models=None):
    """Train the models (project['models'] or `models`) on the training stack; finished runs that still fit the
    stack, the input representation and their teacher are kept, others are retired and trained again."""
    from . import stack as st
    models = model_order(project["models"] if models is None else models)
    sid = st.stack_id(project)
    if sid is None:
        raise FileNotFoundError("the training stack is missing or incomplete (stacks/hashes.json): run the stack "
                                "stage first")
    annual, labels = project.stack_files("x"), project.stack_files("GEDI")
    seasonal = project.stack_files("seasonal") if project.seasonal else None
    if len(labels) != len(annual) or (seasonal is not None and len(seasonal) != len(annual)):
        raise ValueError(f"stack parts differ in number in {project.path('stacks')}")
    from ..data import Stack
    n = len(Stack(annual))
    if len(Stack(labels)) != n or (seasonal and len(Stack(seasonal)) != n):
        raise ValueError("input and label stacks differ in chip count")
    if n < MIN_CHIPS:
        raise ValueError(f"only {n} training cells; at least {MIN_CHIPS} are needed")
    n_lab = _count_labels(labels)
    if n_lab < MIN_LABELS:
        raise ValueError(f"only {n_lab} GEDI label pixels in the training region (at least {MIN_LABELS} needed): "
                         f"enlarge the training region (train_cells) or the GEDI window")
    if any(m in ("kg-unet1", "kg-unet2") for m in models):
        _base_for(project)                         # a wrong UNet-ALS fails now, not after hours of other models
    project.log(f"train: {n} chips, {n_lab:,} GEDI label pixels, input {project['input']}")
    done, kept, retrained = [], [], set()
    for i, m in enumerate(models, 1):
        name, out = MODEL_NAMES[m], project.model_dir(m)
        if (out / "result.json").exists():
            reason = stale_reason(project, m, retrained)
            if reason is None:
                project.log(f"{name}: finished run in {_rel(project, out)}, kept")
                kept.append(name)
                continue
            _retire(project, m, reason)
        _clean_partial(project, m)
        over = dict((project["train_overrides"] or {}).get(m) or {})
        project.log(f"-- {name} ({i}/{len(models)}): {n} chips -> {_rel(project, out)}"
                    + (f", settings {over}" if over else ""))
        t0 = time.time()
        if m == "rf-sls":
            from ..rf import train_rf
            kw = {"encoding": "paper", "input_name": project["input"], "max_rows": project["rf_max_rows"], **over}
            _, meta = train_rf(out, annual=annual, labels=labels, seasonal=seasonal, log=project.log, **kw)
            info = f"{meta['n_fit']:,} pixels, hold-out RMSE {meta['holdout_rmse_m']:.2f} m"
        else:
            from ..train import train as train_unet
            if m == "unet-sls":
                over.setdefault("input", project["input"])
            over = _small_data_guard(project, m, n, over)
            base = _base_for(project) if m in ("kg-unet1", "kg-unet2") else None
            teacher = _teacher_sls(project) if m == "kg-unet2" else None
            _, res = train_unet(m, out, annual, labels, seasonal=seasonal, base=base, teacher_sls=teacher,
                                device=torch_device(project), log=project.log, **over)
            info = f"{res['epochs_run']} epochs, final val loss {res['final_val_loss']:.4f}"
        (out / "stack_id.txt").write_text(sid + "\n", encoding="utf-8")
        if m == "kg-unet2":
            (out / "teacher_id.txt").write_text(_file_id(teacher) + "\n", encoding="utf-8")
        project.log(f"-- {name}: done in {_fmt(time.time() - t0)} ({info})")
        done.append(name)
        retrained.add(m)
    msg = ", ".join(x for x in (f"trained {', '.join(done)}" if done else "",
                                f"kept {', '.join(kept)}" if kept else "") if x)
    return trained_models(project), msg


# ---------------------------------------------------------------------------------------------- predict
def map_profile(profile):
    """GeoTIFF profile of a map on the grid of `profile` (as canopy_height.predict.predict_geotiff writes)."""
    p = dict(profile)
    p.update(count=1, dtype="float32", nodata=np.nan, compress="deflate", predictor=3, tiled=True,
             blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
    p.pop("interleave", None)
    return p


def read_aoi_mask(project, like=None):
    """Boolean study-area mask of the mosaic grid; `like` = rasterio profile the mask must match."""
    import rasterio
    f = project.mosaic("aoi_mask")
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: run the stack stage")
    with rasterio.open(f) as s:
        m = s.read(1) == 1
        g = (s.width, s.height, s.transform, s.crs)
    if like is not None and g != (like["width"], like["height"], like["transform"], like["crs"]):
        raise ValueError(f"{f} is not on the grid of mosaic/annual.tif")
    return m


def _blocks(H, W, inside):
    """Core windows (r0, c0, h, w) of BLOCK x BLOCK pixels that contain study-area pixels."""
    for r0 in range(0, H, BLOCK):
        for c0 in range(0, W, BLOCK):
            h, w = min(BLOCK, H - r0), min(BLOCK, W - c0)
            if inside[r0:r0 + h, c0:c0 + w].any():
                yield r0, c0, h, w


def _read(path, window=None):
    import rasterio
    with rasterio.open(path) as s:
        return s.read(window=window)


def predict_unet_map(model, cfg, annual, seasonal, inside, device):
    """Canopy height of the mosaic. Up to BLOCK + 2 HALO pixels a side the whole raster is predicted at once
    (= canopy_height.predict.predict_geotiff); larger mosaics block by block, each with HALO pixels of context on
    every side (256 px tiles with 32 px blended overlaps inside the block), blocks without study-area pixels
    skipped, so memory stays that of one block."""
    from rasterio.windows import Window
    from ..predict import predict_array
    H, W = inside.shape
    if H <= BLOCK + 2 * HALO and W <= BLOCK + 2 * HALO:
        return predict_array(model, cfg, _read(annual), None if seasonal is None else _read(seasonal),
                             device=device)
    out = np.full((H, W), np.nan, dtype=np.float32)
    for r0, c0, h, w in _blocks(H, W, inside):
        rr0, cc0 = max(0, r0 - HALO), max(0, c0 - HALO)
        rr1, cc1 = min(H, r0 + h + HALO), min(W, c0 + w + HALO)
        win = Window(cc0, rr0, cc1 - cc0, rr1 - rr0)
        p = predict_array(model, cfg, _read(annual, win), None if seasonal is None else _read(seasonal, win),
                          device=device)
        out[r0:r0 + h, c0:c0 + w] = p[r0 - rr0:r0 - rr0 + h, c0 - cc0:c0 - cc0 + w]
    return out


def predict_rf_map(rf, meta, annual, seasonal, inside, chunk=500_000):
    """RF-SLS canopy height of the study-area pixels (block by block; per pixel identical to
    canopy_height.rf.predict_rf_raster, which predicts every pixel of the raster)."""
    from rasterio.windows import Window
    from .. import channels
    from ..rf import encode
    H, W = inside.shape
    n_embed = channels.INPUTS[meta["input"]]["n_embed"]
    out = np.full((H, W), np.nan, dtype=np.float32)
    for r0, c0, h, w in _blocks(H, W, inside):
        win = Window(c0, r0, w, h)
        a = _read(annual, win)
        s = None if seasonal is None else _read(seasonal, win)
        x = channels.assemble(meta["input"], a[None], None if s is None else s[None])[0]
        m = inside[r0:r0 + h, c0:c0 + w] & ~(a == 0).all(axis=0)
        rows = x[:, m].T
        pred = np.empty(rows.shape[0], dtype=np.float32)
        for k in range(0, rows.shape[0], chunk):
            pred[k:k + chunk] = rf.predict(encode(np.ascontiguousarray(rows[k:k + chunk]), meta["encoding"],
                                                  n_embed)).astype(np.float32)
        blk = np.full((h, w), np.nan, dtype=np.float32)
        blk[m] = pred
        out[r0:r0 + h, c0:c0 + w] = blk
    return out


def _write_map(path, pred, profile, inside, model):
    import rasterio
    pred = np.array(pred, dtype=np.float32)
    pred[~inside] = np.nan
    with rasterio.open(path, "w", **map_profile(profile)) as dst:
        dst.write(pred, 1)
        dst.set_band_description(1, "canopy height (m)")
        dst.update_tags(model=MODEL_NAMES[model])


@_stage("predict")
def predict(project):
    """maps/<Model>.tif of every trained model on the study-area mosaics, NaN outside the study area; a map newer
    than its model and the mosaics is kept."""
    import rasterio
    annual = project.mosaic("annual")
    if not annual.exists():
        raise FileNotFoundError(f"{annual} missing: the stack stage builds it from the study-area cells of "
                                f"plan/grid.json (run the stack stage first)")
    seasonal = project.mosaic("seasonal") if project.seasonal else None
    if seasonal is not None and not seasonal.exists():
        raise FileNotFoundError(f"{seasonal} missing: run the stack stage (input {project['input']})")
    models = trained_models(project)
    if not models:
        raise FileNotFoundError("no trained model (models/<model>/result.json): run the train stage first")
    with rasterio.open(annual) as s:
        profile = s.profile
    inside = read_aoi_mask(project, like=profile)
    for leftover in project.path("maps").glob("_*.part") if project.path("maps").exists() else []:
        leftover.unlink(missing_ok=True)                  # temp file of an interrupted predict stage
    done, kept = [], []
    for m in models:
        name, dst = MODEL_NAMES[m], project.map_file(m)
        inputs = [project.model_file(m), annual, project.mosaic("aoi_mask")] + ([seasonal] if seasonal else [])
        if _newer(dst, *inputs):
            project.log(f"{name}: {_rel(project, dst)} is up to date, kept")
            kept.append(name)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f"_{dst.name}.part")          # no *.tif glob (app, report) sees a half-written map
        t0 = time.time()
        if m == "rf-sls":
            from ..rf import load_rf
            rf, meta = load_rf(project.model_file(m))
            pred = predict_rf_map(rf, meta, annual, seasonal, inside)
            del rf
        else:
            from ..models import load_model
            dev = torch_device(project)
            model, cfg = load_model(project.model_file(m), device=dev)
            pred = predict_unet_map(model, cfg, annual, seasonal, inside, dev)
            del model
        _write_map(tmp, pred, profile, inside, m)
        replace_retry(tmp, dst)
        v = pred[inside & np.isfinite(pred)]
        project.log(f"{name}: {_rel(project, dst)} in {_fmt(time.time() - t0)}"
                    + (f" (study-area mean {v.mean():.2f} m)" if v.size else " (no valid pixel in the study area)"))
        done.append(name)
    msg = ", ".join(x for x in (f"mapped {', '.join(done)}" if done else "",
                                f"kept {', '.join(kept)}" if kept else "") if x)
    return [project.map_file(m) for m in models], msg


# ---------------------------------------------------------------------------------------------- report
@_stage("report")
def report(project):
    """report/summary.json, report/metrics.csv and quick-look PNGs (report.report)."""
    from . import report as rp
    summary = rp.report(project)
    return summary, f"{summary.get('n_rows', 0)} metric rows, {len(summary.get('quicklooks', []))} quick-looks"


# ---------------------------------------------------------------------------------------------- run
STAGE_FUNCTIONS = {"plan": plan, "download": download, "stack": stack, "train": train, "predict": predict,
                   "report": report}


STAGE_TITLES = {"plan": "planning the cells", "download": "downloading the inputs and GEDI labels",
                "stack": "building the training data", "train": "training the models",
                "predict": "mapping canopy height", "report": "writing the report"}


def run(project, from_stage="plan", to_stage="report", workers=None):
    """Run the stages from_stage .. to_stage in order; stops at the first failing stage (exception raised)."""
    project = _as_project(project)
    validate(project.cfg)
    if from_stage not in STAGES or to_stage not in STAGES:
        raise ValueError(f"stages are {STAGES}")
    i, j = STAGES.index(from_stage), STAGES.index(to_stage)
    if i > j:
        raise ValueError(f"stage {from_stage} comes after {to_stage}")
    with run_lock(project, f"run {from_stage}..{to_stage}"):
        project.log(f"run: {' -> '.join(STAGES[i:j + 1])} (pid {os.getpid()}, input {project['input']})")
        t0 = time.time()
        for s in STAGES[i:j + 1]:
            project.log(f"== Step {STAGES.index(s) + 1}/{len(STAGES)}: {STAGE_TITLES[s]} ==")
            if s == "download":
                download(project, workers=workers)
            else:
                STAGE_FUNCTIONS[s](project)
        project.log(f"run: finished in {_fmt(time.time() - t0)}")
    return project.status()
