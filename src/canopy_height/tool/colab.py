"""Helpers of the Google Colab notebook (notebooks/chm_tool_colab.ipynb): the same stages as `chm-tool run`, with
what a Colab session needs around them. Nothing here imports Streamlit, and nothing changes the recipes.

- environment() / colab_settings(): GPU, RAM and disk of the session; the RF-SLS row cap for its RAM. The paper's
  batch size 25 fits a T4 (measured on a 439-cell training region: UNet-SLS peaks at 9.6 GB allocated, KG-UNet2 at
  6.8 GB), so the UNets keep the paper's settings. RF-SLS grows with its training rows (380 k rows -> 4 GB forest,
  8 GB while loading), so on < 16 GB RAM rf_max_rows is capped at 500 k (above the paper's training sets).
- fetch_weights(): only the UNet-ALS checkpoint of the chosen input representation (124 MB), from a folder (e.g.
  Google Drive) or a URL, checked against its sha256, then $CHM_WEIGHTS points at it.
- aoi_from_bbox() / AoiDrawer: the study area from a longitude / latitude box or drawn on an ipyleaflet map.
- run(): if the Drive mount goes away mid-run, Drive is mounted again and the run continues (DriveDisconnected
  after repeated failures).
- run(): the project folder lives on Google Drive (downloads survive a disconnect; run again to resume), but the
  training stack is copied to the session's local disk before training, because the UNets read random chips
  every epoch and Drive's FUSE mount is slow at that. The copy is used only while its hashes.json equals the
  project's, so a rebuilt stack is never shadowed by an old copy.
- results_map() / metrics_table() / storage(): the results and the project's disk use.
"""
import errno
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path

from .project import BENCHMARKS, INPUTS, MODEL_NAMES, MODELS, STAGES, Project

# sha256 of the UNet-ALS checkpoints (assets of the GitHub release, project.WEIGHTS_URL)
CHECKPOINT_SHA256 = {
    "UNet-ALS.pth": "b86e3b93e6b25d31326a67da243f098d959067fdbe3cdda731a242370a554a81",
    "UNet-E-ALS.pth": "f4796f364f30026a603fc2ac59023cdfc136f1018ae9cb966c454a4b32e7c97a",
    "UNet-A-ALS.pth": "41f0897b704f71581e38159d9716d0d52706bbbedefbc3007e48fad376a0ba1e",   # formerly UNet-S-ALS.pth
    "UNet-T-ALS.pth": "61fecd223dc46b8aed38d4df67fe8fddf09a19d188419e0e7138d401f23ba0fa",
    "UNet-TE-ALS.pth": "b51ba6fc6861b0dc51feb4c680ec2e1fb27543c6d67dee9344372d774f6f9b86",
}
BENCHMARK_REFS = {"GMTCH": "Meta, Tolan et al. 2024", "GFCH": "UMD, Potapov et al. 2021", "HRCH": "ETH, Lang et al. 2023"}
COLAB_RF_MAX_ROWS = 500_000        # RF-SLS row cap below 16 GB RAM (380 k rows -> 4 GB forest, 8 GB peak on load)
LOW_RAM_GB = 16.0
MIN_GPU_GB = 11.0                  # UNet-SLS at batch 25 peaks at 9.6 GB allocated
MAX_PX = 1024                      # overlay size on the results map


def _gb(n):
    return round(n / 2 ** 30, 1)


def in_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def environment(path="/content"):
    """{'colab', 'gpu', 'gpu_gb', 'ram_gb', 'disk_free_gb'} of this session."""
    import psutil
    env = dict(colab=in_colab(), gpu=None, gpu_gb=0.0, ram_gb=_gb(psutil.virtual_memory().total))
    try:
        import torch
        if torch.cuda.is_available():
            pr = torch.cuda.get_device_properties(0)
            env.update(gpu=pr.name, gpu_gb=_gb(pr.total_memory))
    except ImportError:
        pass
    d = path if os.path.exists(path) else os.getcwd()
    env["disk_free_gb"] = _gb(shutil.disk_usage(d).free)
    return env


def colab_settings(env=None):
    """(settings for Project.create / project.yaml, [warnings]) for this session."""
    env = env or environment()
    settings, warn = {}, []
    if env["ram_gb"] < LOW_RAM_GB:
        settings["rf_max_rows"] = COLAB_RF_MAX_ROWS
    if not env["gpu"]:
        warn.append("No GPU: this session downloads, builds the stack and trains RF-SLS (none of which needs a GPU), "
                    "then stops before the UNets. Afterwards switch to a GPU runtime (Runtime > Change runtime type > "
                    "T4 GPU) and run the notebook again; everything finished is kept on Drive.")
    elif env["gpu_gb"] < MIN_GPU_GB:
        warn.append(f"{env['gpu']} has {env['gpu_gb']} GB: UNet-SLS needs about 10 GB at the paper's batch size 25. "
                    "Set a smaller batch_size in train_overrides (the models then differ from the paper's recipe).")
    return settings, warn


# ---------------------------------------------------------------------------------------------- weights
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_weights(inp, source, dest="/content/chm_weights", log=print):
    """Put the UNet-ALS checkpoint of input representation `inp` into dest/source/ and point $CHM_WEIGHTS at dest.
    `source`: a folder holding the checkpoint (directly or in source/, e.g. the release's weights/ folder copied to
    Google Drive) or a URL prefix the file name is appended to. The file is checked against its sha256."""
    name = INPUTS[inp]["checkpoint"]                    # INPUTS also finds the earlier names IE, I
    out = Path(dest) / "source" / name
    want = CHECKPOINT_SHA256[name]
    if not (out.exists() and sha256(out) == want):
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(name + ".part")
        s = str(source)
        if s.startswith(("http://", "https://")):
            url = s.rstrip("/") + "/" + name
            log(f"downloading {url}")
            urllib.request.urlretrieve(url, tmp)
        else:
            cands = [Path(s) / "source" / name, Path(s) / name]
            src = next((c for c in cands if c.exists()), None)
            if src is None:
                raise FileNotFoundError(f"{name} not found in {s} (looked for {', '.join(map(str, cands))})")
            log(f"copying {src}")
            shutil.copyfile(src, tmp)
        got = sha256(tmp)
        if got != want:
            tmp.unlink()
            raise ValueError(f"{name}: sha256 {got} differs from the release's {want} (incomplete or wrong file)")
        os.replace(tmp, out)
    os.environ["CHM_WEIGHTS"] = str(Path(dest))
    log(f"{name}: ok ({out})")
    return out


# ---------------------------------------------------------------------------------------------- study area
def aoi_from_bbox(west, south, east, north):
    """Study area as a longitude / latitude box (EPSG:4326)."""
    from shapely.geometry import box
    west, south, east, north = map(float, (west, south, east, north))
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError(f"box west={west}, south={south}, east={east}, north={north}: need west < east and "
                         "south < north in degrees")
    return box(west, south, east, north)


class AoiDrawer:
    """ipyleaflet map to draw the study area (rectangle or polygon); .geometry is the last shape drawn. In Colab,
    google.colab.output.enable_custom_widget_manager() must run first (done here)."""

    def __init__(self, center=(20.0, 0.0), zoom=2, height="500px"):
        import ipyleaflet as L
        if in_colab():
            from google.colab import output
            output.enable_custom_widget_manager()
        self.geometry = None
        self.map = L.Map(center=center, zoom=zoom, scroll_wheel_zoom=True, layout={"height": height})
        self.map.add(L.TileLayer(url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/"
                                     "tile/{z}/{y}/{x}", name="Satellite", attribution="Esri"))
        dc = L.DrawControl(polyline={}, circlemarker={}, marker={}, circle={},
                           rectangle={"shapeOptions": {"color": "#e41a1c"}},
                           polygon={"shapeOptions": {"color": "#e41a1c"}})
        dc.on_draw(self._drawn)
        self.map.add(dc)
        self.map.add(L.SearchControl(position="topright", url="https://nominatim.openstreetmap.org/search?format=json&q={s}",
                                     zoom=10))

    def _drawn(self, control, action, geo_json):
        from shapely.geometry import shape
        if action == "created":
            self.geometry = shape(geo_json["geometry"])

    def _ipython_display_(self):
        from IPython.display import display
        display(self.map)


# ---------------------------------------------------------------------------------------------- local stack copy
class LocalStacks(Project):
    """The project, with the training stack read from a local copy (cache/<project name>/stacks) while that copy's
    hashes.json equals the project's; everything else (and all writes) stays in the project folder."""

    def __init__(self, root, cache):
        super().__init__(root)
        self.cache = Path(cache) / self.root.name / "stacks"

    def _cache_valid(self):
        a, b = self.path("stacks", "hashes.json"), self.cache / "hashes.json"
        return a.exists() and b.exists() and a.read_bytes() == b.read_bytes()

    def stack_files(self, key="x"):
        if not self._cache_valid():
            return super().stack_files(key)
        pat = "train_part*.npy" if key == "x" else f"train_{key}_part*.npy"
        return sorted(str(p) for p in self.cache.glob(pat))


def cache_stacks(project, cache="/content/chm_cache", log=print):
    """Copy the project's complete training stack to the local disk (skipped if the copy is current)."""
    from . import stack as st
    p = LocalStacks(project.root, cache)
    if st.stack_id(p) is None:
        return p                                   # no complete stack yet: train will say so
    if p._cache_valid():
        log(f"training stack: local copy {p.cache} is current")
        return p
    src = p.path("stacks")
    files = sorted(src.glob("train_*.npy"))
    need = sum(f.stat().st_size for f in files)
    Path(cache).mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cache).free
    if need > 0.9 * free:
        log(f"training stack ({_gb(need)} GB) does not fit the local disk ({_gb(free)} GB free): read from the "
            "project folder (slower)")
        return p
    if p.cache.exists():
        shutil.rmtree(p.cache)
    p.cache.mkdir(parents=True)
    log(f"copying the training stack ({_gb(need)} GB) to {p.cache} ...")
    for f in files:
        shutil.copyfile(f, p.cache / f.name)
    shutil.copyfile(src / "hashes.json", p.cache / "hashes.json")   # last: marks the copy complete
    return p


# ---------------------------------------------------------------------------------------------- run
CPU_MODELS = ("rf-sls",)          # trained on the CPU whatever the runtime


def pending_models(project):
    """Models of the project without a finished run that fits the current stack (workflow.stale_reason)."""
    from . import workflow
    return [m for m in workflow.model_order(project["models"])
            if not (project.model_dir(m) / "result.json").exists() or workflow.stale_reason(project, m) is not None]


DRIVE = "/content/drive"
DISCONNECT_ERRNOS = {errno.ENOTCONN, errno.ECONNABORTED}   # "Transport endpoint is not connected": FUSE mount gone


class DriveDisconnected(RuntimeError):
    pass


def is_disconnect(exc):
    """True if exc, or an exception it was raised from / during, is the Drive mount going away."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, OSError) and exc.errno in DISCONNECT_ERRNOS:
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def remount_drive(mountpoint=DRIVE, log=print):
    """Mount Google Drive again after a disconnect (Colab only); True on success."""
    if not in_colab():
        return False
    try:
        from google.colab import drive
        drive.mount(mountpoint, force_remount=True)
        return True
    except Exception as e:  # noqa: BLE001
        log(f"remounting Google Drive failed: {type(e).__name__}: {e}")
        return False


def drop_log(project):
    """Remove the project's log handlers without flushing to a dead mount."""
    import logging
    lg = logging.getLogger(f"chm-tool:{project.root}")
    for h in list(lg.handlers):
        try:
            h.close()
        except OSError:
            pass
        lg.removeHandler(h)
    project._log = None


def run(project, to_stage="report", from_stage="plan", workers=None, cache="/content/chm_cache", log=print, gpu=None,
        remounts=2):
    """Run the stages from_stage .. to_stage (resumable, as `chm-tool run`); train / predict / report read the
    training stack from a local copy.

    Without a GPU (gpu=None: torch.cuda.is_available()) and unless project['device'] is set to 'cpu', only the
    models that need no GPU (RF-SLS) are trained and the run stops before the UNets: downloads, the stack and
    RF-SLS can run on a cheap CPU runtime, and the same call on a GPU runtime later keeps them and trains the rest.

    If the Google Drive mount goes away ("Transport endpoint is not connected"), Drive is mounted again and the run
    continues where it stopped, at most `remounts` times; then DriveDisconnected says what to do. Nothing finished
    is lost: every file is written to a temporary name first. Returns the project status."""
    project = project if isinstance(project, Project) else Project(project)
    for attempt in range(remounts + 1):
        try:
            return _run(project, to_stage, from_stage, workers, cache, log, gpu)
        except BaseException as e:
            if not is_disconnect(e):
                raise
            drop_log(project)
            if attempt < remounts:
                log("Google Drive disconnected; mounting it again and continuing (finished files are kept) ...")
                if remount_drive(log=log):
                    continue
            raise DriveDisconnected(
                "Google Drive disconnected (Transport endpoint is not connected). Nothing finished is lost. Mount it "
                "again with  from google.colab import drive; drive.mount('/content/drive', force_remount=True)  and "
                "run cells 5 and 7 again; if that fails: Runtime > Disconnect and delete runtime, then run the cells "
                "from the top.") from None


UNET_MODELS = ("unet-sls", "kg-unet1", "kg-unet2")
RECIPE_WORKERS = 6                 # DataLoader workers of the recipes (train.py) when a run has >= 4 batches


def cap_loader_workers(project, log=print):
    """At most one DataLoader worker per CPU core for the UNets (in memory only, project.yaml is not changed).

    Each worker keeps 2 batches in flight (25 chips of up to 109 channels as float32: 0.4-0.7 GB), so the recipes'
    6 workers alone hold ~5 GB, which crashed a standard Colab runtime (12.7 GB RAM, 2 cores). The worker count
    does not change the model: the chips of every batch are drawn in the main process with the recipe's seed and
    reading a chip is deterministic (2 epochs, 6 vs 2 workers: identical weights)."""
    n = max(1, os.cpu_count() or 1)
    if n >= RECIPE_WORKERS:
        return
    over = dict(project.cfg.get("train_overrides") or {})
    for m in UNET_MODELS:
        o = dict(over.get(m) or {})
        o.setdefault("num_workers", n)
        over[m] = o
    project.cfg["train_overrides"] = over


def _run(project, to_stage, from_stage, workers, cache, log, gpu):
    from . import workflow
    project = project if isinstance(project, Project) else Project(project)
    i, j = STAGES.index(from_stage), STAGES.index(to_stage)
    k = STAGES.index("train")
    if gpu is None:
        import torch
        gpu = torch.cuda.is_available()
    from . import stack as st
    old = os.environ.get(st.SCRATCH_ENV)
    os.environ[st.SCRATCH_ENV] = os.path.join(cache, "stack_build")     # the stack's memmaps on the local disk
    try:
        if i < k:
            workflow.run(project, from_stage, STAGES[min(j, k - 1)], workers=workers)
        if j >= k:
            p = cache_stacks(project, cache, log=log)
            cap_loader_workers(p, log=log)
            try:
                if gpu or str(project["device"] or "").startswith("cpu"):
                    workflow.run(p, STAGES[max(i, k)], to_stage, workers=workers)
                else:
                    cpu = [m for m in pending_models(p) if m in CPU_MODELS]
                    if cpu and i <= k:
                        workflow.train(p, models=cpu)
                    rest = [m for m in pending_models(p) if m not in CPU_MODELS]
                    if rest:
                        names = ", ".join(MODEL_NAMES[m] for m in rest)
                        log(f"No GPU in this session: stopped before {names}. Switch to a GPU runtime (Runtime > "
                            "Change runtime type > T4 GPU) and run the cells again: the downloads, the stack and "
                            "the models trained so far are kept.")
                        return project.status()
                    if j > k:
                        workflow.run(p, STAGES[k + 1], to_stage, workers=workers)
            finally:
                p.close_log()
    finally:
        project.close_log()
        if old is None:
            os.environ.pop(st.SCRATCH_ENV, None)
        else:
            os.environ[st.SCRATCH_ENV] = old
    return project.status()


def status_table(project):
    import pandas as pd
    st = project.status()
    return pd.DataFrame([dict(stage=s, **{k: st[s].get(k) for k in ("state", "started", "finished", "seconds",
                                                                    "message")}) for s in STAGES]).set_index("stage")


# ---------------------------------------------------------------------------------------------- results
def result_layers(project):
    """[(name, GeoTIFF, block reduction)] of the maps, benchmarks, ALS and GEDI in the project (as the app)."""
    p = project
    out = [(MODEL_NAMES[m], p.map_file(m), "mean") for m in MODELS if p.map_file(m).exists()]
    out += [(f"{b} ({BENCHMARK_REFS[b]})", p.mosaic(b), "mean") for b in BENCHMARKS if p.mosaic(b).exists()]
    if p.mosaic("ALS").exists():
        out.append(("ALS reference", p.mosaic("ALS"), "mean"))
    if p.mosaic("GEDI").exists():
        out.append(("GEDI labels (rh95)", p.mosaic("GEDI"), "max"))
    return out


def plan_map(project):
    """folium map of the study area and, once planned, its cells (study area / surrounding / dropped)."""
    import pandas as pd
    from . import viz
    aoi = viz.aoi_geometry(project.aoi_file)
    cells = pd.read_csv(project.cells_file) if project.cells_file.exists() else None
    epsg = json.loads(project.grid_file.read_text())["epsg"] if project.grid_file.exists() else None
    return viz.study_area_map(aoi, cells, epsg)


RANGE_Q = (2.0, 98.0)             # automatic colour scale: these percentiles of the layers' pixels


def colour_range(arrays, vmin=None, vmax=None, q=RANGE_Q):
    """(vmin, vmax) of the common colour scale: given values, else the q percentiles of all finite pixels of the
    layers pooled (each layer weighted equally), rounded outwards to whole metres - the scale spans the heights
    that occur in the study area instead of 0 to the tallest outlier."""
    if vmin is not None and vmax is not None:
        return float(vmin), float(vmax)
    import numpy as np
    rng = np.random.default_rng(0)
    pooled = []
    for a in arrays:
        v = a[np.isfinite(a)]
        if v.size:
            pooled.append(rng.choice(v, 200_000) if v.size > 200_000 else v)
    if not pooled:
        return (0.0 if vmin is None else float(vmin)), (1.0 if vmax is None else float(vmax))
    lo, hi = np.percentile(np.concatenate(pooled), q)
    lo = max(0.0, float(np.floor(lo))) if vmin is None else float(vmin)      # heights: never below 0 by default
    hi = float(np.ceil(hi)) if vmax is None else float(vmax)
    return lo, max(hi, lo + 1.0)


def _mask(project, clip):
    m = project.mosaic("aoi_mask")
    return str(m) if clip and m.exists() else None


def results_map(project, clip=True, vmin=None, vmax=None, height=600):
    """folium map of every result layer on one colour scale (colour_range: automatic unless vmin / vmax given)."""
    from . import viz
    mask = _mask(project, clip)
    layers = result_layers(project)
    if not layers:
        raise FileNotFoundError("no maps yet: the benchmark mosaics appear after the stack stage, the model maps "
                                "after predict")
    warped = [(name, *viz.warp(str(f), max_px=MAX_PX, how=how, mask_path=mask)) for name, f, how in layers]
    lo, hi = colour_range([a for name, a, _ in warped if not name.startswith("GEDI")], vmin, vmax)
    overlays = [(name, viz.png_data_url(viz.colorize(a, hi, vmin=lo)), b) for name, a, b in warped]
    aoi = viz.aoi_geometry(project.aoi_file) if project.aoi_file.exists() else None
    return viz.results_map(overlays, hi, aoi, caption=f"Canopy height (m), input {project['input']}", height=height,
                           vmin=lo)


def compare_figure(project, clip=True, vmin=None, vmax=None, max_px=800, ncols=4):
    """matplotlib figure: every model map and benchmark side by side on one colour scale (colour_range), each
    titled with its study-area mean and median, for comparing the maps by eye (the GEDI labels are too sparse to
    show at this scale and are left out)."""
    import matplotlib.pyplot as plt
    import numpy as np
    from . import viz
    mask = _mask(project, clip)
    panels = []
    for name, f, how in result_layers(project):
        if name.startswith("GEDI"):
            continue
        a, tr, _ = viz.read_band(str(f), mask_path=mask, max_px=max_px, how=how)
        ok = np.isfinite(a)
        if ok.any():                                     # crop to the pixels with values
            r, c = np.where(ok.any(1))[0], np.where(ok.any(0))[0]
            a = a[r[0]:r[-1] + 1, c[0]:c[-1] + 1]
        panels.append((name, a, abs(tr.a)))
    if not panels:
        raise FileNotFoundError("no maps yet: run the predict stage")
    lo, hi = colour_range([a for _, a, _ in panels], vmin, vmax)
    n = len(panels)
    ncols = min(ncols, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows + 0.6), squeeze=False,
                             constrained_layout=True)
    im = None
    for ax, (name, a, px) in zip(axes.flat, panels):
        im = ax.imshow(a, cmap=viz.CMAP, vmin=lo, vmax=hi, interpolation="nearest")
        v = a[np.isfinite(a)]
        stats = f"mean {v.mean():.1f} m, median {np.median(v):.1f} m" if v.size else "no data"
        ax.set_title(f"{name}\n{stats}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.add_artist(_scalebar(ax, a.shape[1], px))
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    cb = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.04, pad=0.02, aspect=50, extend="both")
    cb.set_label(f"Canopy height (m), input {project['input']}; colour scale {lo:g}-{hi:g} m "
                 f"({'set' if vmin is not None or vmax is not None else 'automatic: 2nd-98th percentile'})")
    return fig


def _scalebar(ax, width_px, px_m):
    """1, 2 or 5 x 10^k km bar of about a quarter of the panel width (lower left)."""
    import math
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    target = width_px * px_m / 4
    k = 10 ** math.floor(math.log10(target))
    length = max(s * k for s in (1, 2, 5) if s * k <= target)
    label = f"{length / 1000:g} km" if length >= 1000 else f"{length:g} m"
    return AnchoredSizeBar(ax.transData, length / px_m, label, "lower left", pad=0.3, frameon=True, size_vertical=2)


def metrics_table(project):
    """report/metrics.csv as a DataFrame (None before the report stage)."""
    import pandas as pd
    f = project.path("report", "metrics.csv")
    return pd.read_csv(f) if f.exists() else None


# measured on test projects (MB per 2.56 km cell) and a 439-cell training region (RF-SLS: ~11 kB per fitted
# pixel, ~870 fitted pixels per cell)
LAYER_MB = {"Embedding": 26.0, "S1": 0.9, "S2": 0.82, "DEM": 0.02, "GEDI": 0.02, "S1_asc": 0.9, "S2_season": 3.3}
STACK_MB = {"x": 10.0, "GEDI": 0.26, "seasonal": 5.5}
RF_KB_PER_PIXEL, RF_PIXELS_PER_CELL = 11.0, 870
FIXED_GB = 0.6                     # UNets, mosaics, maps, report


def storage_estimate(project, log=print):
    """Expected size (GB) of the finished project on disk, from the planned cells; warns if Google Drive has less
    free space."""
    from .project import input_layers
    from .workflow import read_cells
    n = sum(r["use"] for r in read_cells(project))
    mb = 0.0
    for layer in input_layers(project["input"]) + ["GEDI"]:
        key = "S1_asc" if layer.startswith("S1_asc") else "S2_season" if layer.startswith("S2_") else layer
        mb += LAYER_MB.get(key, 0.0)
    mb += STACK_MB["x"] + STACK_MB["GEDI"] + (STACK_MB["seasonal"] if project.seasonal else 0.0)
    rf = 0.0
    if "rf-sls" in project["models"]:
        rf = min(project["rf_max_rows"], RF_PIXELS_PER_CELL * n) * RF_KB_PER_PIXEL / 1e6
    gb = n * mb / 1e3 + rf + FIXED_GB
    try:
        free = _gb(shutil.disk_usage(project.root).free) + sum(storage(project).values())
    except OSError:
        free = None
    if free is not None and gb > 0.9 * free:
        log(f"WARNING: about {gb:.0f} GB needed but only about {free:.0f} GB available at {project.root}. Use fewer "
            "training cells (train_cells) or more Drive storage (the Earth embedding is most of the size; input A, "
            "without it, needs about a third).")
    return gb


def storage(project):
    """GB per top-level folder of the project."""
    out = {}
    for d in sorted(project.root.iterdir()):
        if d.is_dir():
            out[d.name] = _gb(sum(f.stat().st_size for f in d.rglob("*") if f.is_file()))
    return out
