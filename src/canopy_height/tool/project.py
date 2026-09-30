"""Project folder of the canopy-height tool: settings (project.yaml), file layout, stage status and logging.

A project is one folder. Every stage reads and writes only inside it (imported cells may point to rasters
elsewhere, see `cell_raster`), so a run can be stopped and resumed at any time:

    project.yaml                 settings (DEFAULTS below, overridden by the user)
    aoi.geojson                  study area as given (dissolved, EPSG:4326)
    plan/grid.json               analysis grid: epsg, origin x0 / y1 (upper-left corner), res 10 m, cell 256 px
    plan/cells.csv               one row per 2.56 km cell (see grid.py): cell, x0, y1, role (aoi / ring), ...
    raw/<cell>_<layer>.tif       downloaded layers (file layer names in LAYER_FILES)
    stacks/train_partNNN.npy     76-band uint16 model input of the training cells, parts of <= 400 chips
    stacks/train_GEDI_partNNN.npy  gridded GEDI rh95 labels, float32, -999 = no label
    stacks/train_index.csv       chip -> cell
    mosaic/<layer>.tif           study-area mosaics: annual (76-band uint16), GEDI, HRCH, GFCH, GMTCH, ALS
    models/<model>/              one training run per model (model.pth or rf_sls.joblib, result.json, ...)
    maps/<model>.tif             canopy-height maps (float32 m, NaN outside the study area / without input)
    report/                      summary.json, metrics.csv, quick-look PNGs
    logs/run.log                 log of every stage
    status.json                  state of every stage (pending / running / done / failed)
"""
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml

from ..channels import ALIASES, _Inputs, canonical as _canonical  # noqa: F401  (ALIASES re-exported)

STAGES = ["plan", "download", "stack", "train", "predict", "report"]
MODELS = ["rf-sls", "unet-sls", "kg-unet1", "kg-unet2"]
MODEL_NAMES = {"rf-sls": "RF-SLS", "unet-sls": "UNet-SLS", "kg-unet1": "KG-UNet1", "kg-unet2": "KG-UNet2"}

# Layers as they are named in raster files (the same names as the paper's data pipeline, so rasters of that
# pipeline can be imported unchanged) and the published products behind the benchmark layers.
X_LAYERS = ["Embedding", "DEM", "S1", "S2"]          # -> 76-band model input (64 + 1 + 2 + 9)
LABEL_LAYER = "GEDI"                                 # gridded GEDI rh95 (training labels)
BENCHMARKS = {                                       # product -> file layer (GMTCH is derived from Tolan_1m)
    "GMTCH": "GMTCH",   # Meta / Tolan et al. (2024), 1 m, aggregated to the 10 m p90
    "GFCH": "UMD",      # UMD / Potapov et al. (2021), 30 m
    "HRCH": "ETH",      # ETH / Lang et al. (2023), 10 m
}
# Input representations (canopy_height.channels.INPUTS) the tool offers: the UNet-ALS checkpoint pre-trained on
# 3DEP with that representation (start / ALS teacher of KG-UNet1/2), the annual layers it needs and whether it
# needs the seasonal composites. Layers a representation does not use are not downloaded (their channels of the
# 76-band annual raster stay 0 = no data, which the model does not read). DEM always comes with the GEDI request.
# Names: A = annual Sentinel-1/2 + DEM, E = Earth embedding, T = four seasonal (temporal) Sentinel-1/2 composites
# + DEM; AE and A were called IE and I before, and those names are still accepted (canonical_input).
INPUTS = _Inputs({
    "AE": dict(checkpoint="UNet-ALS.pth", annual=["Embedding", "DEM", "S1", "S2"], seasonal=False,
               label="Earth embedding + annual Sentinel-1/2 + DEM (paper's main results)"),
    "A": dict(checkpoint="UNet-A-ALS.pth", annual=["DEM", "S1", "S2"], seasonal=False,
              label="annual Sentinel-1/2 + DEM (no embedding)"),
    "E": dict(checkpoint="UNet-E-ALS.pth", annual=["Embedding", "DEM"], seasonal=False,
              label="Earth embedding only (no Sentinel-1: cheapest download)"),
    "T": dict(checkpoint="UNet-T-ALS.pth", annual=["DEM"], seasonal=True,
              label="four-season Sentinel-1/2 + DEM (no embedding)"),
    "TE": dict(checkpoint="UNet-TE-ALS.pth", annual=["Embedding", "DEM"], seasonal=True,
               label="Earth embedding + four-season Sentinel-1/2 + DEM"),
})


def canonical_input(name):
    """Current name of an input representation: the earlier names IE and I -> AE and A (channels.ALIASES)."""
    return _canonical(name)
# seasonal composites as in the paper's seasonal pipeline: season k = DJF, MAM, JJA, SON of the
# window [Dec (year - 1), Dec year); S1 (VV, VH) in dB, S2 (B2 ... B12) as 0-1 reflectance, float64 rasters
SEASONAL_LAYERS = [f"S1_asc_{k}" for k in range(4)] + [f"S2_{k}" for k in range(4)]


def input_layers(inp):
    """Raster layers a representation needs: its annual layers, then the seasonal ones."""
    spec = INPUTS[inp]
    return list(spec["annual"]) + (list(SEASONAL_LAYERS) if spec["seasonal"] else [])


BENCHMARK_SOURCES = {
    "GMTCH": "Meta high-resolution canopy height (Tolan et al. 2024), 1 m, 90th percentile of the 10 x 10 cells",
    "GFCH": "UMD global forest canopy height 2019 (Potapov et al. 2021)",
    "HRCH": "ETH global canopy height 2020 (Lang et al. 2023)",
}

DEFAULTS = dict(
    name=None,                      # free text
    input="AE",                     # input representation (INPUTS); decides the UNet-ALS start and the downloads
    year=2020,                      # year of the annual inputs (Earth embedding, Sentinel-1, Sentinel-2)
    epsg=None,                      # analysis CRS; None = UTM zone of the study-area centroid
    gee_project=None,               # Google Cloud project registered for Earth Engine (required for download)
    gedi_window=["2019-01-01", "2021-12-31"],   # GEDI L2A monthly rasters used as labels (end exclusive)
    built_up_mask=True,             # drop GEDI labels on WorldCover-2020 built-up cells
    train_cells=400,                # training region = study-area cells + surrounding cells up to this many
    ring_max_km=50,                 # never go further than this from the study area for training cells
    water_max=0.95,                 # surrounding cells with >= this WorldCover-2020 water share are dropped
    benchmarks=list(BENCHMARKS),    # published products downloaded for comparison
    models=list(MODELS),            # models to train, in this order
    base_model=None,                # UNet-ALS checkpoint (start / ALS teacher of KG-UNet1/2); None = bundled
    train_overrides={},             # per model: settings passed to the training recipe, e.g. {"kg-unet2": {"epochs": 30}}
    rf_max_rows=2000000,            # RF-SLS fits at most this many labelled pixels (memory bound; paper sets are smaller)
    device=None,                    # "cuda:0", "cpu", ... ; None = GPU if available
    workers=3,                      # parallel Earth Engine requests
    s1_method="local",              # Sentinel-1 composites: "local" = raw scenes from Earth Engine, the gee_s1_ard
                                    # chain in numpy (~1/10 of the EECU, ~1e-5 dB from the paper's rasters);
                                    # "gee" = the chain on Earth Engine (the paper's pipeline, bit for bit)
    als=None,                       # optional: list of ALS canopy-height rasters for an accuracy check
    als_resolution=1.0,             # resolution of those rasters (1 m -> 10 m p90; 10 m on the grid -> used as is)
    als_scale=1.0,                  # factor to metres (0.01 for canopy-height rasters stored in cm)
)

PART = 400                          # chips per stack part file
CELL_PX = 256                       # cell / chip size (pixels)
RES = 10.0                          # grid resolution (m)
CELL_M = CELL_PX * RES              # 2560 m
# the UNet-ALS checkpoints are assets of this GitHub release (<WEIGHTS_URL>/<file>)
WEIGHTS_URL = "https://github.com/Link-dev/chm-tool/releases/download/v1.0.0"


def default_base_model(inp="AE"):
    """UNet-ALS checkpoint of a representation: $CHM_WEIGHTS/source/<file> or weights/source/<file> of the
    repository (downloaded from WEIGHTS_URL; $CHM_UNET_ALS overrides the AE checkpoint)."""
    inp = canonical_input(inp)
    name = INPUTS[inp]["checkpoint"]
    if inp == "AE" and os.environ.get("CHM_UNET_ALS"):
        return Path(os.environ["CHM_UNET_ALS"])
    if os.environ.get("CHM_WEIGHTS"):
        return Path(os.environ["CHM_WEIGHTS"]) / "source" / name
    here = Path(__file__).resolve()
    for up in here.parents:
        cand = up / "weights" / "source" / name
        if cand.exists():
            return cand
    return None


class Project:
    def __init__(self, root):
        self.root = Path(root).resolve()
        f = self.root / "project.yaml"
        if not f.exists():
            raise FileNotFoundError(f"{self.root} is not a project folder (no project.yaml)")
        user = yaml.safe_load(open(f, encoding="utf-8")) or {}
        unknown = set(user) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown settings in project.yaml: {sorted(unknown)}")
        self.cfg = copy.deepcopy(DEFAULTS)
        self.cfg.update(user)
        if self.cfg.get("input") in INPUTS:            # earlier names (IE, I) -> current ones (AE, A)
            self.cfg["input"] = canonical_input(self.cfg["input"])
        self._log = None

    # ------------------------------------------------------------------------------------------ create / save
    @classmethod
    def create(cls, root, aoi, **settings):
        """New project folder from a study-area file (shapefile / .zip / GeoJSON / GeoPackage / KML) or a
        GeoJSON-like dict / shapely geometry in EPSG:4326."""
        from . import grid
        root = Path(root).resolve()
        if (root / "project.yaml").exists():
            raise FileExistsError(f"{root} already holds a project")
        unknown = set(settings) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown settings: {sorted(unknown)}")
        validate({**DEFAULTS, **{k: v for k, v in settings.items() if v is not None}})
        root.mkdir(parents=True, exist_ok=True)
        grid.write_aoi(aoi, root / "aoi.geojson")
        cfg = {k: v for k, v in settings.items() if v is not None}
        if cfg.get("input") in INPUTS:
            cfg["input"] = canonical_input(cfg["input"])
        with open(root / "project.yaml", "w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
        return cls(root)

    def save(self):
        user = {k: v for k, v in self.cfg.items() if v != DEFAULTS[k]}
        with open(self.root / "project.yaml", "w", encoding="utf-8") as fh:
            yaml.safe_dump(user, fh, sort_keys=False, allow_unicode=True)

    def __getitem__(self, k):
        return self.cfg[k]

    # ------------------------------------------------------------------------------------------ paths
    def path(self, *parts, mkdir=False):
        p = self.root.joinpath(*parts)
        if mkdir:
            p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def aoi_file(self):
        return self.path("aoi.geojson")

    @property
    def grid_file(self):
        return self.path("plan", "grid.json")

    @property
    def cells_file(self):
        return self.path("plan", "cells.csv")

    def raw(self, cell, layer):
        """Download target of a layer of a cell."""
        return self.path("raw", f"{cell}_{layer}.tif")

    def cell_raster(self, cell_row, layer):
        """Raster of a layer of a cell: the cell's own folder (imported cells: column src_dir, and lab_dir for
        GEDI) or raw/; a layer an imported cell lacks is looked for in raw/ (where the download stage puts it).
        Returns a Path (which may not exist yet)."""
        d = None
        if layer == LABEL_LAYER or layer.startswith("GEDI"):
            d = _col(cell_row, "lab_dir")
        d = d or _col(cell_row, "src_dir")
        raw = self.raw(cell_row["cell"], layer)
        if d:
            p = Path(d) / f"{cell_row['cell']}_{layer}.tif"
            if p.exists() or not raw.exists():
                return p
        return raw

    def stack_files(self, key="x"):
        """Existing parts of the training stack, in order (key 'x' = inputs, 'GEDI' = labels)."""
        d = self.path("stacks")
        pat = "train_part*.npy" if key == "x" else f"train_{key}_part*.npy"
        return sorted(str(p) for p in d.glob(pat)) if d.exists() else []

    def stack_part(self, key, i):
        return self.path("stacks", f"train_part{i:03d}.npy" if key == "x" else f"train_{key}_part{i:03d}.npy")

    def mosaic(self, layer):
        return self.path("mosaic", f"{layer}.tif")

    def model_dir(self, model):
        return self.path("models", model)

    def model_file(self, model):
        return self.model_dir(model) / ("rf_sls.joblib" if model == "rf-sls" else "model.pth")

    def map_file(self, model):
        return self.path("maps", f"{MODEL_NAMES.get(model, model)}.tif")

    def base_model(self):
        """UNet-ALS checkpoint of the project's input representation (or base_model if set)."""
        p = Path(self.cfg["base_model"]) if self.cfg["base_model"] else default_base_model(self.cfg["input"])
        if p is None or not Path(p).exists():
            raise FileNotFoundError(f"UNet-ALS checkpoint for input {self.cfg['input']} "
                                    f"({INPUTS[self.cfg['input']]['checkpoint']}) not found: download it from "
                                    f"{WEIGHTS_URL}/{INPUTS[self.cfg['input']]['checkpoint']} into weights/source/ "
                                    f"of the repository, or set $CHM_WEIGHTS or base_model in project.yaml")
        return Path(p)

    @property
    def seasonal(self):
        """True if the input representation needs the seasonal composites."""
        return INPUTS[self.cfg["input"]]["seasonal"]

    # ------------------------------------------------------------------------------------------ status / log
    @property
    def status_file(self):
        return self.path("status.json")

    def status(self):
        st = {s: dict(state="pending") for s in STAGES}
        if self.status_file.exists():
            try:
                st.update(json.load(open(self.status_file, encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                pass
        return st

    def set_status(self, stage, state, message="", **extra):
        st = self.status()
        rec = dict(st.get(stage, {}))
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        rec.update(state=state, message=message, **extra)
        if state == "running":
            rec["started"] = now
            rec.pop("finished", None)
        elif state in ("done", "failed"):
            rec["finished"] = now
        st[stage] = rec
        tmp = self.status_file.with_suffix(f".json.{os.getpid()}.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(st, fh, indent=1)
        replace_retry(tmp, self.status_file)

    @property
    def log_file(self):
        return self.path("logs", "run.log")

    def logger(self):
        if self._log is None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            lg = logging.getLogger(f"chm-tool:{self.root}")
            if not lg.handlers:                  # one set of handlers per folder and process
                lg.setLevel(logging.INFO)
                lg.propagate = False
                fmt = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
                for h in (logging.FileHandler(self.log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
                    h.setFormatter(fmt)
                    lg.addHandler(h)
            self._log = lg
        return self._log

    def close_log(self):
        """Release the log file (Windows keeps open files locked)."""
        lg = logging.getLogger(f"chm-tool:{self.root}")
        for h in list(lg.handlers):
            h.close()
            lg.removeHandler(h)
        self._log = None

    def log(self, msg):
        self.logger().info(msg)


YEARS = (2019, 2024)   # Sentinel-2 L2A is global from Dec 2018, GEDI from Apr 2019; the paper used 2019 / 2020


def validate(cfg):
    """Raise ValueError for settings the tool cannot run with (checked at creation and before every run)."""
    if cfg.get("input") not in INPUTS:
        raise ValueError(f"input {cfg.get('input')!r}: choose from {list(INPUTS)}: "
                         + "; ".join(f"{k} = {v['label']}" for k, v in INPUTS.items()))
    y = cfg.get("year")
    if not isinstance(y, int) or not YEARS[0] <= y <= YEARS[1]:
        raise ValueError(f"year {y!r}: choose {YEARS[0]}-{YEARS[1]} (Sentinel-2 surface reflectance is global only "
                         f"from December 2018 and GEDI starts in April 2019; the paper used 2019 and 2020)")
    bad = [m for m in cfg.get("models") or [] if m not in MODELS]
    if bad or not cfg.get("models"):
        raise ValueError(f"models {cfg.get('models')!r}: choose from {MODELS}")
    bad = [b for b in cfg.get("benchmarks") or [] if b not in BENCHMARKS]
    if bad:
        raise ValueError(f"benchmarks {bad}: choose from {list(BENCHMARKS)}")
    if not isinstance(cfg.get("train_cells"), int) or cfg["train_cells"] < 1:
        raise ValueError(f"train_cells {cfg.get('train_cells')!r} must be a positive integer")
    if cfg.get("s1_method", "local") not in ("local", "gee"):
        raise ValueError(f"s1_method {cfg.get('s1_method')!r}: 'local' (raw Sentinel-1 scenes from Earth Engine, "
                         f"processed on this computer; default) or 'gee' (processed on Earth Engine, the paper's "
                         f"pipeline bit for bit, about 10 times the Earth Engine quota)")
    w = cfg.get("gedi_window")
    if not (isinstance(w, (list, tuple)) and len(w) == 2 and str(w[0]) < str(w[1])):
        raise ValueError(f"gedi_window {w!r} must be [start, end] dates, e.g. ['2019-01-01', '2021-12-31']")
    if cfg.get("epsg") is not None:
        from pyproj import CRS
        crs = CRS.from_epsg(int(cfg["epsg"]))
        unit = crs.axis_info[0].unit_name if crs.axis_info else ""
        if not crs.is_projected or unit not in ("metre", "meter"):
            raise ValueError(f"EPSG:{cfg['epsg']} is not a projected CRS in metres (axis unit {unit!r}); the grid "
                             f"is 10 m cells of 2.56 km - use a UTM zone (default) or another metric projection")


def replace_retry(src, dst, tries=40, wait=0.5):
    """os.replace that waits while another process (e.g. the app showing a map) holds dst open on Windows."""
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(wait)


_LOCK_DEPTH = {}


def _proc_start(pid):
    try:
        import psutil
        return psutil.Process(pid).create_time()
    except Exception:           # no such process (or no psutil)
        return None


def lock_owner(project):
    """{'pid', 'started', 'command'} of the live process running this project, else None (stale locks ignored)."""
    f = project.path("logs", "run.lock")
    try:
        rec = json.load(open(f, encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    t = _proc_start(int(rec.get("pid", -1)))
    if t is None or abs(t - float(rec.get("create_time", -1))) > 1.0:
        return None
    return rec


class run_lock:
    """Context manager: one process at a time per project (re-entrant within the process)."""

    def __init__(self, project, command=""):
        self.project, self.command = project, command
        self.key = str(project.root)

    def __enter__(self):
        if _LOCK_DEPTH.get(self.key, 0) == 0:
            owner = lock_owner(self.project)
            if owner and int(owner["pid"]) != os.getpid():
                raise RuntimeError(f"{self.project.root} is being run by process {owner['pid']} "
                                   f"(since {owner.get('started')}, {owner.get('command', '')}); wait for it or stop it")
            f = self.project.path("logs", "run.lock", mkdir=True)
            rec = dict(pid=os.getpid(), create_time=_proc_start(os.getpid()) or 0.0,
                       started=time.strftime("%Y-%m-%d %H:%M:%S"), command=self.command)
            with open(f, "w", encoding="utf-8") as fh:
                json.dump(rec, fh)
        _LOCK_DEPTH[self.key] = _LOCK_DEPTH.get(self.key, 0) + 1
        return self

    def __exit__(self, *exc):
        _LOCK_DEPTH[self.key] -= 1
        if _LOCK_DEPTH[self.key] == 0:
            try:
                self.project.path("logs", "run.lock").unlink()
            except OSError:
                pass
        return False


def _col(row, k):
    v = row.get(k) if hasattr(row, "get") else None
    if v is None:
        return None
    try:
        if v != v:           # NaN from pandas
            return None
    except TypeError:
        pass
    return str(v) if str(v).strip() else None
