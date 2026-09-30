"""Streamlit interface of the canopy-height tool (`chm-tool app` = `streamlit run .../tool/app.py [-- DIR]`).

A thin shell over the project folder (project.py) and the command line. The study area is uploaded or drawn and
becomes a project (`Project.create`); the settings, among them the input representation (project.INPUTS: AE, the
paper default, A, E, T, TE; chosen only when its UNet-ALS checkpoint is found), go to project.yaml
(`Project.save`); every stage runs as a background process `python -m canopy_height.tool run DIR [--from S --to S]`
in its own process group (Windows: with its own hidden console; POSIX: its own session), so the page never blocks
and a run survives closing the browser and the terminal of `chm-tool app`. One process at a time runs a project: a
run started in a terminal (logs/run.lock, or a stage that status.json records as running in a live process)
disables the run buttons, and *Stop* ends it too. The page only reads what the stages write (status.json, logs/,
plan/, maps/, mosaic/, report/, models/<model>/result.json).

PyTorch is never imported by the page and Earth Engine only by the *Check Earth Engine* handler (*Authenticate*
runs in a child process), so the page opens on a machine without credentials or GPU.
"""
import functools
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from canopy_height import channels
    from canopy_height.tool import viz
    from canopy_height.tool.project import (BENCHMARKS, DEFAULTS, INPUTS, LABEL_LAYER, MODEL_NAMES, MODELS, STAGES,
                                            Project, default_base_model, input_layers, lock_owner, validate)
    from canopy_height.tool.project import YEARS as YEAR_RANGE
    from canopy_height.tool.project import WEIGHTS_URL
except ImportError:                  # run from a source tree without installing the package
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from canopy_height import channels
    from canopy_height.tool import viz
    from canopy_height.tool.project import (BENCHMARKS, DEFAULTS, INPUTS, LABEL_LAYER, MODEL_NAMES, MODELS, STAGES,
                                            Project, default_base_model, input_layers, lock_owner, validate)
    from canopy_height.tool.project import YEARS as YEAR_RANGE
    from canopy_height.tool.project import WEIGHTS_URL

MARKER = "canopy_height.tool"        # every run started here has this in its command line
TAIL = 40
MAX_PX = 1024                        # overlay size on the results map
MAX_DOWNLOAD_MB = 200                # larger result files are not offered in the browser (their path is shown)
MAX_HEIGHT = 150.0                   # upper end of the colour-scale input unless a layer goes higher
YEARS = list(range(YEAR_RANGE[0], YEAR_RANGE[1] + 1))
CONFIRM_CELLS = 100                  # a download needs an explicit confirmation above this many study-area cells
CONFIRM_EECU_H = 50                  # ... or this many EECU-hours still to spend
CELL_KM2 = 2.56 * 2.56
N_ANNUAL, N_SEASONAL = len(channels.ANNUAL_BANDS), len(channels.SEASONAL_BANDS)     # 76, 44 uint16 bands
KG_MODELS = ("kg-unet1", "kg-unet2")                 # start from the UNet-ALS checkpoint of the representation
# rough cost of one seasonal composite of one cell (MB on disk, EECU-s) where download.COST has none: the paper's
# seasonal rasters (float64, deflate) take about 0.8 MB (S1) and 3.1 MB (S2) per cell and season; the four S1
# seasons together cost about as much Earth Engine compute as the annual S1
SEASONAL_COST = {**{f"S1_asc_{k}": (0.8, 250) for k in range(4)}, **{f"S2_{k}": (3.1, 15) for k in range(4)}}
AUTH_TIMEOUT = 300                   # s to wait for the Google sign-in
UPLOAD_TYPES = ["zip", "shp", "shx", "dbf", "prj", "cpg", "geojson", "json", "gpkg", "kml"]
MAIN_EXT = [".shp", ".zip", ".gpkg", ".geojson", ".json", ".kml"]
BENCHMARK_REFS = {"GMTCH": "Meta, Tolan et al. 2024", "GFCH": "UMD, Potapov et al. 2021",
                  "HRCH": "ETH, Lang et al. 2023"}
BENCHMARK_RES = {"GMTCH": "1 m, 10 m 90th percentile", "GFCH": "30 m", "HRCH": "10 m"}
DEVICES = ["auto", "cuda:0", "cpu"]
ALS_RES = {1.0: "1 m (aggregated to the 10 m 90th percentile, as in the paper)",
           10.0: "10 m on the analysis grid (used as is)"}
ALS_SCALE = {1.0: "metres (factor 1)", 0.01: "centimetres (factor 0.01)"}
_PROCS = {}                          # Popen objects of the runs started by this server process

TRAIN_CELLS_NOTE = (
    "As at the paper's international sites, the local models are trained on about 400 chips of 2.56 x 2.56 km "
    "(256 x 256 pixels at 10 m): the cells of the study area plus a ring of surrounding cells, widened 1 km at a "
    "time until this number is reached (cells that are almost all water are skipped). GEDI labels are sparse, so "
    "small numbers (tens of cells) give unreliable models. Used by the Plan stage.")
S1_METHOD_LABELS = {
    "local": "On this computer from the raw scenes (default; about 10-50 EECU-seconds per cell)",
    "gee": "On Earth Engine, as the paper's pipeline (bit for bit; about 150-500 EECU-seconds per cell or more)"}
S1_METHOD_NOTE = (
    "Sentinel-1 (inputs AE, A, T, TE) is filtered for speckle and flattened for terrain by the gee_s1_ard chain. "
    "*On this computer*: Earth Engine only serves the raw scenes (about 50 MB per cell pass through, not stored) "
    "and the chain runs here, about a tenth of the Earth Engine compute; the result differs from the paper's by "
    "about 1e-5 dB (under 0.01 % of the model's Sentinel-1 values move by one 0.01 dB step). *On Earth Engine*: the "
    "paper's pipeline unchanged. Cells where the local way does not apply (a CRS other than a WGS84 UTM zone) take "
    "the Earth Engine way. Existing downloads are kept when this is changed.")
YEAR_NOTE = (
    f"{YEARS[0]}-{YEARS[-1]} only: Sentinel-2 surface reflectance (L2A) covers the globe only from December 2018 "
    "and GEDI starts in April 2019 (the paper used 2019 and 2020). Satellite embedding, Sentinel-1 and Sentinel-2 "
    "of this year (default 2020, as at the paper's international sites); GEDI labels always from 2019-2021.")
AUTH_NOTE = (
    "*Authenticate* opens the Google sign-in in a new browser tab on the computer that runs this app (also when "
    "credentials exist, so it can switch accounts) and stores the new Earth Engine credentials there. The page "
    f"waits for the sign-in and gives up after {AUTH_TIMEOUT // 60} minutes. On a remote machine run "
    "`earthengine authenticate --auth_mode=notebook` in a terminal instead.")
INPUT_NOTE = (
    "Sentinel-1 takes more than 90 % of the Earth Engine compute of a cell (speckle filtering and terrain "
    "flattening of every scene). AE and A need the annual "
    "Sentinel-1 composite; T and TE need the four seasonal Sentinel-1 composites (together about the compute of "
    "the annual one) plus four seasonal Sentinel-2 composites. Layers a representation does not use are not "
    "downloaded. All four models are trained on the chosen input; KG-UNet1/2 start from its UNet-ALS checkpoint "
    "(pre-trained on US airborne lidar with the same input). AE is the default.")
GEDI_NOTE = (
    "No agreement with GEDI in this report is an independent accuracy: the local models were trained on these "
    "GEDI labels (the chips of the GEDI validation are held out from the UNets' training but hold labels of the "
    "same product and region, and RF-SLS is scored in-sample), and HRCH (ETH) and GFCH (UMD) were calibrated with "
    "GEDI data of the same period. ALS rasters (tab 2) give an independent check.")


def _input_label(inp):
    return f"{inp}: {INPUTS[inp]['label']}" if inp in INPUTS else f"{inp} (unknown representation)"


def _help_inputs():
    """Markdown table of the input representations for the Help tab."""
    rows = [f"| **{k}** | {v['label']} | `{v['checkpoint']}` | {', '.join(v['annual'])}"
            f"{' + four seasonal S1 / S2 composites' if v['seasonal'] else ''} |" for k, v in INPUTS.items()]
    return "\n".join(["| input | model input | UNet-ALS checkpoint | downloaded layers (+ GEDI) |",
                      "|---|---|---|---|", *rows, "",
                      "Letters: **A** = annual Sentinel-1/2 + DEM, **E** = Earth embedding, **T** = four seasonal "
                      "(temporal) Sentinel-1/2 composites + DEM. AE and A were called IE and I in earlier versions; "
                      "those names are still accepted."])


HELP = f"""
### What the tool does
You give a study area. The tool lays a 10 m grid of 2.56 km cells (256 x 256 pixels) over it, downloads the
model inputs and the GEDI labels of these cells and of a ring of surrounding cells from Google Earth Engine,
trains the four local models of the paper with the paper's settings, and writes a canopy-height map of the
study area for every model, next to three published canopy-height products. Nothing needs tuning: the defaults
are the settings of the paper.

### Model inputs
The inputs of every 10 m pixel come from these layers of the chosen year ({YEARS[0]}-{YEARS[-1]}): the 64-band
Google satellite embedding (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`), SRTM elevation (DEM), Sentinel-1 VV / VH
(border-noise removal, multi-temporal speckle filter, terrain flattening; median in dB) and nine Sentinel-2 bands
(cloud-masked median). The annual composites are medians over the calendar year; the seasonal composites are
medians over the four seasons December-February, March-May, June-August and September-November, from December of
the previous year to November of the chosen year.

Which of them the models read is the **input representation** (tab 2; also at the creation of a project):

{_help_inputs()}

**AE** (the default) is the satellite embedding with the annual Sentinel-1 / 2
composites and the DEM. Every representation has its own UNet-ALS checkpoint, pre-trained on the same US airborne
lidar (USGS 3DEP) with that input, from which KG-UNet1/2 start; RF-SLS and UNet-SLS are trained on the same input.
Layers a representation does not use are not downloaded (the DEM always comes with the GEDI labels). Sentinel-1
takes more than 90 % of the Earth Engine compute; T and
TE need the four seasonal Sentinel-1 composites (together about the compute of the annual one) and four seasonal
Sentinel-2 composites. The representation can be changed after downloading: the next run downloads the missing
layers, rebuilds the training stack and trains the models again.

### The four models
- **RF-SLS**: a random forest (600 trees) that predicts canopy height pixel by pixel from the 76 input bands,
  trained on the local GEDI labels ("SLS": spaceborne-lidar supervision). It is the per-pixel baseline of the
  paper and runs on the CPU.
- **UNet-SLS**: a UNet trained from scratch on 256 x 256 chips with the local GEDI labels (100 epochs). Unlike
  the random forest it uses the spatial context of every pixel.
- **KG-UNet1**: starts from UNet-ALS, the paper's UNet pre-trained on US airborne lidar (USGS 3DEP), and
  fine-tunes its last two blocks on the local GEDI labels (50 epochs). The airborne-lidar knowledge guides the
  model where the GEDI labels are sparse.
- **KG-UNet2**: as KG-UNet1, with two teacher terms in the loss: agreement of the image gradients with UNet-ALS
  (spatial detail) and agreement with UNet-SLS after 4 x 4 average pooling (local height level). It therefore
  needs UNet-SLS of the same project, which is trained first.

### Benchmarks
- **GMTCH**: Meta high-resolution canopy height ([Tolan et al. 2024](https://doi.org/10.1016/j.rse.2023.113888)), 1 m, aggregated to the 90th percentile of
  every 10 m pixel.
- **GFCH**: UMD global forest canopy height 2019 ([Potapov et al. 2021](https://doi.org/10.1016/j.rse.2020.112165)), 30 m.
- **HRCH**: ETH global canopy height 2020 ([Lang et al. 2023](https://doi.org/10.1038/s41559-023-02206-6)), 10 m.

### GEDI labels
GEDI is NASA's spaceborne lidar on the International Space Station. The tool uses the gridded GEDI L2A shots
(`LARSE/GEDI/GEDI02_A_002_MONTHLY`): relative height rh95 of good-quality, non-degraded shots, median over
2019-2021, without built-up land (ESA WorldCover 2020). They cover a few percent of the pixels and are the only
local height reference used in training.

**No agreement with GEDI in the report is an independent accuracy.** The local models were trained on these
labels (the chips of the GEDI validation are held out from the UNets' training, but they hold labels of the same
GEDI product and region, and RF-SLS is scored in-sample), and HRCH (ETH) and GFCH (UMD) were calibrated with GEDI
data of the same period. ALS rasters (tab 2) give an independent accuracy.

### Requirements
- A **Google Cloud project registered for Earth Engine** (https://code.earthengine.google.com/register;
  free for noncommercial and research use, with a monthly compute quota), and the Earth Engine API
  authenticated on this computer (tab 2, *Authenticate*, once; or `earthengine authenticate` in a terminal).
- **Quota**: Sentinel-1 dominates. Processed on this computer from the raw scenes (the default, tab 2) it takes
  about 10 EECU-seconds per cell for the annual composite (AE, A) with ~30 acquisitions a year and about 45 with
  ~150 (two orbits, two satellites), about as much for the four seasonal ones (T, TE), plus 50-250 MB of raw
  scenes per cell passing through the network; processed on Earth Engine as in the paper, about 10 times that
  (150-500 EECU-seconds per cell). E needs about 10 EECU-seconds per cell. Tab 3 shows what is still to
  download and asks for a confirmation before a large download. A project over its quota is slowed down (HTTP
  429) and the tool retries.
- **Disk**: tens of MB per cell for the downloads (the seasonal composites add about 16 MB) plus about 10 MB per
  cell for the training stack (16 MB with the seasonal composites), several GB for the default 400 cells.
- **UNet-ALS checkpoint** of the chosen representation in `weights/source/` of the repository (or
  `$CHM_WEIGHTS/source/`), downloaded from {WEIGHTS_URL}/: KG-UNet1/2 start from it.
- **GPU**: an NVIDIA GPU with CUDA for the three UNets (a CPU works, but slowly); RF-SLS uses the CPU.

### Expected run times (rough, 400 training cells, one recent GPU)
| stage | time |
|---|---|
| Plan | minutes (WorldCover water shares from Earth Engine) |
| Download | hours; depends on the Earth Engine load and quota, resumable |
| Stack | minutes |
| Train | RF-SLS 5-30 min; each UNet 5-30 min on a GPU, hours on a CPU |
| Predict, Report | minutes |

Every stage is resumable: a stopped run is continued by starting it again. A run goes on when this page or the
terminal of `chm-tool app` is closed. One run at a time per project: while a run started here or in a terminal
(`chm-tool run DIR`) is going, the run buttons are disabled.

### Citation
Zhou et al. (in review), Remote Sensing of Environment.
"""


# ---------------------------------------------------------------------------------------------- background runs
def run_command(root, stage=None):
    """Command line of a run of all stages (stage None) or of one stage."""
    cmd = [sys.executable, "-m", "canopy_height.tool", "run", str(Path(root).resolve())]
    return cmd + ["--from", stage, "--to", stage] if stage else cmd


def pid_file(root):
    return Path(root) / "logs" / "app_run.pid"


def process_log(root):
    return Path(root) / "logs" / "app_process.log"


def _child_env():
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    src = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH")) if p)
    return env


def _spawn(cmd, **kw):
    """Popen of a background run. Windows: new process group with its own hidden console (CREATE_NO_WINDOW; not
    DETACHED_PROCESS, under which every DataLoader worker would open a visible console window), outside the app's
    job object where the job allows it; POSIX: new session. Closing the app's terminal does not end the run."""
    if os.name != "nt":
        return subprocess.Popen(cmd, start_new_session=True, **kw)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    try:
        return subprocess.Popen(cmd, creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB, **kw)
    except OSError as e:
        if getattr(e, "winerror", None) != 5:        # ERROR_ACCESS_DENIED: the job does not allow breakaway
            raise
        return subprocess.Popen(cmd, creationflags=flags, **kw)


def start_run(root, stage=None, cmd=None):
    """Start a run (all stages, or one stage) as a background process -> PID (logs/app_run.pid); stdout / stderr
    go to logs/app_process.log. Refuses while any process runs the project (`active_run`). `cmd` overrides the
    command (tests)."""
    root = Path(root).resolve()
    run = active_run(root)
    if run is not None:
        raise RuntimeError(f"this project is already being run by process {run['pid']} ({run['what']}"
                           + (f", since {run['started']}" if run.get("started") else "") + "): wait for it or stop it")
    cmd = list(cmd or run_command(root, stage))
    log = process_log(root)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as out:
        out.write(f"\n# {time.strftime('%Y-%m-%d %H:%M:%S')} {subprocess.list2cmdline(cmd)}\n")
        out.flush()
        proc = _spawn(cmd, cwd=root, env=_child_env(), stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
    _PROCS[proc.pid] = proc
    pid_file(root).write_text(f"{proc.pid}\n", encoding="ascii")
    return proc.pid


def _proc_info(pid):
    """(alive, creation time in s since the epoch or None, command line list or None) of a process."""
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            p = psutil.Process(pid)
            if p.status() == psutil.STATUS_ZOMBIE:
                return False, None, None
            created = p.create_time()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False, None, None
        try:
            cmd = p.cmdline()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            cmd = None
        return True, created, cmd
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        h = k32.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False, None, None
        try:
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(h, ctypes.byref(code)) or code.value != 259:   # STILL_ACTIVE
                return False, None, None
            t = [ctypes.c_ulonglong() for _ in range(4)]
            created = None
            if k32.GetProcessTimes(h, *[ctypes.byref(x) for x in t]):
                created = t[0].value / 1e7 - 11644473600.0
            return True, created, None
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False, None, None
    except PermissionError:
        return True, None, None
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").split("\0")
    except OSError:
        cmd = None
    return True, None, cmd


def current_run(root):
    """PID of the run started from the app if it is still alive and still that process (creation time next to
    the time the PID file was written, command line with canopy_height.tool), else None."""
    f = pid_file(root)
    try:
        pid = int(f.read_text(encoding="ascii").split()[0])
        written = f.stat().st_mtime
    except (OSError, ValueError, IndexError):
        return None
    proc = _PROCS.get(pid)
    if proc is not None and proc.poll() is not None:
        return None
    alive, created, cmd = _proc_info(pid)
    if not alive:
        return None
    if created is not None and not (written - 120 <= created <= written + 5):
        return None                  # the PID now belongs to another process
    if cmd is not None and not any(MARKER in c for c in cmd):
        return None
    return pid


def _alive_since(pid, started):
    """The process is alive and was created before `started` (a status.json time): still the one that started
    the stage, not a later process with a reused PID."""
    alive, created, _ = _proc_info(pid)
    if not alive or created is None:
        return False
    try:
        t = time.mktime(time.strptime(str(started), "%Y-%m-%d %H:%M:%S"))
    except (ValueError, OverflowError):
        return False
    return created <= t + 2.0


def active_run(root):
    """The live process running this project -> dict(pid, started, what, ours) or None: the run started from
    this app (logs/app_run.pid), else the holder of logs/run.lock (e.g. `chm-tool run DIR` in a terminal), else a
    process that status.json records as running a stage and that is still that process."""
    root = Path(root).resolve()
    pid = current_run(root)
    if pid is not None:
        started = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_mtime(pid_file(root))))
        return dict(pid=pid, started=started, what="run started from this app", ours=True)
    try:
        p = Project(root)
    except Exception:                # no or unreadable project.yaml: nothing runs this folder as a project
        return None
    owner = lock_owner(p)
    if owner:
        return dict(pid=int(owner["pid"]), started=owner.get("started", ""),
                    what=owner.get("command") or "run outside this app", ours=False)
    for s, rec in p.status().items():
        pid = rec.get("pid")
        if rec.get("state") == "running" and isinstance(pid, int) and _alive_since(pid, rec.get("started")):
            return dict(pid=pid, started=rec.get("started", ""), what=f"{s} stage outside this app", ours=False)
    return None


def _gone(pid, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        proc = _PROCS.get(pid)
        if (proc is not None and proc.poll() is not None) or not _proc_info(pid)[0]:
            return True
        time.sleep(0.25)
    return False


def stop_run(root, timeout=10.0):
    """Stop the process running this project (`active_run`: started here or in a terminal) and its children ->
    PID stopped, or None. Stages left 'running' in status.json are marked failed."""
    root = Path(root).resolve()
    run = active_run(root)
    if run is None:
        return None
    pid = run["pid"]
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            top = psutil.Process(pid)
            procs = top.children(recursive=True) + [top]
        except psutil.NoSuchProcess:
            procs = []
        for q in procs:
            try:
                q.terminate()
            except psutil.NoSuchProcess:
                pass
        _, left = psutil.wait_procs(procs, timeout=timeout)
        for q in left:
            try:
                q.kill()
            except psutil.NoSuchProcess:
                pass
    elif os.name == "nt":            # a run has its own console, so CTRL_BREAK from here cannot reach it
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        kill = os.killpg if run["ours"] else os.kill
        try:
            kill(pid, signal.SIGTERM)
        except OSError:
            pass
        if not _gone(pid, timeout):
            try:
                kill(pid, signal.SIGKILL)
            except OSError:
                pass
    proc = _PROCS.pop(pid, None)
    if proc is not None:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    process_log(root).parent.mkdir(parents=True, exist_ok=True)
    with open(process_log(root), "a", encoding="utf-8") as out:
        out.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} stopped from the app (PID {pid}, {run['what']})\n")
    try:
        p = Project(root)
        for s, rec in p.status().items():
            if rec.get("state") == "running":
                p.set_status(s, "failed", "stopped from the app")
    except (OSError, ValueError):
        pass
    return pid


def tail(path, n=TAIL, max_bytes=256_000):
    """Last n lines of a text file ('' if it does not exist)."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - max_bytes))
            data = fh.read()
    except OSError:
        return ""
    return "\n".join(data.decode("utf-8", errors="replace").splitlines()[-n:])


# ---------------------------------------------------------------------------------------------- download size
def cell_layers(inp="AE"):
    """File layers downloaded for every used cell with input representation `inp`: its layers and the GEDI labels
    (the DEM always comes with them)."""
    return list(dict.fromkeys([*input_layers(inp), "DEM", LABEL_LAYER]))


def _layer_costs(benchmarks, inp="AE", s1_method="local"):
    """((MB, EECU-s) of one training cell, (MB, EECU-s) added by one study-area cell) from download.layer_cost (and
    SEASONAL_COST for seasonal layers it does not list)."""
    from canopy_height.tool.download import COST, layer_cost

    def cost(x):
        if x in COST:
            return layer_cost(x, s1_method)[:2]
        return SEASONAL_COST.get(x, (0, 0))
    extra = ["Tolan_1m" if b == "GMTCH" else BENCHMARKS[b] for b in benchmarks if b in BENCHMARKS]
    return tuple(tuple(sum(cost(x)[i] for x in ls) for i in (0, 1)) for ls in (cell_layers(inp), extra))


def rough_size(km2, train_cells, benchmarks, inp="AE", s1_method="local"):
    """Download size before the Plan stage: about km2 / 6.55 study-area cells, and as many cells in all as the
    larger of that and train_cells (the ring grows until train_cells is reached)."""
    n_aoi = max(1, math.ceil(km2 / CELL_KM2))
    n = max(n_aoi, int(train_cells))
    (mb_c, s_c), (mb_a, s_a) = _layer_costs(benchmarks, inp, s1_method)
    return dict(planned=False, cells=n, aoi_cells=n_aoi, ring_cells=n - n_aoi, jobs=None,
                mb=n * mb_c + n_aoi * mb_a, eecu_s=n * s_c + n_aoi * s_a)


def stack_mb(inp="AE"):
    """Training stack of one cell (MB): the 76 annual uint16 bands (+ 44 seasonal ones) and the float32 labels."""
    bands = N_ANNUAL + (N_SEASONAL if INPUTS.get(inp, {}).get("seasonal") else 0)
    return (bands * 2 + 4) * 256 * 256 / 1e6


def checkpoint(inp, base_model=None):
    """(UNet-ALS checkpoint KG-UNet1/2 would start from, found): project.yaml's base_model if set, else the
    representation's (project.default_base_model); (None, False) for an unknown representation."""
    if base_model:
        return Path(base_model), Path(base_model).exists()
    if inp not in INPUTS:
        return None, False
    p = default_base_model(inp)
    return p, p is not None and Path(p).exists()


def input_table():
    """What each input representation downloads and trains (a row per representation, AE first)."""
    rows = []
    for k, v in INPUTS.items():
        (_, s), _ = _layer_costs([], k)
        _, found = checkpoint(k)
        rows.append({"input": k, "model input": v["label"],
                     "UNet-ALS checkpoint": v["checkpoint"] + ("" if found else " (missing)"),
                     "annual layers": ", ".join(v["annual"]), "four seasons S1 / S2": "yes" if v["seasonal"] else "no",
                     "model channels": channels.INPUTS[k]["n_channels"], "EECU-s per cell (rough)": int(round(s, -1))})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, ttl=60, max_entries=32)
def _estimate_cached(root, stamp):
    from canopy_height.tool import download
    return download.estimate(Project(root))


def download_size(p):
    """What is still to download -> (dict(planned, cells, aoi_cells, ring_cells, jobs, mb, eecu_s), None) from
    download.estimate once the cells are planned, else rough_size of the study area; (None, error) if unknown."""
    try:
        if p.cells_file.exists():
            stamp = tuple(_mtime(f) for f in (p.cells_file, p.path("raw"), p.path("project.yaml")))
            return dict(_estimate_cached(str(p.root), stamp), planned=True), None
        return rough_size(viz.area_km2(viz.aoi_geometry(p.aoi_file)), p["train_cells"], p["benchmarks"] or [],
                          p["input"], p["s1_method"]), None
    except Exception as e:           # the estimate must not break the page
        return None, f"{type(e).__name__}: {e}"


def needs_confirmation(est):
    """Something is left to download and it is large (CONFIRM_CELLS study-area cells or CONFIRM_EECU_H)."""
    left = est.get("jobs") is None or est["jobs"] > 0
    return left and (est.get("aoi_cells", 0) > CONFIRM_CELLS or est.get("eecu_s", 0) / 3600 > CONFIRM_EECU_H)


def _disk(mb):
    return f"{mb / 1000:,.1f} GB" if mb >= 1000 else f"{mb:,.0f} MB"


def size_text(est, inp="AE"):
    cells = f"{est['cells']:,} cells ({est.get('aoi_cells', 0):,} in the study area, {est.get('ring_cells', 0):,} around it)"
    if est["planned"] and not est.get("jobs"):
        return f"Downloads complete for the {cells} (input {inp})."
    head = (f"**Still to download:** {est['jobs']:,} jobs for the {cells}" if est["planned"] else
            f"**Rough download size** (exact once the cells are planned): about {cells}")
    moved = (f" (plus about {_disk(est['transfer_mb'])} of raw Sentinel-1 scenes passing through, not stored)"
             if est.get("transfer_mb") else "")
    return (head + f", input {inp}: about {_disk(est['mb'])} on disk{moved} and {est['eecu_s'] / 3600:,.1f} "
            f"EECU-hours of Earth Engine compute; the training stack adds about {_disk(est['cells'] * stack_mb(inp))}.")


# ---------------------------------------------------------------------------------------------- helpers
def _k(p, name):
    """Widget key of one project (widgets reset when another project is opened)."""
    h = hashlib.md5(str(p.root).encode("utf-8")).hexdigest()[:8] if p is not None else "new"
    return f"{name}:{h}"


def _mtime(f):
    try:
        return Path(f).stat().st_mtime
    except (OSError, TypeError):
        return 0.0


def _folium(m, key, height=520, returned_objects=()):
    from streamlit_folium import st_folium
    return st_folium(m, key=key, height=height, use_container_width=True, returned_objects=list(returned_objects))


def _project():
    d = st.session_state.get("project_dir")
    if not d:
        return None
    try:
        return Project(d)
    except (OSError, ValueError) as e:
        st.sidebar.error(f"Cannot open {d}: {e}")
        return None


def _new_folder():
    parent = st.session_state.get("new_parent", "").strip().strip('"')
    name = st.session_state.get("new_name", "").strip()
    if not parent or not name:
        return None
    if any(c in name for c in '<>:"/\\|?*'):
        return None
    return Path(parent).expanduser() / name


def _cells(p):
    """(cells.csv DataFrame, grid EPSG) of a planned project, (None, None) before the Plan stage."""
    if not p.cells_file.exists():
        return None, None
    cells = pd.read_csv(p.cells_file)
    epsg = p["epsg"]
    if p.grid_file.exists():
        epsg = json.loads(p.grid_file.read_text(encoding="utf-8")).get("epsg", epsg)
    return cells, epsg


def _start(p, stage):
    try:
        pid = start_run(p.root, stage)
    except (RuntimeError, OSError) as e:
        st.error(f"Could not start the run: {e}")
        return
    _rerun_with(p, f"{time.strftime('%H:%M:%S')} started {stage or 'all stages'} (PID {pid}).")


def _rerun_with(p, msg):
    """Rerun the page showing msg once; the live view takes the new state as its baseline (no second rerun)."""
    st.session_state["run_msg"] = msg
    st.session_state.pop(f"_live:{p.root}", None)
    st.rerun()


# ---------------------------------------------------------------------------------------------- sidebar
def _init_state():
    ss = st.session_state
    goto = ss.pop("_goto_open", None)
    if goto:
        ss["mode"] = "Open existing"
        ss["open_path"] = goto
    if not ss.get("_init"):
        ss["_init"] = True
        start = next((a for a in sys.argv[1:] if Path(a).is_dir()), None) or os.environ.get("CHM_TOOL_PROJECT")
        if start:
            ss.setdefault("open_path", start)
            if (Path(start) / "project.yaml").exists():
                ss.setdefault("project_dir", str(Path(start).resolve()))


def _sidebar(p):
    sb = st.sidebar
    sb.title("Canopy height")
    sb.caption("Local canopy-height maps from GEDI, Sentinel-1/2 and satellite embeddings with the models of "
               "Zhou et al. (in review).")
    mode = sb.radio("Project folder", ["Open existing", "Create new"], key="mode", horizontal=True)
    if mode == "Open existing":
        path = sb.text_input("Folder", key="open_path", placeholder=r"D:\chm_projects\my_site")
        if sb.button("Open", disabled=not path.strip()):
            root = Path(path.strip().strip('"')).expanduser()
            if (root / "project.yaml").exists():
                st.session_state["project_dir"] = str(root.resolve())
                st.rerun()
            sb.error(f"No project.yaml in {root}")
    else:
        sb.text_input("Parent folder", key="new_parent", placeholder=r"D:\chm_projects")
        name = sb.text_input("Project name", key="new_name", placeholder="my_site")
        folder = _new_folder()
        if name.strip() and folder is None and st.session_state.get("new_parent", "").strip():
            sb.error('The name must not contain < > : " / \\ | ? *')
        if folder is not None:
            sb.caption(f"New project folder: {folder}")
            if (folder / "project.yaml").exists():
                sb.warning("This folder already holds a project: open it instead.")
        sb.caption("Then define the study area in tab 1 and press *Create project*.")
    sb.divider()
    if p is None:
        sb.caption("No project open.")
        return
    sb.markdown(f"**Open project:** {p['name'] or p.root.name}")
    sb.caption(str(p.root))
    sb.markdown(f"**Input representation:** {_input_label(p['input'])}")
    status = p.status()
    sb.caption("  \n".join(f"{s}: {status[s].get('state', 'pending')}" for s in STAGES))
    if sb.button("Close project"):
        st.session_state.pop("project_dir", None)
        st.rerun()


# ---------------------------------------------------------------------------------------------- 1 study area
def _tab_study_area(p):
    if p is not None:
        _project_area(p)
    else:
        _define_area()


def _project_area(p):
    if not p.aoi_file.exists():
        st.error(f"{p.aoi_file} is missing.")
        return
    aoi = viz.aoi_geometry(p.aoi_file)
    cells, epsg = _cells(p)
    c1, c2 = st.columns([3, 1])
    with c1:
        _folium(viz.study_area_map(aoi, cells, epsg), key=_k(p, "aoi_map"))
    with c2:
        st.metric("Study area", f"{viz.area_km2(aoi):,.1f} km²")
        st.metric("Input representation", str(p["input"]))
        st.caption(_input_label(p["input"]) + " (tab 2)")
        if cells is None:
            st.info("The 2.56 km cells of the study area and of the training ring are laid out by the Plan stage "
                    "(it looks up the water share of the surrounding cells in Earth Engine).")
            if st.button("Plan cells now", disabled=active_run(p.root) is not None, key=_k(p, "plan_now")):
                _start(p, "plan")
        else:
            counts = viz.cell_counts(cells)
            st.markdown("  \n".join(
                f'<span style="color:{viz.CATEGORY_COLORS[c]}">&#9632;</span> {c}: **{n}**'
                for c, n in counts.items()), unsafe_allow_html=True)
            st.caption("aoi: cells of the study area (mapped and used for training); ring: surrounding training "
                       "cells; dropped (water): surrounding cells that are almost all water.")
        est, err = download_size(p)
        st.markdown(size_text(est, p["input"]) if est else f"Download size not available ({err})")
    st.caption("The study area of a project is fixed; create a new project for another area.")


def _define_area():
    ss = st.session_state
    if ss.get("mode") != "Create new":
        st.info("Open a project folder in the sidebar, or choose *Create new* there to start a project from a "
                "study area.")
    how = st.radio("Study area from", ["Upload a file", "Draw on the map"], horizontal=True, key="aoi_how")
    if how == "Upload a file":
        files = st.file_uploader(
            "Shapefile (.shp with .shx, .dbf, .prj, or all in one .zip), GeoJSON, GeoPackage or KML",
            type=UPLOAD_TYPES, accept_multiple_files=True, key="aoi_files")
        if files:
            _read_upload(files)
    else:
        st.caption("Draw a polygon or a rectangle with the tools on the left of the map (the magnifier on the "
                   "right finds a place). The last drawn shape is used.")
        out = _folium(viz.draw_map(), key="draw_map", returned_objects=["all_drawings", "last_active_drawing"])
        feat = _last_drawing(out)
        if feat is not None and feat != ss.get("aoi_drawn"):
            ss["aoi_drawn"] = feat
            _set_aoi(feat, "drawn on the map")
    if ss.get("aoi_error"):
        st.error(f"Could not read the study area: {ss['aoi_error']}")
    aoi = ss.get("aoi")
    inp = ss.get(_k(None, "input"), DEFAULTS["input"])      # the select box below, as last chosen
    if aoi is not None:
        km2 = viz.area_km2(aoi)
        st.success(f"Study area ({ss.get('aoi_label')}): {km2:,.1f} km², about {max(1, round(km2 / CELL_KM2)):,} "
                   "cells of 2.56 x 2.56 km")
        try:
            est = rough_size(km2, 400, list(BENCHMARKS), inp)
        except Exception:            # the size warning is informative only
            est = None
        if est is not None and est["aoi_cells"] > CONFIRM_CELLS:
            s1 = "S1" in input_layers(inp) or INPUTS[inp]["seasonal"]
            st.warning(f"Large study area: about {est['aoi_cells']:,} cells, roughly {est['mb'] / 1000:,.0f} GB and "
                       f"{est['eecu_s'] / 3600:,.0f} EECU-hours of Earth Engine compute to download with input {inp}"
                       + (" (Sentinel-1 about 10 EECU-seconds per cell processed on this computer, the default, "
                          "150 or more on Earth Engine)" if s1 else "")
                       + ". Check the extent; the Run tab asks for a confirmation before the download starts.")
        if how == "Upload a file":
            _folium(viz.study_area_map(aoi), key="aoi_preview", height=420)
    st.subheader("New project")
    inp, found = _input_picker(None)
    st.caption("The input representation can be changed later in tab 2 (also after downloading).")
    folder = _new_folder()
    if folder is None:
        st.caption("To create the project, choose *Create new* in the sidebar and give a parent folder and a name.")
    ready = aoi is not None and folder is not None and not (folder / "project.yaml").exists() and found
    if st.button("Create project", type="primary", disabled=not ready):
        _create(folder, aoi, inp)


def _last_drawing(out):
    if not out:
        return None
    d = out.get("all_drawings") or []
    return d[-1] if d else out.get("last_active_drawing")


def _set_aoi(src, label):
    from canopy_height.tool import grid
    ss = st.session_state
    try:
        geom = grid.read_aoi(src)
    except Exception as e:           # any reader error goes to the page
        ss.pop("aoi", None)
        ss["aoi_error"] = f"{type(e).__name__}: {e}"
    else:
        ss.update(aoi=geom, aoi_label=label, aoi_error=None)


def pick_aoi_file(paths):
    """The file of a set of uploaded files that grid.read_aoi should read (a shapefile's .shp first, then .zip,
    .gpkg, .geojson, .json, .kml), or None."""
    found = {}
    for f in sorted(Path(x) for x in paths):
        found.setdefault(f.suffix.lower(), f)
    return next((found[e] for e in MAIN_EXT if e in found), None)


def _read_upload(files):
    ss = st.session_state
    sig = tuple((f.name, f.size) for f in files)
    if sig == ss.get("aoi_upload"):
        return
    ss["aoi_upload"] = sig
    tmp = Path(tempfile.mkdtemp(prefix="chm_aoi_"))
    try:
        for f in files:
            (tmp / Path(f.name).name).write_bytes(f.getvalue())
        main = pick_aoi_file(tmp.iterdir())
        if main is None:
            ss.pop("aoi", None)
            ss["aoi_error"] = "no .shp, .zip, .gpkg, .geojson, .json or .kml among the files (a shapefile needs its .shp)"
            return
        _set_aoi(main, main.name)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _create(folder, aoi, inp=None):
    ss = st.session_state
    settings = dict(name=ss.get("new_name", "").strip() or None,
                    gee_project=ss.get(_k(None, "gee_project"), "").strip() or None)
    if inp is not None and inp != DEFAULTS["input"]:
        settings["input"] = inp
    if not checkpoint(inp or DEFAULTS["input"])[1]:
        st.error(f"Could not create the project: the UNet-ALS checkpoint of input {inp} is missing.")
        return
    try:
        p = Project.create(folder, aoi, **settings)
    except Exception as e:           # folder / permission / geometry errors go to the page
        st.error(f"Could not create the project: {type(e).__name__}: {e}")
        return
    for k in ("aoi", "aoi_label", "aoi_upload", "aoi_drawn", "aoi_error"):
        ss.pop(k, None)
    ss["project_dir"] = str(p.root)
    ss["_goto_open"] = str(p.root)
    st.rerun()


# ---------------------------------------------------------------------------------------------- 2 settings
def _check_ee(project):
    try:
        import ee
        ee.Initialize(project=project)
    except Exception as e:           # credentials / project / network errors are shown as they are
        return False, f"Earth Engine could not be initialised for project '{project}': {e}"
    return True, f"Earth Engine initialised for project '{project}'."


def _authenticate(timeout=AUTH_TIMEOUT, cmd=None):
    """ee.Authenticate(auth_mode='localhost', force=True) in a child process: it opens the Google sign-in in a
    browser tab and waits for it; after `timeout` s the child is ended (which frees its local port for the next
    try). `cmd` overrides the command (tests) -> (ok, message)."""
    cmd = cmd or [sys.executable, "-c", "import ee; ee.Authenticate(auth_mode='localhost', force=True)"]
    kw = dict(creationflags=subprocess.CREATE_NO_WINDOW) if os.name == "nt" else {}
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL,
                           env=_child_env(), **kw)
    except subprocess.TimeoutExpired:
        return False, (f"No sign-in within {timeout / 60:g} minutes: authentication given up. Press *Authenticate* "
                       "again, or run `earthengine authenticate` in a terminal.")
    except OSError as e:
        return False, f"Authentication failed: {e}"
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip().splitlines()
        return False, "Authentication failed: " + (err[-1] if err else f"exit code {r.returncode}")
    return True, "New Earth Engine credentials are stored on this computer; *Check Earth Engine* tests them."


def _tab_settings(p):
    ss = st.session_state
    st.subheader("Google Earth Engine")
    if p is None:
        key = _k(None, "gee_project")
        if key not in ss:
            ss[key] = ""
        gp = st.text_input("Google Cloud project registered for Earth Engine", key=key,
                           help="The project id (not its name), as in https://console.cloud.google.com").strip()
    else:
        gp = (p["gee_project"] or "").strip()
        st.markdown(f"Google Cloud project of this project: **{gp or '(not set)'}** (set it in the settings below)")
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Check Earth Engine", disabled=not gp):
        with st.spinner("Initialising Earth Engine ..."):
            ss["gee_check"] = _check_ee(gp)
    if c2.button("Authenticate"):
        with st.spinner(f"Waiting for the Google sign-in in the new browser tab (up to {AUTH_TIMEOUT // 60} "
                        "minutes) ..."):
            ss["gee_check"] = _authenticate()
    if ss.get("gee_check"):
        ok, msg = ss["gee_check"]
        (st.success if ok else st.error)(msg)
    st.caption(AUTH_NOTE)
    if p is None:
        st.info("Create or open a project to change its settings.")
        return
    _settings_form(p)
    _replan(p)


def invalid_settings(cfg):
    """project.validate's message for settings the tool cannot run with, else None."""
    try:
        validate(cfg)
    except Exception as e:           # ValueError, or pyproj's CRSError for an unknown EPSG code
        return str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}"
    return None


def _options(known, current):
    """Choices of a select box: the known values and the project's current value (shown, never rewritten)."""
    return list(known) + ([current] if current not in known else [])


def has_results(p):
    """Something was built from the downloads: rasters in raw/, a training stack or a trained model."""
    return (any(p.path("raw").glob("*.tif")) if p.path("raw").exists() else False) or bool(p.stack_files("x")) \
        or any((p.model_dir(m) / "result.json").exists() for m in MODELS)


def _input_picker(p):
    """'Input representation' select box (new project: p None) with the table of what each representation
    downloads and trains, the cost note and the check of its UNet-ALS checkpoint -> (choice, checkpoint found).
    The choice is saved by the caller (Save settings / Create project)."""
    ss = st.session_state
    key = _k(p, "input")
    current = str(p["input"]) if p is not None else DEFAULTS["input"]
    opts = _options(INPUTS, current)
    if ss.get(key + ":stored", current) != current or ss.get(key) not in opts:
        ss[key] = current            # first show, or project.yaml changed meanwhile (saved here or elsewhere)
    ss[key + ":stored"] = current
    inp = st.selectbox("Input representation", opts, key=key, format_func=_input_label,
                       help="The satellite layers the models read. Default: AE.")
    st.dataframe(input_table(), hide_index=True)
    st.caption(INPUT_NOTE)
    base = p["base_model"] if p is not None else None
    ck, found = checkpoint(inp, base)
    found = found and inp in INPUTS
    if base:
        st.caption(f"project.yaml sets base_model = {base}: KG-UNet1/2 start from it whatever the representation, "
                   f"so it must be a UNet-ALS checkpoint of input {inp}.")
    if inp not in INPUTS:
        st.warning(f"Unknown input representation {inp!r} in project.yaml: choose one of {', '.join(INPUTS)}.")
    elif not found:
        name = f"{base}" if base else INPUTS[inp]["checkpoint"]
        st.warning(f"The UNet-ALS checkpoint of input {inp} ({name}) is not found"
                   + (f" (looked for {ck})" if ck is not None and not base else "")
                   + f": KG-UNet1/2 start from it. Download it from {WEIGHTS_URL}/ into weights/source/ of the "
                   "repository (or set $CHM_WEIGHTS to a folder with source/), or choose another representation. "
                   + ("The settings cannot be saved with this representation." if p is not None else
                      "A project cannot be created with this representation."))
    if p is not None and inp != current:
        if has_results(p):
            new = [l for l in input_layers(inp) if current not in INPUTS or l not in input_layers(current)]
            st.info(f"Changing the input representation from {current} to {inp} after downloading is allowed. The "
                    "next run (*Run all* in tab 3) downloads the layers of the new representation that are not "
                    "downloaded yet" + (f" ({', '.join(new)})" if new else "") + " and keeps the ones already "
                    f"downloaded, rebuilds the training stack and the study-area mosaics, and trains the models "
                    f"again with {inp}; the maps and the report follow.")
        st.caption(f"Not saved yet: *Save settings* below stores input {inp}.")
    return inp, found


def _settings_form(p):
    ss = st.session_state
    msg = ss.pop(_k(p, "settings_msg"), None)
    if msg:
        st.success(msg)
    raw_done = p.path("raw").exists() and any(p.path("raw").glob("*.tif"))
    bad = invalid_settings(p.cfg)
    if bad:
        st.error(f"project.yaml holds a setting the tool cannot run with: {bad}"
                 + (" The year cannot be changed once layers are downloaded: start a new project."
                    if raw_done and p["year"] not in YEARS else " Correct it below and save."))
    als = p["als"]
    als = [als] if isinstance(als, str) else list(als or [])
    years = _options(YEARS, p["year"])
    als_opts = _options(ALS_RES, float(p["als_resolution"]))
    scales = _options(ALS_SCALE, float(p["als_scale"]))
    devices = _options(DEVICES, p["device"] or "auto")
    tc = p["train_cells"] if isinstance(p["train_cells"], int) else 400
    ring_km, wmax = float(p["ring_max_km"]), float(p["water_max"])
    st.subheader("Project settings")
    inp, found = _input_picker(p)
    with st.form(_k(p, "settings")):
        name = st.text_input("Project name", value=p["name"] or "")
        gee_project = st.text_input("Google Cloud project registered for Earth Engine", value=p["gee_project"] or "",
                                    help="The project id (not its name), as in https://console.cloud.google.com")
        s1_opts = _options(S1_METHOD_LABELS, p["s1_method"])
        s1_method = st.radio("Sentinel-1 processing", s1_opts, index=s1_opts.index(p["s1_method"]),
                             format_func=lambda m: S1_METHOD_LABELS.get(m, f"{m} (not supported)"))
        st.caption(S1_METHOD_NOTE)
        year = st.selectbox("Year of the annual inputs", years, index=years.index(p["year"]), disabled=raw_done,
                            format_func=lambda y: str(y) if y in YEARS else f"{y} (not supported)")
        st.caption(YEAR_NOTE + (" Fixed: layers are already downloaded; start a new project for another year."
                                if raw_done else ""))
        train_cells = st.number_input("Training cells", min_value=min(5, tc), max_value=max(5000, tc), step=50,
                                      value=tc)
        c1, c2 = st.columns(2)
        ring_max_km = c1.number_input("Largest distance of training cells from the study area (km)",
                                      min_value=min(0.0, ring_km), max_value=max(200.0, ring_km), step=5.0,
                                      value=ring_km)
        water_max = c2.number_input("Drop surrounding cells with a water share of at least",
                                    min_value=min(0.0, wmax), max_value=max(1.0, wmax), step=0.01, value=wmax)
        st.caption(TRAIN_CELLS_NOTE + (" The cells are already planned: a new training region takes effect after "
                                       "*Plan the cells again* below." if p.cells_file.exists() else ""))
        models = st.multiselect("Models", MODELS, default=[m for m in p["models"] if m in MODELS],
                                format_func=lambda m: MODEL_NAMES[m])
        benchmarks = st.multiselect("Benchmarks", list(BENCHMARKS),
                                    default=[b for b in p["benchmarks"] if b in BENCHMARKS],
                                    format_func=lambda b: f"{b}: {BENCHMARK_REFS[b]} ({BENCHMARK_RES[b]})")
        st.markdown("**Optional: ALS canopy height for an independent accuracy check**")
        als_text = st.text_area("ALS canopy-height rasters (one path per line)", value="\n".join(als))
        c1, c2 = st.columns(2)
        als_res = c1.selectbox("Resolution of the ALS rasters", als_opts,
                               index=als_opts.index(float(p["als_resolution"])),
                               format_func=lambda r: ALS_RES.get(r, f"{r:g} m"))
        als_scale = c2.selectbox("Units of the ALS values", scales, index=scales.index(float(p["als_scale"])),
                                 format_func=lambda s: ALS_SCALE.get(s, f"factor {s:g} to metres"))
        st.caption("The tool works in metres: canopy-height rasters stored in centimetres (as some national lidar "
                   "products) need the factor 0.01.")
        device = st.selectbox("Device for the UNets", devices, index=devices.index(p["device"] or "auto"),
                              format_func=lambda d: "auto (GPU if available)" if d == "auto" else d)
        saved = st.form_submit_button("Save settings", type="primary", disabled=not found)
    if not found:
        st.caption("*Save settings* is disabled until an input representation whose UNet-ALS checkpoint is found "
                   "is chosen (see above).")
    if not saved:
        return
    paths = [s.strip().strip('"') for s in als_text.splitlines() if s.strip()]
    new = dict(name=name.strip() or None, gee_project=gee_project.strip() or None, train_cells=int(train_cells),
               ring_max_km=float(ring_max_km), water_max=float(water_max),
               models=[m for m in MODELS if m in models], benchmarks=[b for b in BENCHMARKS if b in benchmarks],
               als=paths or None, als_resolution=float(als_res), als_scale=float(als_scale),
               device=None if device == "auto" else device, input=inp, s1_method=s1_method)
    if not raw_done:
        new["year"] = year
    errors = []
    ck, ok = checkpoint(inp, p["base_model"])
    if not ok:
        errors.append(f"The UNet-ALS checkpoint of input {inp} is missing "
                      f"({ck or INPUTS.get(inp, {}).get('checkpoint')}): put it in weights/source/ or choose another "
                      "input representation.")
    if "kg-unet2" in models and "unet-sls" not in models and not p.model_file("unet-sls").exists():
        errors.append("KG-UNet2 needs UNet-SLS of this project as its GEDI teacher: select UNet-SLS too.")
    missing = [s for s in paths if not Path(s).exists()]
    if missing:
        errors.append("ALS rasters not found: " + ", ".join(missing))
    bad = invalid_settings({**p.cfg, **new})
    if bad:
        errors.append(bad)
    if errors:
        for e in errors:
            st.error(e)
        st.error("Settings not saved.")
        return
    region = ("train_cells", "ring_max_km", "water_max")
    if p.cells_file.exists() and any(new[k] != p.cfg[k] for k in region):
        ss[_k(p, "replan")] = True
    old = p["input"]
    p.cfg.update(new)
    p.save()
    msg = f"Saved to {p.path('project.yaml')}"
    if inp != old:
        msg += (f". Input representation {old} -> {inp}" + (": *Run all* in tab 3 downloads the missing layers, "
                "rebuilds the training stack and trains the models again." if has_results(p) else "."))
    ss[_k(p, "settings_msg")] = msg
    st.rerun()


def plan_imported(cells):
    """cells.csv comes from import-cells (cells with their own raster folders)."""
    return cells is not None and "src_dir" in cells and cells["src_dir"].notna().any()


def delete_plan(p):
    """Remove plan/grid.json and plan/cells.csv, so the Plan stage lays out the cells again; downloads, stack,
    models and maps stay, and so does plan/landcover.csv (a cache of WorldCover look-ups by cell name, so no water
    share is asked from Earth Engine twice) -> names removed."""
    removed = []
    for f in (p.cells_file, p.grid_file):
        if f.exists():
            f.unlink()
            removed.append(f"plan/{f.name}")
    p.set_status("plan", "pending", "plan deleted from the app (plan again)")
    return removed


def _replan(p):
    if not p.cells_file.exists():
        return
    ss = st.session_state
    changed = ss.get(_k(p, "replan"), False)
    if changed:
        st.warning("The training-region settings changed after the cells were planned: plan/cells.csv still holds "
                   "the old training region. Plan the cells again for the new settings to take effect.")
    with st.expander("Plan the cells again", expanded=changed):
        cells, _ = _cells(p)
        if plan_imported(cells):
            st.caption("The cells of this project were imported (import-cells): import them again from the command "
                       "line instead.")
            return
        st.markdown("Deletes plan/grid.json and plan/cells.csv; nothing else. Downloads (raw/) are kept, so the "
                    "cells of the new plan that are already downloaded are not fetched again, and plan/landcover.csv "
                    "keeps the WorldCover shares already looked up. Then run *Plan* (or *Run all*) in tab 3: the "
                    "Stack stage rebuilds the training stack from the new plan, and models trained on the old "
                    "stack are trained again.")
        run = active_run(p.root)
        if run is not None:
            st.caption(f"A run is going (PID {run['pid']}): stop it or wait for it first.")
        ok = st.checkbox("Delete the current plan", key=_k(p, "replan_ok"))
        if st.button("Re-plan", disabled=not ok or run is not None, key=_k(p, "replan_btn")):
            removed = delete_plan(p)
            ss.pop(_k(p, "replan"), None)
            ss[_k(p, "settings_msg")] = (f"Removed {', '.join(removed) or 'nothing'}; run *Plan* or *Run all* in "
                                         "tab 3 to plan the cells again.")
            ss.pop(f"_live:{p.root}", None)          # the live view takes the new plan state as its baseline
            st.rerun()


# ---------------------------------------------------------------------------------------------- 3 run
def _download_guard(p):
    """Download size above the run buttons -> False while a large download is not confirmed (checkbox)."""
    est, err = download_size(p)
    if est is None:
        st.caption(f"Download size not available ({err})")
        return True
    st.markdown(size_text(est, p["input"]))
    if not needs_confirmation(est):
        return True
    ok = st.checkbox(f"I have checked the size: download about {est['eecu_s'] / 3600:,.0f} EECU-hours and "
                     f"{_disk(est['mb'])} for {est['cells']:,} cells",
                     key=_k(p, f"confirm_download:{est['cells']}"))
    if not ok:
        st.caption(f"*Run all* and *Download* need this confirmation (more than {CONFIRM_CELLS} study-area cells or "
                   f"{CONFIRM_EECU_H} EECU-hours). The download uses the monthly Earth Engine quota of the Google "
                   "Cloud project; a project over its quota is slowed down.")
    return ok


def _tab_run(p):
    if p is None:
        st.info("Open or create a project first.")
        return
    run = active_run(p.root)
    busy = run is not None
    if not p["gee_project"]:
        st.warning("No Google Cloud project is set (tab 2): the Plan and Download stages need Earth Engine.")
    bad = invalid_settings(p.cfg)
    if bad:
        st.error(f"The settings cannot be run (tab 2 or project.yaml): {bad}")
    elif any(m in KG_MODELS for m in p["models"] or []) and not checkpoint(p["input"], p["base_model"])[1]:
        st.warning(f"The UNet-ALS checkpoint of input {p['input']} is missing: KG-UNet1/2 cannot be trained (tab 2).")
    msg = st.session_state.pop("run_msg", None)
    if msg:
        st.success(msg)
    allowed = _download_guard(p)
    cols = st.columns(len(STAGES) + 2)
    clicked = False
    if cols[0].button("Run all", type="primary", disabled=busy or not allowed, key=_k(p, "run_all")):
        clicked, stage = True, None
    for c, s in zip(cols[1:], STAGES):
        if c.button(s.capitalize(), disabled=busy or (s == "download" and not allowed), key=_k(p, f"run_{s}")):
            clicked, stage = True, s
    if cols[-1].button("Stop", disabled=not busy, key=_k(p, "stop")):
        with st.spinner("Stopping ..."):
            stopped = stop_run(p.root)
        _rerun_with(p, f"{time.strftime('%H:%M:%S')} stopped the run (PID {stopped})." if stopped else "No run to stop.")
    if clicked:
        _start(p, stage)
    st.caption("*Run all* runs Plan, Download, Stack, Train, Predict and Report in turn; a stage button runs that "
               "stage only. A run goes on in the background when this page or the terminal of `chm-tool app` is "
               "closed. Every stage is resumable: finished parts are skipped, so a stopped run continues where it "
               "stopped.")
    _live(str(p.root))


@st.fragment(run_every=3)
def _live(root):
    """Run state, status table and log tail, refreshed every 3 s."""
    ss = st.session_state
    p = Project(root)
    run = active_run(root)
    status = p.status()
    sig = (run["pid"] if run else None, tuple((s, r.get("state")) for s, r in status.items()))
    key = f"_live:{Path(root).resolve()}"
    if key in ss and ss[key] != sig:
        ss[key] = sig
        st.rerun()                   # run started / ended or a stage changed: refresh buttons and sidebar
    ss[key] = sig
    running = [s for s, r in status.items() if r.get("state") == "running"]
    if run is not None and run["ours"]:
        st.info(f"Run in progress (PID {run['pid']}).")
    elif run is not None:
        st.info(f"Run in progress outside this app (PID {run['pid']}, {run['what']}, since {run['started']}): the "
                "run buttons stay disabled until it ends; *Stop* ends it.")
    elif running:
        st.warning(f"status.json marks {', '.join(running)} as running, but no process of this project is alive: "
                   "the run was interrupted. Start it again to resume.")
    rows = [dict(stage=s, state=r.get("state", ""), message=r.get("message", ""), started=r.get("started", ""),
                 finished=r.get("finished", "")) for s, r in status.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.markdown(f"**Log** (logs/run.log, last {TAIL} lines)")
    st.code(tail(p.log_file) or "(empty)", language="text")
    with st.expander("Process output (logs/app_process.log)"):
        st.code(tail(process_log(root)) or "(empty)", language="text")
    st.caption(f"Updated {time.strftime('%H:%M:%S')}")


# ---------------------------------------------------------------------------------------------- 4 results
@st.cache_data(show_spinner=False, max_entries=32)
def _warped(path, mtime, mask, mask_mtime, how, max_px):
    return viz.warp(path, max_px=max_px, how=how, mask_path=mask or None)


@st.cache_data(show_spinner=False, max_entries=64)
def _overlay(path, mtime, mask, mask_mtime, how, max_px, vmax):
    a, bounds = _warped(path, mtime, mask, mask_mtime, how, max_px)
    return viz.png_data_url(viz.colorize(a, vmax)), bounds


def result_layers(p):
    """[(name, GeoTIFF, block reduction)] of the maps of the known models (paper order; temporary files such as
    maps/<Model>.partial.tif are never listed), benchmarks, ALS and GEDI that exist in the project."""
    out = [(MODEL_NAMES[m], p.map_file(m), "mean") for m in MODELS if p.map_file(m).exists()]
    out += [(f"{b} ({BENCHMARK_REFS[b]})", p.mosaic(b), "mean") for b in BENCHMARKS if p.mosaic(b).exists()]
    if p.mosaic("ALS").exists():
        out.append(("ALS reference", p.mosaic("ALS"), "mean"))
    if p.mosaic("GEDI").exists():
        out.append(("GEDI labels (rh95)", p.mosaic("GEDI"), "max"))
    return out


def model_inputs(p):
    """{model: input representation} of the trained models (models/<model>/result.json: 'input' of RF-SLS,
    model / recipe 'input' of the UNets)."""
    out = {}
    for m in MODELS:
        try:
            r = json.loads((p.model_dir(m) / "result.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(r, dict):
            continue
        sub = [r] + [r[k] for k in ("model", "recipe") if isinstance(r.get(k), dict)]
        inp = next((d["input"] for d in sub if d.get("input")), None)
        if inp:
            out[m] = str(inp)
    return out


def results_input(p):
    """(input representation of the results: that of the trained models, else the project's; warning when a
    trained model used another representation than project.yaml, else None)."""
    cur = p["input"]
    used = model_inputs(p)
    other = {m: i for m, i in used.items() if i != cur}
    kinds = sorted(set(used.values()))
    shown = kinds[0] if len(kinds) == 1 else (" / ".join(kinds) if kinds else str(cur))
    if not other:
        return shown, None
    names = ", ".join(f"{MODEL_NAMES[m]} ({i})" for m, i in other.items())
    return shown, (f"Trained with another input representation than the project's {cur}: {names}. The maps and the "
                   f"report below are theirs; *Run all* (tab 3) downloads the missing layers, rebuilds the training "
                   f"stack and trains the models again with {cur}.")


def _tab_results(p):
    if p is None:
        st.info("Open or create a project first.")
        return
    inp, warn = results_input(p)
    if warn:
        st.warning(warn)
    layers = result_layers(p)
    if layers:
        st.subheader(f"Canopy-height maps: input {inp}")
        st.caption(" / ".join(_input_label(i) for i in inp.split(" / ")))
        _results_map(p, layers, inp)
    else:
        st.info("No maps yet: the benchmark mosaics appear after the Stack stage, the model maps after Predict.")
    _report(p, inp)
    if layers:
        _downloads(p, layers, inp)


def _results_map(p, layers, inp=None):
    c1, c2 = st.columns([4, 1])
    with c2:
        clip = st.checkbox("Clip to the study area", value=True, key=_k(p, "clip"))
    mask = p.mosaic("aoi_mask")
    mask = str(mask) if clip and mask.exists() else ""
    mt = _mtime(mask) if mask else 0.0
    arrays, ok = [], []
    with st.spinner("Rendering the maps ..."):
        for name, f, how in layers:
            try:
                arrays.append(_warped(str(f), _mtime(f), mask, mt, how, MAX_PX)[0])
                ok.append((name, f, how))
            except Exception as e:   # one unreadable raster must not hide the others
                st.warning(f"{f.name}: {type(e).__name__}: {e}")
    if not ok:
        return
    auto = float(viz.common_vmax(arrays))
    top = max(MAX_HEIGHT, auto)
    with c2:
        vmax = st.number_input("Colour scale maximum (m)", min_value=1.0, max_value=top, step=5.0, value=auto,
                               key=_k(p, f"vmax:{top:g}"))
        st.caption("All layers share the 0..max scale (viridis); no data is transparent. Switch layers on and off "
                   "with the layer control of the map.")
    high = [name for (name, _, _), a in zip(ok, arrays) if viz.common_vmax([a]) > MAX_HEIGHT]
    if high:
        st.warning(f"{', '.join(high)}: values above {MAX_HEIGHT:g} m, not plausible canopy heights. Check the "
                   "units of the ALS rasters (tab 2, *Units of the ALS values*; 0.01 for centimetres) and run the "
                   "Stack and Report stages again.")
    overlays = [(name, *_overlay(str(f), _mtime(f), mask, mt, how, MAX_PX, float(vmax))) for name, f, how in ok]
    aoi = viz.aoi_geometry(p.aoi_file) if p.aoi_file.exists() else None
    caption = f"Canopy height (m), input {inp}" if inp else "Canopy height (m)"
    with c1:
        _folium(viz.results_map(overlays, vmax, aoi, caption=caption), key=_k(p, "results_map"), height=600)


def read_summary(rep):
    """(report/summary.json as a dict or None, error message or None)."""
    f = Path(rep) / "summary.json"
    if not f.exists():
        return None, None
    try:
        s = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return None, f"summary.json: {e}"
    return (s, None) if isinstance(s, dict) else (None, "summary.json does not hold a JSON object")


def quicklook_files(rep, summary=None):
    """Quick-look PNGs of the report: those summary.json lists under 'quicklooks' (in its order, files of report/
    that exist), else every PNG of report/."""
    rep = Path(rep)
    listed = summary.get("quicklooks") if isinstance(summary, dict) else None
    if isinstance(listed, list):
        files = [rep / Path(str(x)).name for x in listed]
        return [f for i, f in enumerate(files) if f.suffix.lower() == ".png" and f.exists() and f not in files[:i]]
    return sorted(rep.glob("*.png")) if rep.exists() else []


def report_notes(summary):
    """(stale models [(model, why)], notes [str]) of summary.json."""
    if not isinstance(summary, dict):
        return [], []
    stale = summary.get("stale_models") or {}
    stale = list(stale.items()) if isinstance(stale, dict) else [(str(m), "") for m in stale]
    notes = summary.get("notes") or []
    return stale, [str(n) for n in (notes if isinstance(notes, list) else [notes])]


def _report(p, inp=None):
    rep = p.path("report")
    summary, err = read_summary(rep)
    shown = (summary or {}).get("input") or inp or p["input"]
    st.subheader(f"Report: input {shown}")
    metrics = rep / "metrics.csv"
    csvs = ([metrics] if metrics.exists() else []) + sorted(set(rep.glob("*.csv")) - {metrics}) if rep.exists() else []
    pngs = quicklook_files(rep, summary)
    if not csvs and not pngs and summary is None and err is None:
        st.info("The report appears after the Report stage.")
        return
    if err:
        st.warning(err)
    stale, notes = report_notes(summary)
    if stale:
        st.warning("Models that do not match the current training stack (not scored; train them again with the "
                   "Train stage or *Run all*): " + "; ".join(f"{m}: {why}" if why else m for m, why in stale))
    for f in csvs:
        st.markdown(f"**{f.name}**")
        try:
            st.dataframe(pd.read_csv(f), hide_index=True)
        except (OSError, ValueError, pd.errors.ParserError) as e:
            st.warning(f"{f.name}: {e}")
    st.caption(GEDI_NOTE)
    if notes:
        st.info("**Notes of the report**\n\n" + "\n".join(f"- {n}" for n in notes))
    if summary is not None:
        with st.expander("summary.json"):
            st.json(summary)
    if pngs:
        st.markdown("**Quick looks**")
        cols = st.columns(3)
        for i, f in enumerate(pngs):
            cols[i % 3].image(str(f), caption=f.stem)


def download_items(layers, max_mb=MAX_DOWNLOAD_MB):
    """[(name, path, MB, mtime, offered in the browser)] of the result files that exist."""
    out = []
    for name, f, _ in layers:
        try:
            s = Path(f).stat()
        except OSError:                  # replaced or removed by a running stage meanwhile
            continue
        mb = s.st_size / 1e6
        out.append((name, Path(f), mb, s.st_mtime, mb <= max_mb))
    return out


def _downloads(p, layers, inp=None):
    st.subheader(f"Downloads (GeoTIFF): input {inp or p['input']}")
    tag = p.root.name
    for name, f, mb, mtime, offered in download_items(layers):
        if not offered:
            st.caption(f"{name}: {mb:,.0f} MB, too large for the browser; the file is {f}")
            continue
        st.download_button(f"{name} ({f.name}, {mb:,.1f} MB)", data=functools.partial(viz.file_bytes, str(f), mtime),
                           file_name=f"{tag}_{f.name}", mime="image/tiff", on_click="ignore",
                           key=_k(p, f"dl_{f.parent.name}_{f.name}"))


# ---------------------------------------------------------------------------------------------- page
def main():
    st.set_page_config(page_title="Canopy height tool", layout="wide")
    _init_state()
    p = _project()
    _sidebar(p)
    st.title("Canopy height tool")
    tabs = st.tabs(["1 Study area", "2 Settings", "3 Run", "4 Results", "5 Help"])
    with tabs[0]:
        _tab_study_area(p)
    with tabs[1]:
        _tab_settings(p)
    with tabs[2]:
        _tab_run(p)
    with tabs[3]:
        _tab_results(p)
    with tabs[4]:
        st.markdown(HELP)


if __name__ == "__main__":
    main()
