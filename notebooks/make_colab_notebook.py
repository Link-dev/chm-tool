"""Writes chm_tool_colab.ipynb next to this file (edit the cells here, not in the notebook): python make_colab_notebook.py"""
import json
from pathlib import Path

CELLS = []


def md(text):
    CELLS.append(dict(cell_type="markdown", metadata={}, source=text.strip("\n")))


def code(text):
    meta = {"cellView": "form"} if "#@param" in text or "#@title" in text else {}
    CELLS.append(dict(cell_type="code", metadata=meta, execution_count=None, outputs=[], source=text.strip("\n")))


md(r"""
# Canopy height for your study area (chm-tool)

10 m canopy-height maps for a study area you choose, with the models of *Leveraging cross-sensor LiDAR
observations and Earth embeddings for canopy height prediction*. The notebook downloads the inputs and GEDI labels
from Google Earth Engine, trains four local models (RF-SLS, UNet-SLS, KG-UNet1, KG-UNet2) and shows their maps next
to GMTCH, GFCH and HRCH. Documentation: [github.com/Link-dev/chm-tool](https://github.com/Link-dev/chm-tool).

**You need**
- a Google Cloud project registered for Earth Engine
  ([free for non-commercial use](https://code.earthengine.google.com/register));
- a T4 GPU runtime for the UNets (*Runtime > Change runtime type*). Without a GPU, cell 7 stops after the steps
  that need none (download, training data, RF-SLS): switch to a GPU runtime and run the notebook again;
- space on Google Drive, where the project is kept: about 20 GB for the default 400 cells with input AE (cell 5
  prints the estimate for your area).

Set everything in cell 1 and run the cells in order. If the session ends, run them again: finished parts are kept.

| Input | Layers |
|---|---|
| **AE** (default) | Earth embedding + annual Sentinel-1/2 + DEM |
| **A** | annual Sentinel-1/2 + DEM |
| **E** | Earth embedding (fastest download) |
| **T** | four-season Sentinel-1/2 + DEM |
| **TE** | Earth embedding + four-season Sentinel-1/2 + DEM |
""")

code(r"""
#@title 1. Settings
PROJECT_NAME = "my_area"            #@param {type:"string"}
GEE_PROJECT = ""                    #@param {type:"string"}
INPUT = "AE: Earth embedding + annual Sentinel-1/2 + DEM"  #@param ["AE: Earth embedding + annual Sentinel-1/2 + DEM", "A: annual Sentinel-1/2 + DEM", "E: Earth embedding (fastest download)", "T: four-season Sentinel-1/2 + DEM", "TE: Earth embedding + four-season Sentinel-1/2 + DEM"]
YEAR = 2020                         #@param {type:"integer"}
TRAIN_CELLS = 400                   #@param {type:"integer"}
#@markdown Study area: a longitude / latitude box, a shape drawn on a map (cell 4), a file on Google Drive, or an
#@markdown uploaded file (zipped shapefile, GeoJSON, GeoPackage, KML).
AOI_MODE = "box"                    #@param ["box", "draw", "drive file", "upload"]
WEST = 173.44                       #@param {type:"number"}
SOUTH = -41.29                      #@param {type:"number"}
EAST = 173.53                       #@param {type:"number"}
NORTH = -41.22                      #@param {type:"number"}
AOI_FILE = "/content/drive/MyDrive/chm-tool/study_area.geojson"                #@param {type:"string"}
#@markdown **Advanced** (the defaults are fine): parallel Earth Engine requests, Sentinel-1 processing ("local" uses
#@markdown far less Earth Engine quota), the last stage to run, and where the project, code and weights are.
WORKERS = 3                         #@param {type:"integer"}
S1_METHOD = "local"                 #@param ["local", "gee"]
RUN_TO = "report"                   #@param ["plan", "download", "stack", "train", "predict", "report"]
DRIVE_DIR = "/content/drive/MyDrive/chm-tool"                                  #@param {type:"string"}
CODE_SOURCE = "git+https://github.com/Link-dev/chm-tool@main"                  #@param {type:"string"}
WEIGHTS_SOURCE = "https://github.com/Link-dev/chm-tool/releases/download/v1.0.0"  #@param {type:"string"}

INPUT_LABEL = INPUT
INPUT = INPUT.split(":")[0].strip()          # the code before the colon: AE, A, E, T or TE
print(f"project {PROJECT_NAME}: input {INPUT_LABEL}; year {YEAR}; {TRAIN_CELLS} training cells")
""")

code(r"""
#@title 2. Google Drive and installation
import glob, os, subprocess, sys, zipfile
from google.colab import drive
drive.mount("/content/drive")
os.makedirs(DRIVE_DIR, exist_ok=True)

def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)

src = CODE_SOURCE.strip()
if src.endswith(".zip"):
    with zipfile.ZipFile(src) as z:
        z.extractall("/content/chm_code")
    tomls = sorted(glob.glob("/content/chm_code/**/pyproject.toml", recursive=True), key=len)
    if not tomls:
        raise FileNotFoundError(f"no pyproject.toml in {src}")
    pip(os.path.dirname(tomls[0]) + "[colab]")
elif os.path.isdir(src):
    pip(src + "[colab]")
else:
    pip(f"canopy-height[colab] @ {src}")
import importlib
importlib.invalidate_caches()
import canopy_height
from canopy_height.tool import colab
print("chm-tool", canopy_height.__version__, "installed")
""")

code(r"""
#@title 3. Session check and Earth Engine sign-in
env = colab.environment()
print(f"GPU: {env['gpu'] or 'none'}; RAM: {env['ram_gb']} GB")
SESSION_SETTINGS, warnings = colab.colab_settings(env)
for w in warnings:
    print("WARNING:", w)

import ee
from canopy_height.tool.gee import io as gee_io
if not GEE_PROJECT:
    raise ValueError("enter your Earth Engine Cloud project in cell 1 (GEE_PROJECT)")
ee.Authenticate()
gee_io.ee_check(GEE_PROJECT)
print(f"Earth Engine project {GEE_PROJECT}: ok")
""")

code(r"""
#@title 4. Study area
from pathlib import Path
ROOT = Path(DRIVE_DIR) / PROJECT_NAME
drawer = None
if (ROOT / "project.yaml").exists():
    print(f"{ROOT} already holds a project: it is opened in cell 5 (the study area below is not used)")
elif AOI_MODE == "box":
    AOI = colab.aoi_from_bbox(WEST, SOUTH, EAST, NORTH)
    print("study area: box", AOI.bounds)
elif AOI_MODE == "draw":
    drawer = colab.AoiDrawer(center=((SOUTH + NORTH) / 2, (WEST + EAST) / 2), zoom=11)
    print("Draw a rectangle or polygon (tools on the left), then run cell 5.")
    display(drawer)
elif AOI_MODE == "drive file":
    AOI = AOI_FILE
    if not Path(AOI).exists():
        raise FileNotFoundError(f"{AOI} not found (AOI_FILE in cell 1)")
    print("study area file:", AOI)
else:
    from google.colab import files
    up = files.upload()
    name = next(iter(up))
    AOI = str(Path("/content") / name)
    Path(AOI).write_bytes(up[name])
    print("study area file:", AOI)
""")

code(r"""
#@title 5. Create or open the project, plan the training region
from canopy_height.tool.project import Project
if (ROOT / "project.yaml").exists():
    p = Project(ROOT)
else:
    if drawer is not None:
        if drawer.geometry is None:
            raise ValueError("no shape drawn yet: draw the study area on the map of cell 4")
        AOI = drawer.geometry
    p = Project.create(ROOT, AOI, name=PROJECT_NAME, input=INPUT, year=YEAR, gee_project=GEE_PROJECT,
                       train_cells=TRAIN_CELLS, workers=WORKERS, s1_method=S1_METHOD, **SESSION_SETTINGS)
# settings of cell 1 that may be changed later (the tool rebuilds what they affect)
for k, v in dict(input=INPUT, year=YEAR, gee_project=GEE_PROJECT, workers=WORKERS, s1_method=S1_METHOD,
                 **SESSION_SETTINGS).items():
    p.cfg[k] = v
p.save()
colab.run(p, to_stage="plan")
est = colab.storage_estimate(p)
print(f"estimated project size on Drive: {est:.1f} GB")
display(colab.plan_map(p))
""")

code(r"""
#@title 6. UNet-ALS weights (start of KG-UNet1/2)
colab.fetch_weights(p["input"], WEIGHTS_SOURCE)
""")

code(r"""
#@title 7. Run: download, stack, train, predict, report up to RUN_TO (run again to resume)
colab.run(p, to_stage=RUN_TO, workers=WORKERS)
colab.status_table(p)
""")

code(r"""
#@title 8. Results
CLIP_TO_STUDY_AREA = True   #@param {type:"boolean"}
#@markdown Colour scale in metres; leave empty for automatic (2nd-98th percentile of the maps in the study area).
VMIN = ""                   #@param {type:"string"}
VMAX = ""                   #@param {type:"string"}
import matplotlib.pyplot as plt
lo = float(VMIN) if VMIN.strip() else None
hi = float(VMAX) if VMAX.strip() else None
colab.compare_figure(p, clip=CLIP_TO_STUDY_AREA, vmin=lo, vmax=hi)
plt.show()
display(colab.results_map(p, clip=CLIP_TO_STUDY_AREA, vmin=lo, vmax=hi))
mt = colab.metrics_table(p)
if mt is not None:
    display(mt)
print("maps (GeoTIFF, m):", p.path("maps"))
print("report:", p.path("report"))
print("disk use (GB):", colab.storage(p))
""")


md(r"""
**Notes**
- The maps are in `maps/` of the project folder on Drive (GeoTIFF, metres).
- The agreement with GEDI in the report is not an independent accuracy: the models were trained on these labels.
- All settings: [docs/tool.md](https://github.com/Link-dev/chm-tool/blob/main/docs/tool.md).
""")

nb = dict(nbformat=4, nbformat_minor=5, cells=CELLS,
          metadata=dict(accelerator="GPU", colab=dict(provenance=[], gpuType="T4"),
                        kernelspec=dict(name="python3", display_name="Python 3"),
                        language_info=dict(name="python")))
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell{i:02d}"
    c["source"] = c["source"].splitlines(keepends=True)
out = Path(__file__).with_name("chm_tool_colab.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(out)
