"""Writes chm_pretrained.ipynb next to this file (edit the cells here, not in the notebook): python make_pretrained_notebook.py"""
import json
from pathlib import Path

CELLS = []


def md(text):
    CELLS.append(dict(cell_type="markdown", metadata={}, source=text.strip("\n")))


def code(text):
    meta = {"cellView": "form"} if "#@param" in text or "#@title" in text else {}
    CELLS.append(dict(cell_type="code", metadata=meta, execution_count=None, outputs=[], source=text.strip("\n")))


md(r"""
# Canopy height with the paper's models (chm-tool)

10 m canopy-height maps with the published models of *Leveraging cross-sensor LiDAR observations and Earth
embeddings for canopy height prediction*, without training. Choose the region your study area lies in:

| Region | Models |
|---|---|
| **CONUS** (United States) | UNet-ALS, trained on USGS 3DEP airborne lidar |
| **EBR** (Switzerland), **MRF** (New Zealand), **MUR** (Mexico), **SER** (Malaysia), **SPC** (Brazil) | the site's UNet-SLS, KG-UNet1 and KG-UNet2, trained on local GEDI labels, and UNet-ALS |

Draw the study area inside the region on the map of cell 4; outside the region the maps stay empty. If nothing is
drawn, the region's default study area of about 7 x 7 km is mapped. For other places, train models with
[chm_finetune.ipynb](https://colab.research.google.com/github/Link-dev/chm-tool/blob/main/notebooks/chm_finetune.ipynb).
Documentation: [github.com/Link-dev/chm-tool](https://github.com/Link-dev/chm-tool).

**You need** a Google Cloud project registered for Earth Engine
([free for non-commercial use](https://code.earthengine.google.com/register)). A GPU is not needed (it is faster).
Set everything in cell 1 and run the cells in order.
""")

code(r"""
#@title 1. Settings
REGION = "MRF (New Zealand)"        #@param ["CONUS (United States)", "EBR (Switzerland)", "MRF (New Zealand)", "MUR (Mexico)", "SER (Malaysia)", "SPC (Brazil)"]
GEE_PROJECT = ""                    #@param {type:"string"}
PROJECT_NAME = "my_map"             #@param {type:"string"}
#@markdown Study area: draw it on the map of cell 4 (nothing drawn = the region's default study area) or upload a file
#@markdown (zipped shapefile, GeoJSON, GeoPackage, KML); at most 200 cells of 2.56 km.
STUDY_AREA = "draw on the map"      #@param ["draw on the map", "upload a file"]
#@markdown Year of the inputs (0 = the year the region's models were trained with).
YEAR = 0                            #@param {type:"integer"}
#@markdown Also map the published products GMTCH, GFCH and HRCH for comparison (a little more to download).
BENCHMARKS = False                  #@param {type:"boolean"}
#@markdown **Advanced** (the defaults are fine): parallel Earth Engine requests, Sentinel-1 processing ("local" uses
#@markdown far less Earth Engine quota), and where the projects, code and weights are.
WORKERS = 3                         #@param {type:"integer"}
S1_METHOD = "local"                 #@param ["local", "gee"]
DRIVE_DIR = "/content/drive/MyDrive/chm-tool"                                  #@param {type:"string"}
CODE_SOURCE = "git+https://github.com/Link-dev/chm-tool@main"                  #@param {type:"string"}
WEIGHTS_SOURCE = "https://huggingface.co/datasets/Link-Dev/canopy-height-data/resolve/main/weights"  #@param {type:"string"}

REGION = REGION.split()[0]                   # the code: CONUS, EBR, MRF, MUR, SER or SPC
print(f"region {REGION}, study area: {STUDY_AREA}")
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
from canopy_height.tool import colab, pretrained
print("chm-tool", canopy_height.__version__, "installed")
""")

code(r"""
#@title 3. Earth Engine sign-in
env = colab.environment()
print(f"GPU: {env['gpu'] or 'none (fine, only slower)'}; RAM: {env['ram_gb']} GB")
import ee
from canopy_height.tool.gee import io as gee_io
if not GEE_PROJECT:
    raise ValueError("enter your Earth Engine Cloud project in cell 1 (GEE_PROJECT)")
ee.Authenticate()
gee_io.ee_check(GEE_PROJECT)
print(f"Earth Engine project {GEE_PROJECT}: ok")
""")

code(r"""
#@title 4. Study area (green: the region, red: the study area)
from pathlib import Path
print(pretrained.describe(REGION))
AOI, drawer = pretrained.default_aoi(REGION), None
if STUDY_AREA == "upload a file":
    AOI = colab.upload_aoi()
    print("study area:", Path(AOI).name)
    display(pretrained.region_map(REGION, AOI, cache_dir=DRIVE_DIR))
else:
    drawer = colab.draw_map(AOI, outline=pretrained.region_geometry(REGION, DRIVE_DIR),
                            bounds=pretrained.view_bounds(REGION))
    if drawer is None:
        display(pretrained.region_map(REGION, AOI, cache_dir=DRIVE_DIR))
    else:
        print("Draw the study area inside the green region (rectangle or polygon tools on the left; the last shape "
              "drawn counts), then run cell 5. If nothing is drawn, cell 5 maps the default study area (red).")
        display(drawer)
""")

code(r"""
#@title 5. Map canopy height: download the inputs, then predict (run again to resume)
if drawer is not None:
    AOI = drawer.aoi
    print("study area:", "the shape drawn on the map" if drawer.drawn is not None else "the default study area")
info = pretrained.check(REGION, AOI, YEAR or None, cache_dir=DRIVE_DIR)
print(f"{info['cells']} cells of 2.56 km, {info['inside']:.0%} inside the {REGION} region, inputs of {info['year']}")
for note in info["notes"]:
    print("NOTE:", note)
p = pretrained.open_or_create(Path(DRIVE_DIR) / PROJECT_NAME, REGION, AOI, year=YEAR or None,
                              gee_project=GEE_PROJECT, workers=WORKERS, s1_method=S1_METHOD,
                              benchmarks=["GMTCH", "GFCH", "HRCH"] if BENCHMARKS else [])
MAPS = pretrained.run(p, weights_source=WEIGHTS_SOURCE, weights_dir="/content/chm_weights", workers=WORKERS)
""")

code(r"""
#@title 6. Results
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
print("maps (GeoTIFF, m):", p.path("maps"))
""")

md(r"""
**Notes**
- The maps are in `maps/` of the project folder on Drive (GeoTIFF, metres, the study area's UTM zone), one per model.
- The site models map canopy height in their site's training region. UNet-ALS was trained in CONUS; in the site
  regions its map shows how it transfers. Outside the region the maps are empty.
- A new study area, region or year needs a new `PROJECT_NAME` (run cell 1 and then cell 5: the drawing is kept).
""")

nb = dict(nbformat=4, nbformat_minor=5, cells=CELLS,
          metadata=dict(colab=dict(provenance=[]), kernelspec=dict(name="python3", display_name="Python 3"),
                        language_info=dict(name="python")))
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell{i:02d}"
    c["source"] = c["source"].splitlines(keepends=True)
out = Path(__file__).with_name("chm_pretrained.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(out)
