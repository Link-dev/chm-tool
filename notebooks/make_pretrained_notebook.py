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
| **CONUS** (contiguous United States) | UNet-ALS, trained on USGS 3DEP airborne lidar |
| **EBR** Entlebuch (Switzerland), **MRF** Mount Richmond (New Zealand), **MUR** Middle Usumacinta (Mexico), **SER** Sepilok and Danum (Malaysia), **SPC** São Paulo (Brazil) | the site's UNet-SLS, KG-UNet1 and KG-UNet2, trained on local GEDI labels |

Draw the study area inside the region (cell 4 shows it on a map); outside the region the maps stay empty. Each
region has a default study area of about 7 x 7 km that runs as it is. For other places, train models with
[chm_finetune.ipynb](https://colab.research.google.com/github/Link-dev/chm-tool/blob/main/notebooks/chm_finetune.ipynb).
Documentation: [github.com/Link-dev/chm-tool](https://github.com/Link-dev/chm-tool).

**You need** a Google Cloud project registered for Earth Engine
([free for non-commercial use](https://code.earthengine.google.com/register)). A GPU is not needed (it is faster).
Set everything in cell 1 and run the cells in order.
""")

code(r"""
#@title 1. Settings
REGION = "MRF: Mount Richmond Forest Park, New Zealand"  #@param ["CONUS: Contiguous United States", "EBR: Entlebuch Biosphere Reserve, Switzerland", "MRF: Mount Richmond Forest Park, New Zealand", "MUR: Middle Usumacinta, Mexico", "SER: Sepilok and Danum Valley, Malaysia", "SPC: Sao Paulo, Brazil"]
GEE_PROJECT = ""                    #@param {type:"string"}
PROJECT_NAME = "my_map"             #@param {type:"string"}
#@markdown Study area: the region's default area, a longitude / latitude box, a shape drawn on a map (cell 4), a file
#@markdown on Google Drive, or an uploaded file (zipped shapefile, GeoJSON, GeoPackage, KML); at most 200 cells of 2.56 km.
AOI_MODE = "region default"         #@param ["region default", "box", "draw", "drive file", "upload"]
WEST = 173.44                       #@param {type:"number"}
SOUTH = -41.29                      #@param {type:"number"}
EAST = 173.53                       #@param {type:"number"}
NORTH = -41.22                      #@param {type:"number"}
AOI_FILE = "/content/drive/MyDrive/chm-tool/study_area.geojson"                #@param {type:"string"}
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

REGION = REGION.split(":")[0].strip()        # the code before the colon
print(f"region {REGION}, study area: {AOI_MODE}")
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
from shapely.geometry import mapping
drawer = None
print(pretrained.describe(REGION))
if AOI_MODE == "region default":
    AOI = pretrained.default_aoi(REGION)
elif AOI_MODE == "box":
    AOI = colab.aoi_from_bbox(WEST, SOUTH, EAST, NORTH)
elif AOI_MODE == "draw":
    w, s, e, n = pretrained.default_aoi(REGION).bounds
    drawer = colab.AoiDrawer(center=((s + n) / 2, (w + e) / 2), zoom=4 if REGION == "CONUS" else 10,
                             outline=mapping(pretrained.region_geometry(REGION, DRIVE_DIR)))
    print("Draw a rectangle or polygon inside the green region (tools on the left), then run cell 5.")
    display(drawer)
elif AOI_MODE == "drive file":
    AOI = AOI_FILE
    if not Path(AOI).exists():
        raise FileNotFoundError(f"{AOI} not found (AOI_FILE in cell 1)")
else:
    from google.colab import files
    up = files.upload()
    name = next(iter(up))
    AOI = str(Path("/content") / name)
    Path(AOI).write_bytes(up[name])
if drawer is None:
    info = pretrained.check(REGION, AOI, YEAR or None, cache_dir=DRIVE_DIR)
    print(f"study area: {info['cells']} cells of 2.56 km, {info['inside']:.0%} inside the {REGION} region, "
          f"inputs of {info['year']}")
    for note in info["notes"]:
        print("NOTE:", note)
    display(pretrained.region_map(REGION, AOI, cache_dir=DRIVE_DIR))
""")

code(r"""
#@title 5. Map canopy height: download the inputs, then predict (run again to resume)
if drawer is not None:
    if drawer.geometry is None:
        raise ValueError("no shape drawn yet: draw the study area on the map of cell 4")
    AOI = drawer.geometry
    info = pretrained.check(REGION, AOI, YEAR or None, cache_dir=DRIVE_DIR)
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
- The models map canopy height where they were trained: UNet-ALS in the contiguous United States, the site models
  in their site's training region. Outside the region the maps are empty.
- A new study area, region or year needs a new `PROJECT_NAME`.
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
