# chm-tool: canopy height for your own study area

`chm-tool` applies the method of *Leveraging cross-sensor LiDAR observations and Earth embeddings for canopy
height prediction* to any study area. You give it a study area; it downloads the model inputs and gridded GEDI
labels from Google Earth Engine, trains the paper's four local models with the paper's settings, and writes
10 m canopy-height maps next to three published canopy-height products. There is nothing to tune.

| Model | What it is |
|---|---|
| RF-SLS | random forest on the GEDI-labelled pixels of the training region |
| UNet-SLS | UNet trained from scratch on the GEDI labels |
| KG-UNet1 | the UNet pre-trained on US airborne lidar (UNet-ALS), last two blocks fine-tuned on the GEDI labels |
| KG-UNet2 | as KG-UNet1, plus two teachers: UNet-ALS (image-gradient agreement) and the area's UNet-SLS (agreement after 4 x 4 pooling) |

| Benchmark | Product |
|---|---|
| GMTCH | Meta high-resolution canopy height (Tolan et al. 2024), 1 m, aggregated to the 10 m 90th percentile |
| GFCH | UMD global forest canopy height 2019 (Potapov et al. 2021), 30 m |
| HRCH | ETH global canopy height 2020 (Lang et al. 2023), 10 m |

## What you need

- This package with the tool extras (`pip install -e ".[tool]"` in the repository). A CUDA GPU is strongly
  recommended for the UNets. With the paper's batch size of 25, UNet-SLS needs about 14 GB of GPU memory and
  KG-UNet1/2 about 7 GB; on smaller GPUs set a smaller `batch_size` in `train_overrides` (the models then differ
  from the paper's recipe).
- The UNet-ALS checkpoint of the chosen input representation (below), from the assets of the
  [GitHub release](https://github.com/Link-dev/chm-tool/releases), in `weights/source/` of the repository (or in
  `$CHM_WEIGHTS/source/`). KG-UNet1/2 start from it. The Colab notebook downloads it by itself.
- A Google Earth Engine account and a Google Cloud project registered for Earth Engine (free for non-commercial
  use). Sign in once with `earthengine authenticate` (or the *Authenticate* button of the app).

## Quick start

Interface: `chm-tool app` opens a local web page. Upload a study area (shapefile, zipped shapefile, GeoJSON,
GeoPackage, KML) or draw it on the map, enter your Earth Engine project, and press *Run all*. Progress, the log,
the maps and the report are shown on the page; the run continues in the background if the page is closed.

Command line (the same stages):

```bash
chm-tool init my_area --aoi study_area.shp --gee-project my-ee-project
chm-tool run my_area            # plan, download, stack, train, predict, report
chm-tool status my_area
```

Every stage is resumable: run the same command again after an interruption and finished parts are kept.

Google Colab (no local installation): `notebooks/chm_tool_colab.ipynb` runs the same stages in a Colab GPU
session (T4: the paper's batch size fits). The project folder is kept on Google Drive so a run resumes after the
session ends; the training stack is copied to the session's disk for training; only the UNet-ALS checkpoint of
the chosen input representation is fetched (sha256-checked); RF-SLS fits at most 500 000 pixels below 16 GB RAM.
Mind the Drive space: 400 cells with input AE need about 20 GB (the notebook prints the estimate after planning).
The helpers are in `canopy_height.tool.colab`; install with `pip install ".[colab]"` (no Streamlit).

## What the tool does

1. **Plan.** The study area is put on a 10 m grid in its UTM zone, whose origin is the upper-left corner of the
   study area. The grid is cut into 2.56 km cells (256 x 256 pixels, one training chip each). The training region
   covers the study area and extends outwards in whole-kilometre steps until it holds `train_cells` cells
   (default 400, as the paper's international training regions), at most `ring_max_km` (50 km) away; surrounding
   cells that are >= 95 % water (ESA WorldCover 2020) are left out. Smaller training regions train faster but
   give less reliable UNets; below about 100 cells the results should be treated as a demonstration.
2. **Download** (Google Earth Engine, the paper's recipes unchanged):
   - Earth embedding (Google Satellite Embedding, annual), SRTM DEM;
   - Sentinel-1 VV/VH: GRD, both orbits, border-noise mask, multi-temporal Quegan speckle filter (15 x 15 boxcar,
     10 images), volume-model terrain flattening, dB, annual median;
   - Sentinel-2 B2-B8, B11, B12: surface reflectance, cloud probability < 20 %, scenes < 40 % cloud, QA60
     cloud/cirrus mask, annual median;
   - for the seasonal representations, the same Sentinel-1 / Sentinel-2 as four seasonal medians (DJF, MAM, JJA,
     SON of December of the year before to November; Sentinel-2 cloud limits 50 %);
   - GEDI L2A monthly rh95 (quality flag 1, degrade flag 0), median of 2019-01-01 to 2021-12-30, without
     WorldCover built-up cells: the training labels;
   - the three benchmarks for the study-area cells.
3. **Stack**: the training chips (76-band annual input, 44-band seasonal input if needed, GEDI labels) and the
   study-area mosaics.
4. **Train** RF-SLS, UNet-SLS, KG-UNet1 and KG-UNet2 with the paper's settings (see [models.md](models.md)). KG-UNet2 uses the
   recipe defaults (50 epochs, 4 x 4 pooling, teacher terms from epoch 11);
   the per-site settings of the paper are listed in [models.md](models.md) and can be set through `train_overrides`.
5. **Predict** every model over the study area (256 px tiles with 32 px blended overlaps; large areas block by
   block with 256 px of context).
6. **Report** (`report/metrics.csv`, `report/summary.json`, quick-look PNGs):
   - GEDI labels of the validation chips (never seen by the UNets; RF-SLS saw 80 % of all labelled pixels);
   - statistics of every map and benchmark inside the study area and their agreement with the GEDI labels there.
     This is *not* an independent accuracy: the local models were trained on these labels and HRCH / GFCH were
     calibrated with GEDI data of the same period;
   - if you give airborne-lidar canopy height (`--als`; 1 m rasters are aggregated to the 10 m 90th percentile,
     10 m rasters on the grid are used as they are; `--als-scale 0.01` for centimetres): the paper's accuracy
     convention (256 px chips with >= 1 % of cells above 1 m, reference 1-80 m, chip-median RMSE and r2, pooled
     mean error).

## Input representations

| `--input` | Model input | UNet-ALS used | Downloads |
|---|---|---|---|
| **AE** (default, the paper's main results) | Earth embedding + annual Sentinel-1/2 + DEM (76 channels) | `UNet-ALS.pth` | embedding, annual S1 + S2 |
| A | annual Sentinel-1/2 + DEM (12) | `UNet-A-ALS.pth` | annual S1 + S2 |
| E | Earth embedding (64) | `UNet-E-ALS.pth` | embedding only: by far the cheapest |
| T | four-season Sentinel-1/2 + DEM (45) | `UNet-T-ALS.pth` | seasonal S1 + S2 |
| TE | Earth embedding + four-season Sentinel-1/2 + DEM (109) | `UNet-TE-ALS.pth` | embedding, seasonal S1 + S2 |

Sentinel-1 is processed by default on the computer that runs the tool (`s1_method: local`, see below): about 10
EECU-seconds per cell (annual, or the four seasons together) plus about 50 MB of raw scenes per cell passing
through the network (up to ~250 MB where there are many acquisitions), so a representation with Sentinel-1 costs
about 80-130 EECU-seconds per cell in all (E about 10). Processed on Earth Engine (`s1_method: gee`) Sentinel-1
alone takes about 150-500 EECU-seconds per cell (it grows with the number of acquisitions). Changing the representation of a project later downloads the missing
layers and trains the models again.

Letters: **A** = annual Sentinel-1/2 + DEM, **E** = Earth embedding (Google Satellite Embedding), **T** = four
seasonal (temporal) Sentinel-1/2 composites + DEM. AE and A were called IE and I in earlier versions (and
`UNet-A-ALS.pth` was `UNet-S-ALS.pth`); the earlier names are still accepted in settings, checkpoints and project
files.

## Settings (`project.yaml`)

`input`, `year` (2019-2024: Sentinel-2 surface reflectance is global from December 2018, GEDI starts in April
2019; the paper used 2019 and 2020), `gee_project`, `train_cells`, `ring_max_km`, `water_max`, `gedi_window`,
`built_up_mask`, `benchmarks`, `models`, `train_overrides`, `rf_max_rows` (RF-SLS fits at most 2 million labelled
pixels, above the paper's training sets), `device`, `workers` (parallel Earth Engine requests), `s1_method`,
`als`, `als_scale`, `epsg` (a projected CRS in metres; default: UTM zone). Settings that change the data (year, GEDI
window, input) are checked against the downloaded files, and stale stacks and models are rebuilt; earlier model
runs are kept as `models/<model>.stale-<time>`.

### Sentinel-1 processing (`s1_method`)

The Sentinel-1 composites (inputs AE, A, T, TE) are the gee_s1_ard chain: border-noise mask, multi-temporal Quegan
speckle filter (15 x 15 boxcar, the 10 closest acquisitions), volume-model terrain flattening on SRTM, dB, median.

- `local` (default): Earth Engine only serves the raw scenes of a cell (VV, VH, their validity, the SRTM elevation
  on each scene's grid, the incidence angle; 2-3 requests per cell, about 50 MB for 40 scenes) and the chain runs
  in numpy on the tool's computer (a few seconds per cell). The acquisition metadata (each acquisition's filter
  set and look direction) is requested once for all cells and kept in `raw/s1_local_<year>[_seasonal].json`.
  Earth Engine's own arithmetic is reproduced (masked pixels in the convolution, bilinear resampling, the
  terrain algorithms' grid and pixel sizes, the histogram median above 64 acquisitions), so the result equals the
  Earth Engine way to about 1e-5 dB. Billed Earth Engine compute per cell is about a tenth of the Earth Engine way
  (about 10 instead of about 140 EECU-seconds with ~30 acquisitions a year). Cells whose CRS is not a WGS84 UTM
  zone take the Earth Engine way.
- `gee`: the chain runs on Earth Engine, as in the paper's pipeline (bit for bit).

Rasters record the way in their `s1_method` tag; changing the setting keeps existing downloads.

## Run times (measured on an RTX 3090 Ti, 20 CPU threads)

| Step | Example | Time |
|---|---|---|
| Download, annual inputs + GEDI + benchmarks | 16 cells, 1 Earth Engine worker, project in throttled mode | 12 min |
| Download, seasonal S1/S2 (T / TE) | 16 cells | 9 min |
| RF-SLS | training region of 439 cells, 380 k pixels (CPU) | 8 min |
| UNet-SLS / KG-UNet1 / KG-UNet2 | training region of 439 cells | 22 / 10 / 10 min |
| Predict + report | 2 x 2 cells | seconds |

A training region of the default 400 cells therefore needs a few hours of Earth Engine downloads (depending on
the project's quota) and about one hour of training.

**Small training regions.** The UNets need many chips: with only a handful of cells one batch per epoch holds the
whole region, the BatchNorm statistics adapt to those few chips, and the UNet maps are not reliable (RF-SLS is less
affected). Use at least about 100 cells, preferably the default 400.

## Reproducibility

- Downloads use the paper's Earth Engine recipes and the training stacks are encoded as in the paper. Earth Engine
  collections change over time (e.g. Sentinel-2 reprocessing), so inputs downloaded later can differ slightly
  from the paper's.
- Training is repeatable bit for bit on the same GPU and library versions; another GPU model or other library
  versions give slightly different weights.

## Limits

- Earth Engine quota: a 400-cell training region costs roughly 30-40 thousand EECU-seconds with the local
  Sentinel-1 way (about 90 thousand or more with `s1_method: gee`); non-commercial projects that exceed their
  monthly quota are throttled (the tool waits and retries).
- Study areas that cross the 180th meridian must be split; very large or scattered study areas are better run as
  several projects (the plan stage warns).
- The maps are as good as the GEDI labels of the region: cloud-contaminated GEDI shots and GEDI's saturation in
  tall, dense forest propagate into the local models (see the paper).
