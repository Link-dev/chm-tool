# The models and the `chm` command

Train and apply the canopy-height models of *Leveraging cross-sensor LiDAR observations and Earth
embeddings for canopy height prediction*: a UNet pre-trained on US airborne lidar (UNet-ALS), and four
models trained on local GEDI labels (RF-SLS, UNet-SLS, KG-UNet1, KG-UNet2).

**Your own study area:** `chm-tool` (sub-package `canopy_height.tool`, see [tool.md](tool.md)) downloads
the inputs and GEDI labels for any study area from Google Earth Engine with the paper's recipes, trains the four
local models below with the paper's settings and maps canopy height next to GMTCH, GFCH and HRCH, from the command
line or a local web interface (`chm-tool app`).

## Install

```bash
pip install -e .            # Python >= 3.10; PyTorch with CUDA recommended for UNets
```

Dependencies: numpy, torch, scikit-learn, rasterio, pyyaml. RF-SLS needs only numpy and scikit-learn.

## Data format

| Array | Shape, dtype | Content |
|---|---|---|
| annual input | `[n, 76, 256, 256]` uint16 | 0-63 annual Earth embedding, 64 DEM, 65-66 Sentinel-1 VV/VH, 67-75 Sentinel-2 B2-B8, B11, B12; 0 = no data |
| labels | `[n, 256, 256]` float32 | canopy height in m; -999 = missing |

Chip stacks may be split over several `.npy` files (pass them in order; they are memory-mapped).
GeoTIFFs on a common 10 m grid can be cut into chips with `chm tile`, and 1 m ALS canopy height can be
aggregated to the 10 m grid (90th percentile of the 10 x 10 cells) with `chm prepare-als`.

## Weights

`UNet-ALS.pth` is the source-domain UNet (76-channel input). It is used for prediction without local
training, and it is the starting point and ALS teacher of KG-UNet1 / KG-UNet2. It and the UNet-ALS of the other
input representations are assets of the [GitHub release](https://github.com/Link-dev/chm-tool/releases) (put them
in `weights/source/`). Checkpoints store their input layout and normalisation; `chm info <checkpoint>` shows them.

## Train on local GEDI labels

The four models of the paper, for one site (inputs `X*.npy`, GEDI labels `G*.npy`):

```bash
chm train --recipe rf-sls   --annual X1.npy X2.npy --labels G1.npy G2.npy --out runs/rf
chm train --recipe unet-sls --annual X1.npy X2.npy --labels G1.npy G2.npy --out runs/sls
chm train --recipe kg-unet1 --annual X1.npy X2.npy --labels G1.npy G2.npy --out runs/kg1 \
          --base weights/source/UNet-ALS.pth
chm train --recipe kg-unet2 --annual X1.npy X2.npy --labels G1.npy G2.npy --out runs/kg2 \
          --base weights/source/UNet-ALS.pth --teacher-sls runs/sls/model.pth
```

KG-UNet2 needs the site's UNet-SLS as its GEDI teacher, so train UNet-SLS first.

| Recipe | Start | Trained blocks | Defaults |
|---|---|---|---|
| `rf-sls` | - | - | pixels with label > 0; 80 % fit a RandomForest (600 trees, max_depth 50, min_samples_leaf 4, max_features sqrt, random_state 42) |
| `unet-sls` | random | all | 100 epochs, batch 25, Adam 5e-4, lr x 0.5 after epoch 50, last epoch kept |
| `kg-unet1` | UNet-ALS | `up4`, `outc` | as unet-sls, 50 epochs |
| `kg-unet2` | UNet-ALS | `up4`, `outc` | as kg-unet1; loss 0.5 x label + 0.25 x gradient agreement with UNet-ALS + 0.25 x agreement with UNet-SLS after 4 x 4 average pooling, teacher terms from epoch 11; zero labels treated as missing |

All UNet recipes use an 80/20 chip split (`train_test_split`, random_state 42) and seed 42. Every
setting can be changed on the command line (`chm train --help`) or with `--config file.yaml`.

The paper tuned four KG-UNet2 settings per international site (the recipe defaults are those of SER):

| Site | `epochs` | `kg2_pool` | `kg2_gate` (teacher terms from epoch) | `zero_to_nodata` |
|---|---|---|---|---|
| SER | 50 | 4 | 10 (11) | true |
| SPC | 30 | 4 | 10 (11) | true |
| MUR | 50 | 8 | 10 (11) | true |
| EBR | 50 | 8 | 10 (11) | true |
| MRF | 30 | 4 | 5 (6) | false |

Outputs: `model.pth` (UNets) or `rf_sls.joblib`, plus `result.json`, and for UNets `history.csv` and
`split.npz`.

RF-SLS feature encoding (`--encoding`): `paper` (default) subtracts 10000 from the embedding channels
in unsigned 16-bit arithmetic, so values below 10000 wrap around; this is how the paper's RF-SLS models
were trained. `plain` uses the raw values and is recommended for new models.

## Predict

```bash
chm predict --model weights/source/UNet-ALS.pth --annual X.npy --out pred.npy          # chip stack
chm predict --model runs/kg2/model.pth --annual site_annual.tif --out chm.tif    # GeoTIFF, any size
chm predict --model runs/rf/rf_sls.joblib --annual X.npy --out pred_rf.npy
```

GeoTIFFs are predicted in 256 x 256 tiles with 32-pixel overlaps that are blended linearly. Output is a
float32 canopy-height raster, NaN where all input bands are 0. TF32 is disabled by default
(`--tf32` to enable).

`--zero-restore off` keeps normalised values in no-data input cells instead of resetting them to 0. The
paper's NEON predictions of UNet-ALS, KG-UNet1 and KG-UNet2 were made this way; everything else uses
the checkpoint setting.

## Evaluate

```bash
chm evaluate --pred pred.npy --ref als_p90.npy                         # international-site convention
chm evaluate --pred pred.npy --ref als_p95.npy --mask-ref als_p90.npy  # p95 on the p90 pixels
chm evaluate --pred pred.npy --ref als_p90.npy --no-cap80              # NEON convention (use pooled)
```

Evaluated cells have reference > 1 m. References of -999 or < 0 are excluded, and so are references
> 80 m unless `--no-cap80` is given. The output gives medians over chips of RMSE, MAE, r2 (squared
Pearson correlation) and mean error, and pooled metrics over all cells.

## Reproducibility

With the tested versions (PyTorch 2.6.0 / CUDA 12.4 on an NVIDIA A100; scikit-learn 1.5.1 for RF-SLS),
retraining reproduces the paper's weights exactly. Other GPUs or library
versions can change results slightly.
