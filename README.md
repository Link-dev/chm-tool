# chm-tool

10 m canopy-height maps for your own study area, with the models of *Leveraging cross-sensor LiDAR observations
and Earth embeddings for canopy height prediction*.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Link-dev/chm-tool/blob/main/notebooks/chm_tool_colab.ipynb)

You choose a study area. The tool downloads the model inputs (Sentinel-1/2, Google Satellite Embedding, SRTM) and
gridded GEDI labels from Google Earth Engine, trains the paper's four local models with the paper's settings, and
writes 10 m canopy-height maps next to three published products, with a short report. There is nothing to tune.

| Model | What it is |
|---|---|
| RF-SLS | random forest on the GEDI-labelled pixels |
| UNet-SLS | UNet trained from scratch on the GEDI labels |
| KG-UNet1 | UNet pre-trained on US airborne lidar (UNet-ALS), fine-tuned on the GEDI labels |
| KG-UNet2 | as KG-UNet1, guided by UNet-ALS and UNet-SLS as teachers |

Benchmarks shown alongside: GMTCH (Meta, [Tolan et al. 2024](https://doi.org/10.1016/j.rse.2023.113888)),
GFCH (UMD, [Potapov et al. 2021](https://doi.org/10.1016/j.rse.2020.112165)), HRCH (ETH, [Lang et al. 2023](https://doi.org/10.1038/s41559-023-02206-6)).

## Google Colab (nothing to install)

1. Click **Open in Colab** above.
2. In cell 1, enter your Google Earth Engine Cloud project
   ([free registration](https://code.earthengine.google.com/register) for non-commercial use), and choose the input
   and the study area (a box, a shape drawn on a map, or an uploaded file).
3. Run the cells in order. The project is kept on your Google Drive, so running the notebook again resumes it.

You need a Google account, a T4 GPU runtime for the UNets (downloads can run on a CPU runtime first), and space on
Google Drive: about 20 GB for the default training region of 400 cells with input AE (the notebook prints the
estimate for your area).

## On your own computer

```bash
git clone https://github.com/Link-dev/chm-tool
cd chm-tool
pip install -e ".[tool]"
```

Download the UNet-ALS checkpoint of the input you will use from the
[release](https://github.com/Link-dev/chm-tool/releases/tag/v1.0.0) into `weights/source/` (`UNet-ALS.pth` for the
default input AE), sign in to Earth Engine once, and start the local web interface or the command line:

```bash
earthengine authenticate
chm-tool app
chm-tool init my_area --aoi study_area.shp --gee-project my-ee-project
chm-tool run my_area
```

A CUDA GPU is strongly recommended for the UNets. Every stage can be resumed after an interruption.

## Input representations

| Input | Model input | Weights |
|---|---|---|
| **AE** (default) | Earth embedding + annual Sentinel-1/2 + DEM | `UNet-ALS.pth` |
| A | annual Sentinel-1/2 + DEM | `UNet-A-ALS.pth` |
| E | Earth embedding | `UNet-E-ALS.pth` |
| T | four-season Sentinel-1/2 + DEM | `UNet-T-ALS.pth` |
| TE | Earth embedding + four-season Sentinel-1/2 + DEM | `UNet-TE-ALS.pth` |

## Documentation

- [docs/tool.md](docs/tool.md): what the tool does, settings, Sentinel-1 processing, run times, limits
- [docs/models.md](docs/models.md): the models and the `chm` command (train, predict and evaluate on chip stacks)

## Weights and data

- UNet-ALS weights: assets of the [v1.0.0 release](https://github.com/Link-dev/chm-tool/releases/tag/v1.0.0),
  trained on canopy-height models derived from USGS 3DEP lidar ([Allred et al. 2025](https://doi.org/10.1038/s41597-025-04655-z)).
- The paper's evaluation data: Zenodo ([DOI]).

## Citation

If you use this tool, please cite the paper (see [CITATION.cff](CITATION.cff)):

> Zhou, J., et al. Leveraging cross-sensor LiDAR observations and Earth embeddings for canopy height prediction.
> [Journal, year, DOI]

## License

Code and weights: [MIT](LICENSE). The Sentinel-1 processing includes gee_s1_ard ([Mullissa et al. 2021](https://doi.org/10.3390/rs13101954), MIT), see
`src/canopy_height/tool/gee/third_party/gee_s1_ard/LICENSE`.
