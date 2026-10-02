# chm-tool

10 m canopy-height maps with the models of *Leveraging cross-sensor LiDAR observations and Earth embeddings for
canopy height prediction*. Two Google Colab notebooks, nothing to install:

| Notebook | What it does | Where | Time |
|---|---|---|---|
| **Map with the paper's models** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Link-dev/chm-tool/blob/main/notebooks/chm_pretrained.ipynb) | maps a study area with the published models, no training | the contiguous US and the training regions of five international sites | minutes |
| **Train models for your study area** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Link-dev/chm-tool/blob/main/notebooks/chm_finetune.ipynb) | trains RF-SLS, UNet-SLS, KG-UNet1 and KG-UNet2 on local GEDI labels and maps the study area | anywhere | hours |

| Model | What it is |
|---|---|
| UNet-ALS | UNet pre-trained on US airborne lidar (USGS 3DEP) |
| RF-SLS | random forest on the GEDI-labelled pixels |
| UNet-SLS | UNet trained from scratch on the GEDI labels |
| KG-UNet1 | UNet-ALS fine-tuned on the GEDI labels |
| KG-UNet2 | as KG-UNet1, guided by UNet-ALS and UNet-SLS as teachers |

Benchmarks shown alongside: GMTCH (Meta, [Tolan et al. 2024](https://doi.org/10.1016/j.rse.2023.113888)),
GFCH (UMD, [Potapov et al. 2021](https://doi.org/10.1016/j.rse.2020.112165)), HRCH (ETH, [Lang et al. 2023](https://doi.org/10.1038/s41559-023-02206-6)).

## Map with the paper's models

Choose one of six regions: **CONUS** (United States, UNet-ALS) or the training region of an international site,
**EBR** (Switzerland), **MRF** (New Zealand), **MUR** (Mexico), **SER** (Malaysia) or **SPC** (Brazil), each with its
UNet-SLS, KG-UNet1 and KG-UNet2, and UNet-ALS. Draw the study area inside the region (at most 200 cells of 2.56 km)
or upload it as a file; if nothing is drawn, the region's default study area is mapped. Outside the region the maps
stay empty. Only the study area's inputs are downloaded, so a small study area takes minutes.

## Train models for your study area

The notebook downloads the inputs and GEDI labels of a training region around the study area (200 cells of 2.56 km
by default), trains the four local models with the paper's settings and maps the study area. It needs a T4 GPU
runtime for the UNets and about 10 GB on Google Drive (the notebook prints the estimate for your area); the project
is kept on Drive, so running the notebook again resumes it.

## On your own computer

```bash
git clone https://github.com/Link-dev/chm-tool
cd chm-tool
pip install -e ".[tool]"
```

Download the UNet-ALS checkpoint of the input you will use from
[`weights/source/`](https://huggingface.co/datasets/Link-Dev/canopy-height-data/tree/main/weights/source) into
`weights/source/` (`UNet-ALS.pth` for the default input AE), sign in to Earth Engine once, and start the local web
interface or the command line:

```bash
earthengine authenticate
chm-tool app
chm-tool init my_area --aoi study_area.shp --gee-project my-ee-project
chm-tool run my_area
```

A CUDA GPU is strongly recommended for the UNets. Every stage can be resumed after an interruption.

## Documentation

- [docs/tool.md](docs/tool.md): what the tool does, settings, Sentinel-1 processing, run times, limits
- [docs/models.md](docs/models.md): the models and the `chm` command (train, predict and evaluate on chip stacks)

## Weights and data

All in the Hugging Face dataset [Link-Dev/canopy-height-data](https://huggingface.co/datasets/Link-Dev/canopy-height-data):

- `weights/source/`: UNet-ALS of every input, trained on canopy-height models derived from USGS 3DEP lidar
  ([Allred et al. 2025](https://doi.org/10.1038/s41597-025-04655-z));
- `weights/<SITE>/`: the paper's models of the five international sites (and NEON);
- the paper's training and evaluation data.

## Citation

If you use this tool, please cite the paper (see [CITATION.cff](CITATION.cff)):

> Zhou, J., et al. Leveraging cross-sensor LiDAR observations and Earth embeddings for canopy height prediction.
> [Journal, year, DOI]

## License

Code and weights: [MIT](LICENSE). The Sentinel-1 processing includes gee_s1_ard ([Mullissa et al. 2021](https://doi.org/10.3390/rs13101954), MIT), see
`src/canopy_height/tool/gee/third_party/gee_s1_ard/LICENSE`.
