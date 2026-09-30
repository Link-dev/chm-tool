"""canopy-height tool: from a study area to canopy-height maps with the paper's four local models.

A reader picks a study area; the tool downloads the model inputs and gridded GEDI labels from Google Earth
Engine, trains RF-SLS, UNet-SLS, KG-UNet1 and KG-UNet2 with the paper's settings and writes their maps next to
the published benchmarks GMTCH, GFCH and HRCH, without settings to tune. Command line: `chm-tool` or
`python -m canopy_height.tool`; interface: `chm-tool app`; from Python:

    from canopy_height.tool import Project, workflow
    p = Project.create("projects/my_area", "study_area.geojson", gee_project="my-project")
    workflow.run(p)

Importing the package loads no PyTorch or Earth Engine code; the stages import what they need.
"""
from .project import BENCHMARKS, DEFAULTS, MODEL_NAMES, MODELS, STAGES, Project

__all__ = ["Project", "STAGES", "MODELS", "MODEL_NAMES", "BENCHMARKS", "DEFAULTS"]
