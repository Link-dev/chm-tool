"""Canopy height with the paper's published models, without training (notebooks/chm_pretrained.ipynb).

Six regions (REGIONS): CONUS, where UNet-ALS was trained (USGS 3DEP airborne lidar across the contiguous United
States), and the training regions of the five international sites, where their UNet-SLS, KG-UNet1 and KG-UNet2 were
trained on local GEDI labels; UNet-ALS maps the site regions too. A study area is mapped inside the chosen region
only (NaN outside). Only the study-area cells are downloaded (input AE: Earth embedding, annual Sentinel-1/2, DEM;
no GEDI), at most MAX_CELLS cells of 2.56 km.

    p = pretrained.create("projects/danum", "SER", pretrained.default_aoi("SER"), gee_project="my-project")
    pretrained.run(p)            # plan, download, mosaic, weights, maps/<model>.tif
"""
import json
import math
import time
from importlib import resources
from pathlib import Path

import numpy as np

from . import grid, weights
from .project import Project, input_layers, replace_retry, run_lock

INPUT = "AE"                     # the input of every published site model and of UNet-ALS.pth
MAX_CELLS = 200                  # largest study area (2.56 km cells)
REGION_FILE = "region.json"      # the chosen region, in the project folder
CONUS_FILE = "conus.geojson"     # cache of the CONUS outline from Earth Engine
STATES = "TIGER/2018/States"
NOT_CONUS = ["AK", "HI", "PR", "VI", "GU", "AS", "MP"]
ALS_WEIGHTS = "source/UNet-ALS.pth"   # UNet-ALS (input AE), mapped in every region
CONUS = dict(region="CONUS", country="United States", year=2020, models=["UNet-ALS"], weights="source",
             default_box=[-72.22, 42.50, -72.13, 42.57])                   # around Harvard Forest, Massachusetts


def _sites():
    with resources.files("canopy_height").joinpath("data/regions.geojson").open("r", encoding="utf-8") as fh:
        fc = json.load(fh)
    return {f["properties"]["region"]: dict(f["properties"], geometry=f["geometry"]) for f in fc["features"]}


REGIONS = {"CONUS": CONUS, **_sites()}


def label(region):
    """'MRF (New Zealand)'."""
    return f"{region} ({REGIONS[region]['country']})"


def describe(region):
    r = REGIONS[region]
    if region == "CONUS":
        return f"{label(region)}: UNet-ALS, trained with {r['year']} inputs"
    return (f"{label(region)}: {', '.join(r['models'])} (trained here with {r['year']} inputs) and UNet-ALS "
            f"(trained in CONUS)")


def model_files(region):
    """{model name: path of its weights under weights/ of the published dataset}: UNet-ALS, then the site's models."""
    r = REGIONS[region]
    return {"UNet-ALS": ALS_WEIGHTS, **{m: f"{r['weights']}/{m}.pth" for m in r["models"] if m != "UNet-ALS"}}


def default_aoi(region):
    """A study area of about 3 x 3 cells inside the region (near the site's ALS test area)."""
    import shapely
    return shapely.box(*REGIONS[region]["default_box"])


def view_bounds(region):
    """(west, south, east, north) a map of the region opens on: the whole region of a site; for CONUS the default
    study area and about 20 km around it."""
    if region == "CONUS":
        w, s, e, n = REGIONS[region]["default_box"]
        return w - 0.25, s - 0.2, e + 0.25, n + 0.2
    from shapely.geometry import shape
    return shape(REGIONS[region]["geometry"]).bounds


def region_geometry(region, cache_dir=None):
    """The region as a shapely geometry in EPSG:4326. CONUS comes from Earth Engine (the 48 contiguous states and
    DC of TIGER/2018/States; Earth Engine must be initialised), cached in cache_dir/conus.geojson."""
    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union
    if region != "CONUS":
        return shape(REGIONS[region]["geometry"])
    f = Path(cache_dir) / CONUS_FILE if cache_dir else None
    if f is not None and f.exists():
        return shape(json.loads(f.read_text(encoding="utf-8")))
    import ee
    states = ee.FeatureCollection(STATES).filter(ee.Filter.inList("STUSPS", NOT_CONUS).Not())
    geom = shape(states.union(maxError=1000).geometry().simplify(maxError=1000).getInfo())
    parts = getattr(geom, "geoms", [geom])                  # a GeometryCollection: keep its polygons
    geom = unary_union([g for g in parts if g.geom_type in ("Polygon", "MultiPolygon")])
    if f is not None:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(mapping(geom)), encoding="utf-8")
    return geom


def _aoi_cells(aoi):
    """(grid, cells of the study area as a DataFrame, study area in the grid CRS) on the tool's 10 m UTM grid."""
    epsg = grid.utm_epsg(aoi)
    aoi_xy = grid.to_crs(aoi, epsg)
    g = grid.make_grid(aoi_xy, epsg)
    cells, _ = grid._evaluate(aoi_xy, g, 0.0, None)
    cells = cells[cells.area >= grid.MIN_AREA].sort_values(["y1", "x0"], ascending=[False, True])
    return g, cells, aoi_xy


def check(region, aoi, year=None, cache_dir=None):
    """Size of a study area and its share inside the region -> dict(cells, inside, year, notes); ValueError when it
    does not overlap the region or has more than MAX_CELLS cells."""
    aoi = grid.read_aoi(aoi)
    g, cells, aoi_xy = _aoi_cells(aoi)
    reg = grid.to_crs(region_geometry(region, cache_dir), g["epsg"])
    inside = float(aoi_xy.intersection(reg).area / aoi_xy.area) if aoi_xy.area else 0.0
    if inside == 0:
        raise ValueError(f"the study area is outside the {region} region: draw it inside the region (the map shows "
                         f"the region), choose the region it lies in, or train models for it with "
                         f"chm_finetune.ipynb")
    if len(cells) > MAX_CELLS:
        raise ValueError(f"the study area covers {len(cells)} cells of 2.56 km, more than {MAX_CELLS}: draw a smaller "
                         f"one or split it")
    year = int(year or REGIONS[region]["year"])
    notes = []
    if inside < 0.999:
        notes.append(f"{1 - inside:.0%} of the study area is outside the {region} region and is left empty")
    if year != REGIONS[region]["year"]:
        notes.append(f"the {region} models were trained with {REGIONS[region]['year']} inputs, not {year}")
    return dict(cells=len(cells), inside=inside, year=year, notes=notes)


def create(root, region, aoi, year=None, gee_project=None, benchmarks=(), workers=None, s1_method=None):
    """New project folder for mapping `aoi` with the models of `region` (REGIONS)."""
    if region not in REGIONS:
        raise ValueError(f"region {region!r}: choose one of {', '.join(REGIONS)}")
    p = Project.create(root, aoi, name=Path(root).name, input=INPUT, year=int(year or REGIONS[region]["year"]),
                       gee_project=gee_project, benchmarks=list(benchmarks), workers=workers, s1_method=s1_method)
    p.path(REGION_FILE).write_text(json.dumps(dict(region=region)), encoding="utf-8")
    return p


def open_or_create(root, region, aoi, year=None, **settings):
    """The project in `root` if it maps the same study area with the same region and year (its downloads are kept),
    else a new one; ValueError if `root` holds another study area, region or year."""
    root = Path(root)
    if not (root / "project.yaml").exists():
        return create(root, region, aoi, year=year, **settings)
    p = Project(root)
    year = int(year or REGIONS[region]["year"])
    same_aoi = grid.read_aoi(p.aoi_file).equals_exact(grid.read_aoi(aoi), 1e-7)
    if region_of(p) != region or not same_aoi or int(p["year"]) != year:
        raise ValueError(f"{root} holds another study area, region or year: choose another PROJECT_NAME")
    for k, v in settings.items():
        if v is not None:
            p.cfg[k] = list(v) if k == "benchmarks" else v
    p.save()
    return p


def region_of(project):
    f = project.path(REGION_FILE)
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: create the project with pretrained.create")
    return json.loads(f.read_text(encoding="utf-8"))["region"]


def plan(project):
    """plan/grid.json and plan/cells.csv of the study-area cells only (no surrounding training cells)."""
    import pandas as pd
    aoi = grid.read_aoi(project.aoi_file)
    g, cells, aoi_xy = _aoi_cells(aoi)
    if cells.empty:
        raise ValueError("the study area does not cover 1 m^2 of any grid cell")
    if len(cells) > MAX_CELLS:
        raise ValueError(f"the study area covers {len(cells)} cells of 2.56 km, more than {MAX_CELLS}")
    if grid.aoi_pixels(aoi_xy, cells, need=1) == 0:
        raise ValueError("the study area covers no 10 m pixel centre: draw a larger study area")
    out = cells.assign(role="aoi", dist_m=0, water=np.nan, built=np.nan, tree=np.nan, use=True)
    grid._write_json(g, project.grid_file)
    grid._write_csv(pd.DataFrame(out)[grid.CELL_COLS], project.cells_file)
    project.log(f"plan: {len(out)} cells of 2.56 km, EPSG:{g['epsg']}")
    return out


def region_mask(region, profile, cache_dir=None):
    """Pixels of the raster grid `profile` whose centre lies in the region."""
    from rasterio.features import rasterize
    geom = grid.to_crs(region_geometry(region, cache_dir), profile["crs"].to_epsg())
    return rasterize([(geom, 1)], out_shape=(profile["height"], profile["width"]), transform=profile["transform"],
                     fill=0, dtype="uint8").astype(bool)


def run(project, weights_source=weights.WEIGHTS_URL, weights_dir="weights", workers=None, log=None):
    """Plan, download, mosaic and map the study area with the region's models -> {model: map file}."""
    import rasterio
    from . import download, stack, workflow
    from ..models import load_model
    project = project if isinstance(project, Project) else Project(project)
    log = log or project.log
    region = region_of(project)
    t0 = time.time()
    with run_lock(project, f"pretrained {region}"):
        log(f"== Step 1/4: planning the study area ({label(region)}: {', '.join(model_files(region))}) ==")
        plan(project)
        log("== Step 2/4: downloading the inputs ==")
        layers = input_layers(INPUT) + list(project["benchmarks"] or [])
        res = download.run_download(project, workers=workers, layers=layers)
        if res["failed"]:
            raise RuntimeError(f"{len(res['failed'])} downloads failed, e.g. {res['failed'][0]}: run again to retry")
        log("== Step 3/4: building the study-area mosaic ==")
        stack.build_mosaics(project)
        log("== Step 4/4: mapping canopy height ==")
        annual = project.mosaic("annual")
        with rasterio.open(annual) as s:
            profile = s.profile
        aoi = workflow.read_aoi_mask(project, like=profile)
        inside = aoi & region_mask(region, profile, project.root)
        if not inside.any():
            raise ValueError(f"no pixel of the study area lies in the {region} region")
        if inside.sum() < aoi.sum():
            log(f"note: {1 - inside.sum() / aoi.sum():.0%} of the study area is outside the {region} region and is "
                f"left empty")
        dev = workflow.torch_device(project)
        out = {}
        for name, rel in model_files(region).items():
            path = weights.fetch(rel, weights_source, weights_dir, log=log)
            model, cfg = load_model(path, device=dev)
            pred = workflow.predict_unet_map(model, cfg, annual, None, inside, dev)
            del model
            dst = project.path("maps", f"{name}.tif")
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(f"_{dst.name}.part")
            pred = np.array(pred, dtype=np.float32)
            pred[~inside] = np.nan
            with rasterio.open(tmp, "w", **workflow.map_profile(profile)) as d:
                d.write(pred, 1)
                d.set_band_description(1, "canopy height (m)")
                d.update_tags(model=name, region=region, input=INPUT)
            replace_retry(tmp, dst)
            v = pred[np.isfinite(pred)]
            log(f"{name}: {dst.relative_to(project.root)}"
                + (f" (mean {v.mean():.1f} m, median {np.median(v):.1f} m)" if v.size else " (no valid pixel)"))
            out[name] = dst
        log(f"done in {math.ceil(time.time() - t0)} s")
    project.close_log()
    return out


def region_map(region, aoi=None, cache_dir=None, height=450):
    """folium map of the region (green) and, if given, the study area (red)."""
    import folium
    from shapely.geometry import mapping
    reg = region_geometry(region, cache_dir)
    focus = grid.read_aoi(aoi) if aoi is not None else reg
    w, s, e, n = focus.bounds
    m = folium.Map(tiles="OpenStreetMap", height=height)
    folium.GeoJson(mapping(reg), name=f"{region} region",
                   style_function=lambda _: dict(color="#2e7d32", weight=2, fillOpacity=0.15)).add_to(m)
    if aoi is not None:
        folium.GeoJson(mapping(focus), name="study area",
                       style_function=lambda _: dict(color="#c62828", weight=2, fillOpacity=0.05)).add_to(m)
    m.fit_bounds([[s, w], [n, e]])
    folium.LayerControl().add_to(m)
    return m
