"""Study area and the 2.56 km cells of the analysis grid (plan/grid.json, plan/cells.csv).

The study area is read from any common vector format (geometry only), dissolved and stored in EPSG:4326
(aoi.geojson); a study area across the 180th meridian must be split into separate projects. The analysis grid is
10 m in the UTM zone of the study area's centroid, anchored at the upper-left corner of its bounding box, and cut
into 256 x 256 px cells (one cell = one chip of the training stack and of the study-area mosaic).
The anchor is snapped with a 0.1 m tolerance, so a study area drawn on a pipeline grid (e.g. anchor
584670 / 550110 in EPSG:32650) survives the EPSG:4326 round trip of aoi.geojson unchanged
and the tool's cells coincide with the pipeline's.

Training region (the paper's international sites, `train_cover400`, generalised): the cells of the study area
plus the ring of cells whose boundary distance to it is <= D, D the smallest whole km at which the usable cells
(study area + ring cells with < water_max WorldCover-2020 water) reach `train_cells` (~400 patches in the
paper). Land cover is looked up only for cells that can still be selected and cached in plan/landcover.csv.
A study area of more than `train_cells` cells is trained on all of its cells (logged with the cost).
Cells of the paper pipeline can be imported instead (`import_cells`), in the pipeline's row order, which
defines the training / validation split and so the reproduction of the paper's models.
"""
import json
import locale
import math
import tempfile
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pyproj import CRS, Proj, Transformer
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry

from . import project as project_mod
from .project import CELL_M, CELL_PX, RES, replace_retry

AOI_SUFFIXES = (".shp", ".geojson", ".json", ".gpkg", ".kml")      # readable directly (and inside a .zip)
GEOJSON_TYPES = {"FeatureCollection", "Feature", "Polygon", "MultiPolygon", "GeometryCollection", "Point",
                 "MultiPoint", "LineString", "MultiLineString"}
SNAP = 0.01            # grid-origin tolerance in pixels (0.1 m)
MIN_AREA = 1.0         # m^2 of overlap for a cell to belong to the study area
RING_STEP = 1000.0     # the ring grows in whole km
FEW_PIXELS = 100       # study areas with fewer 10 m pixels are logged as not meaningful
MAX_SCALE_ERROR = 0.02     # analysis CRS: largest deviation of its scale from 1 in the study area (error above)
WARN_SCALE_ERROR = 0.005   # (warning above)
PAPER_CELLS = 400                                   # training cells of the paper's regions
STACK_BYTES_PER_CELL = (76 * 2 + 4) * CELL_PX ** 2  # uint16 inputs + float32 GEDI labels
RF_BYTES_PER_CELL = 9.6e6                           # RF-SLS forest at the paper's GEDI density (4.2 GB / 439 cells)
CELL_COLS = ["cell", "x0", "y1", "col", "row", "role", "dist_m", "water", "built", "tree", "use"]
IMPORT_COLS = ["src_dir", "lab_dir", "epsg"]
ROW_COLS = ["piece", "tile", "kind", "min_dist_m", "epsg", "tile_x0", "tile_y1", "padded", "x_src", "lab_src"]


# ================================================================================================ study area
def read_aoi(src):
    """Study area as one shapely (Multi)Polygon in EPSG:4326.

    src: path (.shp, .zip holding a shapefile / GeoJSON / GeoPackage / KML, .geojson, .json, .gpkg, .kml), a
    GeoJSON dict (geometry, Feature or FeatureCollection; a legacy "crs" member is honoured) or a shapely
    geometry in EPSG:4326. All features of all layers are dissolved; only polygonal parts are kept. Data with
    another CRS are reprojected; data without a CRS must have lon/lat coordinates. Attributes are never read.
    A study area across the 180th meridian is refused (split it into separate projects).
    """
    parts = []
    for geoms, crs in _read_parts(src):
        geoms = [g for g in geoms if g is not None and not g.is_empty]
        if geoms:
            parts.extend(_to_4326(geoms, crs, src))
    geom = _dissolve(parts, src)
    _check_lon_span(geom, src)
    return geom


def write_aoi(src, dst):
    """Study area (anything `read_aoi` accepts) -> GeoJSON FeatureCollection with one feature, EPSG:4326."""
    geom = read_aoi(src)
    fc = {"type": "FeatureCollection",
          "features": [{"type": "Feature", "properties": {}, "geometry": mapping(geom)}]}
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(fc, fh)
    replace_retry(tmp, dst)
    return geom


def _read_parts(src):
    """-> [(list of shapely geometries, CRS or None), ...]"""
    if isinstance(src, BaseGeometry):
        return [([src], None)]
    if isinstance(src, dict):
        return [_from_geojson(src)]
    p = Path(src)
    if not p.exists():
        raise FileNotFoundError(f"study-area file not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".zip":
        with tempfile.TemporaryDirectory(prefix="chm_aoi_") as td:
            with zipfile.ZipFile(p) as z:
                z.extractall(td)
            found = sorted(f for f in Path(td).rglob("*") if f.suffix.lower() in AOI_SUFFIXES
                           and not f.name.startswith("._") and "__MACOSX" not in f.parts)
            shp = [f for f in found if f.suffix.lower() == ".shp"]
            if not (shp or found):
                raise ValueError(f"{p.name}: no shapefile / GeoJSON / GeoPackage / KML inside the zip")
            return [part for f in (shp or found) for part in _read_parts(f)]
    if suffix in (".geojson", ".json"):
        d = _load_json(p)
        if isinstance(d, dict) and d.get("type") in GEOJSON_TYPES:
            return [_from_geojson(d)]
    return _read_file(p)


def _load_json(p):
    """JSON file (UTF-8, else the system encoding, e.g. GBK); None if it cannot be decoded as JSON."""
    raw = Path(p).read_bytes()
    for enc in ("utf-8-sig", locale.getpreferredencoding(False)):
        try:
            return json.loads(raw.decode(enc))
        except (ValueError, LookupError):          # UnicodeDecodeError / JSONDecodeError / unknown codec
            continue
    return None


def _read_file(p):
    """Geometries and CRS of every layer (attributes are not read, so their encoding does not matter)."""
    import geopandas as gpd
    import pyogrio
    out = []
    for name, _ in pyogrio.list_layers(p):
        df = gpd.read_file(p, layer=name, engine="pyogrio", columns=[])
        if isinstance(df, gpd.GeoDataFrame) and df.geometry is not None:
            out.append((list(df.geometry), df.crs))
    return out


def _from_geojson(d):
    t = d.get("type")
    if t == "FeatureCollection":
        geoms = [shape(f["geometry"]) for f in d.get("features", []) if f.get("geometry")]
    elif t == "Feature":
        geoms = [shape(d["geometry"])] if d.get("geometry") else []
    elif t in GEOJSON_TYPES:
        geoms = [shape(d)]
    else:
        raise ValueError(f"not a GeoJSON object (type {t!r})")
    crs = None
    name = ((d.get("crs") or {}).get("properties") or {}).get("name")
    if name:
        crs = CRS.from_user_input(name)
    return geoms, crs


def _to_4326(geoms, crs, src):
    if crs is None:
        minx, miny, maxx, maxy = shapely.total_bounds(geoms)
        if not (-180 <= minx <= maxx <= 180 and -90 <= miny <= maxy <= 90):
            wrapped = -360 <= minx <= maxx <= 360 and -90 <= miny <= maxy <= 90
            raise ValueError(f"{_label(src)}: no coordinate reference system and the coordinates are not "
                             "longitude / latitude (add the .prj of the shapefile or save the file in EPSG:4326)"
                             + ("; if they are longitudes beyond +-180, the study area crosses the 180th meridian: "
                                "split it into the parts east and west of it and run them as separate projects"
                                if wrapped else ""))
        return geoms
    crs = CRS.from_user_input(crs)
    if crs == CRS.from_epsg(4326):
        return geoms
    tr = Transformer.from_crs(crs, 4326, always_xy=True)
    return [_transform(g, tr) for g in geoms]


def _dissolve(geoms, src):
    polys = [p for g in geoms for p in _polygons(shapely.force_2d(shapely.make_valid(g)))]
    u = shapely.union_all(polys) if polys else Polygon()
    if not u.is_valid:
        u = u.buffer(0)
    polys = _polygons(u)
    if not polys or sum(p.area for p in polys) == 0:
        raise ValueError(f"{_label(src)}: no polygon (the study area must be a polygon, not points or lines)")
    return polys[0] if len(polys) == 1 else MultiPolygon(polys)


def _polygons(g):
    if isinstance(g, Polygon):
        return [] if g.is_empty else [g]
    if isinstance(g, (MultiPolygon, GeometryCollection)):
        return [p for sub in g.geoms for p in _polygons(sub)]
    return []


def _label(src):
    return Path(src).name if isinstance(src, (str, Path)) else "study area"


def _check_lon_span(geom4326, src=None):
    """Refuse a lon/lat geometry spanning more than 180 degrees of longitude (it crosses the 180th meridian)."""
    minx, _, maxx, _ = geom4326.bounds
    if maxx - minx > 180:
        raise ValueError(f"{_label(src)}: the study area crosses the 180th meridian (longitudes {minx:.2f} to "
                         f"{maxx:.2f}): split it into the parts east and west of it and run them as separate projects")


# ================================================================================================ CRS / grid
def _centroid(geom4326):
    """(lon, lat) of the centroid of a lon/lat geometry, computed in a local equal-area projection."""
    minx, miny, maxx, maxy = geom4326.bounds
    laea = CRS.from_proj4(f"+proj=laea +lat_0={(miny + maxy) / 2} +lon_0={(minx + maxx) / 2} +datum=WGS84 +units=m")
    c = _transform(geom4326, Transformer.from_crs(4326, laea, always_xy=True)).centroid
    return Transformer.from_crs(laea, 4326, always_xy=True).transform(c.x, c.y)


def utm_epsg(geom4326):
    """EPSG code of the UTM zone of the study area's centroid: 326zz (north) or 327zz (south)."""
    _check_lon_span(geom4326)
    lon, lat = _centroid(geom4326)
    zone = min(max(int(math.floor((lon + 180.0) / 6.0)) + 1, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def scale_error(geom4326, epsg):
    """Largest deviation from 1 of the scale of EPSG:epsg at the centroid and bounding-box corners of a lon/lat
    geometry (< 0.001 inside a UTM zone, 0.41 for Web Mercator at 45 degrees; inf outside the CRS's domain)."""
    minx, miny, maxx, maxy = geom4326.bounds
    lon, lat = _centroid(geom4326)
    f = Proj(CRS.from_epsg(int(epsg))).get_factors([lon, minx, minx, maxx, maxx], [lat, miny, maxy, miny, maxy])
    k = np.concatenate([np.atleast_1d(f.meridional_scale), np.atleast_1d(f.parallel_scale)]).astype(float)
    return float(np.max(np.where(np.isfinite(k), np.abs(k - 1), np.inf)))


@lru_cache(maxsize=16)
def _transformer(src_epsg, dst_epsg):
    return Transformer.from_crs(int(src_epsg), int(dst_epsg), always_xy=True)


def _transform(geom, tr):
    def f(xy):
        x, y = tr.transform(xy[:, 0], xy[:, 1])
        out = np.column_stack([x, y])
        if not np.isfinite(out).all():
            raise ValueError("study area outside the domain of the target CRS")
        return out
    return shapely.transform(geom, f)


def to_crs(geom4326, epsg, src_epsg=4326):
    """Reproject a shapely geometry from EPSG:4326 (or src_epsg) to EPSG:epsg (x = easting / longitude)."""
    if int(epsg) == int(src_epsg):
        return geom4326
    return _transform(geom4326, _transformer(src_epsg, epsg))


def make_grid(aoi_xy, epsg):
    """Analysis grid of a study area given in EPSG:epsg: upper-left corner of its bounding box snapped to 10 m."""
    minx, _, _, maxy = aoi_xy.bounds
    return {"epsg": int(epsg), "x0": float(math.floor(minx / RES + SNAP) * RES),
            "y1": float(math.ceil(maxy / RES - SNAP) * RES), "res": RES, "cell_px": CELL_PX}


def _cell_m(grid):
    return float(grid["res"]) * int(grid["cell_px"])


def cell_box(grid, col, row):
    """Footprint (shapely box, grid CRS) of cell (col, row); col / row may be negative."""
    m = _cell_m(grid)
    x = grid["x0"] + col * m
    y = grid["y1"] - row * m
    return shapely.box(x, y - m, x + m, y)


def cell_name(x0, y1):
    """Cell name from its upper-left corner, e.g. x584670y550110."""
    return f"x{round(x0)}y{round(y1)}"


def aoi_pixels(aoi_xy, cells, need=None):
    """Number of 10 m pixel centres of the cells (columns x0, y1) inside the study area aoi_xy (grid CRS), as
    mosaic/aoi_mask.tif counts them; counting stops once `need` is reached."""
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    geom, n = mapping(aoi_xy), 0
    for x0, y1 in zip(cells["x0"], cells["y1"]):
        t = from_origin(float(x0), float(y1), RES, RES)
        n += int(rasterize([(geom, 1)], out_shape=(CELL_PX, CELL_PX), transform=t, fill=0, all_touched=False,
                           dtype="uint8").sum())
        if need is not None and n >= need:
            break
    return n


def training_cost(n_cells):
    """Approximate cost of training on n_cells cells: training stack and RF-SLS forest (GB) and UNet epoch time
    relative to the paper's ~400-cell regions."""
    return dict(stack_gb=n_cells * STACK_BYTES_PER_CELL / 1e9, rf_gb=n_cells * RF_BYTES_PER_CELL / 1e9,
                unet_x=n_cells / PAPER_CELLS)


def _evaluate(aoi_xy, grid, D, done):
    """Distance and study-area overlap of the cells that may lie within D of the study area, outside the
    (col0, col1, row0, row1) window `done` already evaluated -> (DataFrame, window)."""
    m = _cell_m(grid)
    minx, miny, maxx, maxy = aoi_xy.bounds
    x0, y1 = grid["x0"], grid["y1"]
    win = (math.floor((minx - D - x0) / m) - 1, math.floor((maxx + D - x0) / m) + 1,
           math.floor((y1 - maxy - D) / m) - 1, math.floor((y1 - miny + D) / m) + 1)
    cols, rows = np.meshgrid(np.arange(win[0], win[1] + 1), np.arange(win[2], win[3] + 1))
    cols, rows = cols.ravel(), rows.ravel()
    if done is not None:
        new = ~((cols >= done[0]) & (cols <= done[1]) & (rows >= done[2]) & (rows <= done[3]))
        cols, rows = cols[new], rows[new]
    bx, by = x0 + cols * m, y1 - rows * m
    boxes = shapely.box(bx, by - m, bx + m, by)
    raw = shapely.distance(boxes, aoi_xy)
    area = np.zeros(len(boxes))
    hit = raw == 0
    if hit.any():
        area[hit] = shapely.area(shapely.intersection(boxes[hit], aoi_xy))
    df = pd.DataFrame({"cell": [cell_name(x, y) for x, y in zip(bx, by)], "x0": bx.astype(float),
                       "y1": by.astype(float), "col": cols.astype(int), "row": rows.astype(int),
                       "dist_m": np.rint(raw).astype(int), "area": area})
    return df, win


# ================================================================================================ planning
def _worldcover(project):
    """Default land-cover lookup: WorldCover-2020 shares from Earth Engine (initialised on first use)."""
    ready = []

    def lookup(cells, epsg):
        from .gee import io as gee_io
        from .gee.layers import worldcover_shares
        if not ready:
            gee_io.ee_init(project["gee_project"])
            ready.append(True)
        return worldcover_shares(cells, epsg, cell_m=CELL_M)
    return lookup


def _write_csv(df, f):
    f = Path(f)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.name + ".tmp")
    df.to_csv(tmp, index=False)
    replace_retry(tmp, f)


def _write_json(d, f):
    f = Path(f)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(d) + "\n")
    replace_retry(tmp, f)


def plan_cells(project, landcover=None):
    """Grid and training cells of a project -> plan/grid.json, plan/cells.csv; returns the cells.

    landcover(list of (x0, y1), epsg) -> list of (water, built, tree); default: WorldCover-2020 shares from
    Earth Engine. Looked up for the study-area cells (for information) and for the ring cells within the
    current D only; cached in plan/landcover.csv. The chosen ring distance is returned in `.attrs["ring_m"]`
    (None: the study area alone has `train_cells` cells, no ring).
    Fails (before any Earth Engine request) on invalid settings, an analysis CRS whose scale in the study area
    is off by more than MAX_SCALE_ERROR, and a study area that covers no 10 m pixel centre.
    """
    project_mod.validate(project.cfg)
    aoi = read_aoi(project.aoi_file)
    epsg = int(project["epsg"] or utm_epsg(aoi))
    err = scale_error(aoi, epsg)
    if err > MAX_SCALE_ERROR:
        raise ValueError(f"EPSG:{epsg} distorts distances by {err:.1%} in the study area, so a 10 m pixel would not "
                         f"be 10 m on the ground: leave epsg empty (UTM zone of the study area) or use a local "
                         f"projection")
    if err > WARN_SCALE_ERROR:
        project.log(f"plan: WARNING EPSG:{epsg} distorts distances by up to {err:.1%} in the study area")
    aoi_xy = to_crs(aoi, epsg)
    grid = make_grid(aoi_xy, epsg)
    target, wmax = int(project["train_cells"]), float(project["water_max"])
    cap = max(float(project["ring_max_km"]) * 1000.0, 0.0)

    lc_file = project.path("plan", "landcover.csv", mkdir=True)
    cache = (pd.read_csv(lc_file, dtype={"cell": str}).drop_duplicates("cell", keep="last").set_index("cell")
             if lc_file.exists() else pd.DataFrame(columns=["water", "built", "tree"], dtype=float))
    lookup = landcover or _worldcover(project)

    def fill(cells):
        nonlocal cache
        miss = cells[~cells.cell.isin(cache.index)]
        if len(miss):
            vals = lookup([(float(x), float(y)) for x, y in zip(miss.x0, miss.y1)], epsg)
            new = pd.DataFrame([tuple(map(float, v)) for v in vals], index=pd.Index(miss.cell, name="cell"),
                               columns=["water", "built", "tree"])
            cache = pd.concat([cache, new]) if len(cache) else new
            _write_csv(cache.rename_axis("cell").reset_index(), lc_file)

    cells, win = _evaluate(aoi_xy, grid, 0.0, None)
    aoi_cells = cells[cells.area >= MIN_AREA].sort_values(["y1", "x0"], ascending=[False, True])
    if aoi_cells.empty:
        raise ValueError("the study area does not cover 1 m^2 of any grid cell")
    n_px = aoi_pixels(aoi_xy, aoi_cells, need=FEW_PIXELS)
    if n_px == 0:
        raise ValueError("the study area covers no 10 m pixel centre (it is narrower than a 10 m pixel): every map "
                         "would be empty; draw a larger study area")
    if n_px < FEW_PIXELS:
        project.log(f"plan: WARNING the study area covers only {n_px} pixels of 10 m: results for such a small area "
                    f"are not meaningful")
    fill(aoi_cells)
    n_aoi = len(aoi_cells)
    if n_aoi > target:
        c = training_cost(n_aoi)
        project.log(f"plan: WARNING the study area alone has {n_aoi} cells, more than train_cells {target}: all "
                    f"{n_aoi} are training cells (no ring) -> training stack ~{c['stack_gb']:.0f} GB, RF-SLS forest "
                    f"~{c['rf_gb']:.0f} GB of memory and disk, UNet epochs ~{c['unet_x']:.1f} x as long as for the "
                    f"paper's ~{PAPER_CELLS}-cell regions; split the study area into several projects to stay near "
                    f"the paper's scale")
    ring, D, usable = cells.iloc[:0], None, n_aoi
    if n_aoi < target:
        D, reach = 0.0, 0.0
        while True:
            if D > reach:
                more, win = _evaluate(aoi_xy, grid, D, win)
                if len(more):
                    cells = pd.concat([cells, more], ignore_index=True)
                reach = D
            ring = cells[(cells.area < MIN_AREA) & (cells.dist_m <= D)]
            fill(ring)
            usable = n_aoi + int((~(cache.loc[ring.cell, "water"].to_numpy() >= wmax)).sum())
            if usable >= target or D >= cap:
                break
            D = min(D + RING_STEP, cap)
        ring = ring.assign(_ny=-ring.y1).sort_values(["dist_m", "_ny", "x0"]).drop(columns="_ny")

    out = pd.concat([aoi_cells.assign(role="aoi", dist_m=0)] + ([ring.assign(role="ring")] if len(ring) else []),
                    ignore_index=True)
    lc = cache.loc[out.cell]
    for k in ("water", "built", "tree"):
        out[k] = lc[k].to_numpy(dtype=float)
    out["use"] = (out.role == "aoi") | ~(out.water >= wmax)
    out = out[CELL_COLS]
    _write_json(grid, project.grid_file)
    _write_csv(out, project.cells_file)
    out.attrs["ring_m"] = None if D is None else int(D)

    n_ring = int((out.role == "ring").sum())
    n_water = int((~out.use).sum())
    msg = (f"plan: EPSG:{epsg}, origin ({grid['x0']:.0f}, {grid['y1']:.0f}); {n_aoi} aoi cells + "
           + (f"{n_ring} ring cells <= {D / 1000:g} km ({n_water} dropped: water >= {wmax:g})" if D is not None
              else "no ring (aoi cells >= train_cells)")
           + f" -> {int(out.use.sum())} training cells (train_cells {target})")
    if usable < target:
        msg += f"; ring_max_km {project['ring_max_km']:g} reached with fewer cells than train_cells"
    project.log(msg)
    return out


# ================================================================================================ import
def import_cells(project, rows, role_map={"test": "aoi"}, default_role="ring"):
    """cells.csv from a table of existing 256 x 256 rasters `<src_dir>/<cell>_<layer>.tif`, e.g. the paper
    pipeline's plans\\rows_<site>.csv (columns piece, tile, kind, min_dist_m, epsg, tile_x0, tile_y1,
    padded, x_src, lab_src). Row order is kept (it defines the training split). role = role_map[kind], else
    default_role. plan/grid.json gets the common grid of the cells, or "cells_on_common_grid": false. On a common
    grid, the project's study area (aoi.geojson) must cover a pixel of the imported aoi cells (their maps).
    """
    t = rows.copy() if isinstance(rows, pd.DataFrame) else pd.read_csv(rows)
    name = "table" if isinstance(rows, pd.DataFrame) else Path(rows).name
    miss = [c for c in ROW_COLS if c not in t.columns]
    if miss:
        raise ValueError(f"{name}: missing columns {miss}")
    t = t.reset_index(drop=True)
    padded = t["padded"].astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])
    bad = (pd.to_numeric(t["tile"]) != 1) | padded
    if bad.any():
        raise ValueError(f"{name}: {int(bad.sum())} rows are not whole 256 x 256 pieces (tile != 1 or padded), "
                         f"e.g. {t.loc[bad, 'piece'].iloc[0]} tile {t.loc[bad, 'tile'].iloc[0]}")
    dup = t["piece"].astype(str).duplicated()
    if dup.any():
        raise ValueError(f"{name}: duplicated pieces, e.g. {t.loc[dup, 'piece'].iloc[0]}")
    roles = [role_map.get(k, default_role) for k in t["kind"].astype(str)]
    if set(roles) - {"aoi", "ring"}:
        raise ValueError(f"roles must be 'aoi' or 'ring', got {sorted(set(roles) - {'aoi', 'ring'})}")
    dist = pd.to_numeric(t["min_dist_m"]).round()
    x0, y1 = t["tile_x0"].astype(float), t["tile_y1"].astype(float)
    epsg = t["epsg"].astype(int)
    out = pd.DataFrame({"cell": t["piece"].astype(str), "x0": x0, "y1": y1, "col": np.nan, "row": np.nan,
                        "role": roles, "dist_m": dist.astype(int) if dist.notna().all() else dist,
                        "water": np.nan, "built": np.nan, "tree": np.nan, "use": True,
                        "src_dir": t["x_src"].astype(str), "lab_dir": t["lab_src"].astype(str), "epsg": epsg})
    out = out[CELL_COLS + IMPORT_COLS]

    def on_grid(d):
        return bool(np.all(np.abs(d - np.round(d / CELL_M) * CELL_M) <= SNAP * RES))
    one_epsg = epsg.nunique() == 1
    common = one_epsg and on_grid(x0 - x0.min()) and on_grid(y1.max() - y1)
    grid = {"epsg": int(epsg.iloc[0]) if one_epsg else None, "x0": float(x0.min()) if common else None,
            "y1": float(y1.max()) if common else None, "res": RES, "cell_px": CELL_PX,
            "cells_on_common_grid": common}
    n_aoi = int((out.role == "aoi").sum())
    if common and n_aoi and project.aoi_file.exists():
        aoi_xy = to_crs(read_aoi(project.aoi_file), grid["epsg"])
        if aoi_pixels(aoi_xy, out[out.role == "aoi"], need=1) == 0:
            raise ValueError(f"{name}: the project's study area ({project.aoi_file.name}) covers no pixel of the "
                             f"{n_aoi} imported study-area cells, so their maps would be empty: create the project "
                             f"with the study area of these cells")
    _write_json(grid, project.grid_file)
    _write_csv(out, project.cells_file)
    project.log(f"import-cells: {len(out)} cells from {name} ({n_aoi} aoi, {len(out) - n_aoi} ring), "
                f"EPSG:{grid['epsg'] if one_epsg else 'mixed'}, "
                + ("on one common grid" if common else "not on one common grid (cells_on_common_grid false)"))
    return out
