"""Download stage: every layer of every planned cell from Earth Engine onto the analysis grid, with the requests
and file formats used for the paper's data.

Per used cell of plan/cells.csv (use == True): the input layers of project['input'] (project.input_layers: annual
Embedding / DEM / S1 / S2 as the representation needs them, the seasonal S1_asc_0..3 / S2_0..3 for T and TE) and
the GEDI label layer (median rh95 over project['gedi_window'], WorldCover-2020 built-up cells masked if
project['built_up_mask']) with DEM; per study-area cell (role aoi) also the benchmarks of project['benchmarks']:
HRCH -> ETH, GFCH -> UMD, GMTCH -> Tolan_1m (1 m), then GMTCH = p90 of the 10 x 10 Tolan pixels of every 10 m cell,
computed locally. Layers the representation does not use are not downloaded.

File formats (imported cells are simply skipped): float64 with NaN for Embedding / S1 / S2 / GEDI and the seasonal
layers (S1_asc_k: VV, VH in dB; S2_k: B2 ... B12 as 0-1 reflectance), DEM int16, ETH / UMD uint8 (0 = no data), Tolan_1m uint8 (255 = no
data) on the 1 m grid nested in the cell, GMTCH float32 (NaN). One job = one layer, except DEM + GEDI (one request);
Sentinel-1 depends on project['s1_method']: 'local' (default) fetches the raw scenes of the cell (gee.s1_raw) and
runs the chain in numpy (s1_local), with the acquisition metadata requested once for all cells (s1_regions, file
raw/s1_local_<year>[_seasonal].json) and the four S1_asc_k of a cell as one job; 'gee' runs the chain on Earth
Engine: S1 first fetches the per-cell acquisition metadata, the four S1_asc_k jobs of a cell share one metadata
request of the seasonal window (S1MetaCache). S1 rasters carry the tag s1_method. A season without any
acquisition / image is written as NaN (no data). A job
writes <file>.partial.tif and renames it when complete, so an interrupted run resumes by skipping the files that
exist. Existing downloads made with another year / GEDI window / built-up mask than project.yaml (raster tags) stop
the stage (SettingsChanged).
"""
import math
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np

from .project import (BENCHMARKS, CELL_PX, LABEL_LAYER, RES, SEASONAL_LAYERS, X_LAYERS, input_layers,
                      replace_retry)

INT_LAYERS = {"DEM": ("int16", 0), "ETH": ("uint8", 0), "UMD": ("uint8", 0), "Tolan_1m": ("uint8", 255)}
S1_SEASONAL = tuple(l for l in SEASONAL_LAYERS if l.startswith("S1_asc_"))
S2_SEASONAL = tuple(l for l in SEASONAL_LAYERS if l.startswith("S2_"))
SEASONS = ("DJF", "MAM", "JJA", "SON")              # season k of S1_asc_k / S2_k (months 12-2, 3-5, 6-8, 9-11)
SEASONAL_BANDS = {"S1": ["VV", "VH"], "S2": ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12"]}
CHUNK = {"Embedding": 256, "S1": 256, "S2": 512, "DEM": 1024, "ETH": 1024, "UMD": 1024, "GEDI": 1024,
         "Tolan_1m": 2560,                          # getDownloadURL chunk (px); split 2 x 2 on memory errors
         **{l: 512 for l in S1_SEASONAL}, **{l: 768 for l in S2_SEASONAL}}
BUNDLE = ("DEM", LABEL_LAYER)                       # downloaded together in one request
CHECK_EVERY = 50                                    # progress line every so many cells while checking existing files
FILE_LAYERS = X_LAYERS + SEASONAL_LAYERS + [LABEL_LAYER, "ETH", "UMD", "Tolan_1m", "GMTCH"]
YEAR_LAYERS = ("Embedding", "S1", "S2", *SEASONAL_LAYERS)          # content depends on project['year']
# rough cost of one 256 x 256 cell per job (MB on disk, Earth Engine EECU-seconds). Sentinel-1 by the Earth Engine
# way (s1_method: gee) dominates the compute: billed 139 EECU-s for a cell with 31 acquisitions / year, growing
# with the number of acquisitions (whole-project averages of the paper's exports: 650-2900 per cell). Seasonal S1: every acquisition of the window is processed once,
# in its season, so the four seasons together cost about the annual S1; seasonal S2 is stored as 0-1 floats (less
# compressible than the annual DN): 3.4-3.5 MB per season
COST = {"Embedding": (25.6, 5), "S1": (0.9, 150), "S2": (0.8, 60), "DEM": (0.01, 1), "GEDI": (0.02, 5),
        "ETH": (0.02, 1), "UMD": (0.01, 1), "Tolan_1m": (1.2, 5),
        **{l: (0.9, 40) for l in S1_SEASONAL}, **{l: (3.4, 20) for l in S2_SEASONAL}}
# EECU-s range per cell (Earth Engine way): S1 100-3000, a quarter of that per season; others 0.5-2 x COST
EECU_RANGE = {"S1": (100, 3000), **{l: (25, 750) for l in S1_SEASONAL}}
# Sentinel-1 by the local way (s1_method: local, the default; gee.s1_raw + s1_local): Earth Engine only serves the
# raw scenes. Billed 10.4 EECU-s per cell (annual, 31 acquisitions) and 10.9 for the four seasons together,
# with the acquisition metadata requested once for all cells; about 48 MB of raw scenes pass through per cell (not
# stored). With 109-173 acquisitions per cell: ~44 EECU-s and 160-260 MB per cell (Earth Engine way ~450).
COST_S1_LOCAL = {"S1": (0.9, 10), **{l: (0.9, 3) for l in S1_SEASONAL}}
EECU_RANGE_S1_LOCAL = {"S1": (6, 60), **{l: (1.5, 15) for l in S1_SEASONAL}}
TRANSFER_MB = {"S1": 50, **{l: 12.5 for l in S1_SEASONAL}}
def is_s1(layer):
    """Annual or seasonal Sentinel-1 layer."""
    return layer == "S1" or layer in S1_SEASONAL


def layer_cost(layer, s1_method="local"):
    """(MB on disk, typical EECU-s, (low, high) EECU-s) of one layer of one cell."""
    if is_s1(layer) and s1_method == "local":
        return COST_S1_LOCAL[layer][0], COST_S1_LOCAL[layer][1], EECU_RANGE_S1_LOCAL[layer]
    mb, s = COST[layer]
    return mb, s, EECU_RANGE.get(layer, (0.5 * s, 2 * s))


class SettingsChanged(RuntimeError):
    """Existing downloads were made with another year / GEDI window / built-up mask than project.yaml."""


# ================================================================================================ plan
def load_plan(project):
    """(grid dict, used cells of plan/cells.csv in stack order)."""
    import json
    import pandas as pd
    if not project.cells_file.exists() or not project.grid_file.exists():
        raise FileNotFoundError(f"no plan in {project.root}: run the plan stage first")
    with open(project.grid_file, encoding="utf-8") as fh:
        grid = json.load(fh)
    P = pd.read_csv(project.cells_file)
    use = P["use"].astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes"])
    return grid, P[use].reset_index(drop=True)


def cell_geo(row, grid):
    """(epsg, x0, y1) of a cell: its own epsg column (imported cells) or the grid's."""
    e = row.get("epsg")
    epsg = int(e) if e is not None and e == e and str(e).strip() else int(grid["epsg"])
    return epsg, float(row["x0"]), float(row["y1"])


def _cell_inputs(inp):
    return list(dict.fromkeys(input_layers(inp) + ["DEM", LABEL_LAYER]))


def cell_layers(project, row):
    """File layers a cell needs: the inputs of project['input'] (project.input_layers) + DEM + GEDI for every used
    cell (DEM comes with the GEDI request), benchmark layers for aoi cells."""
    lay = _cell_inputs(project["input"])
    if row["role"] == "aoi":
        lay += [BENCHMARKS[b] for b in project["benchmarks"]]
    return lay


def _wanted(layers):
    if not layers:
        return None
    alias = {k: v for k, v in BENCHMARKS.items() if k != "GMTCH"}
    out = {alias.get(l, l) for l in layers}
    bad = out - set(FILE_LAYERS)
    if bad:
        raise ValueError(f"unknown layers {sorted(bad)}; known: {FILE_LAYERS}")
    return out


def gedi_tags(gedi_window, built_up_mask):
    """Label settings as stored in the GEDI raster tags."""
    gw = [str(d) for d in gedi_window]
    return dict(gedi_start=gw[0], gedi_end=gw[1], built_up_mask=str(bool(built_up_mask)))


def settings_mismatch(project, row, layer):
    """Why the existing raw/ raster of a layer was downloaded with other settings than project.yaml, else None.

    Compares the tags written by export: year (YEAR_LAYERS) and gedi_start / gedi_end / built_up_mask (GEDI).
    Tags that are missing or empty are accepted, and so are an imported cell's own rasters (its table defines
    them).
    """
    if layer not in YEAR_LAYERS and layer != LABEL_LAYER:
        return None
    f = project.cell_raster(row, layer)
    if f != project.raw(row["cell"], layer) or not f.exists():
        return None
    import rasterio
    with rasterio.open(f) as s:
        tags = s.tags()
    want = ({"year": str(project["year"])} if layer in YEAR_LAYERS
            else gedi_tags(project["gedi_window"], project["built_up_mask"]))
    diff = [f"{k} {tags[k].strip()} (project: {v})" for k, v in want.items()
            if tags.get(k, "").strip() not in ("", v)]
    return ", ".join(diff) or None


def jobs_for(project, row, wanted=None, stale=None):
    """Missing layers of a cell -> list of download bundles ([layer], [DEM, GEDI], and with s1_method local the
    cell's missing S1_asc_k together: one fetch of the raw scenes serves the four seasons).

    stale: a list that collects {'cell', 'layer', 'file', 'reason'} of the cell's existing downloads made with
    other settings (settings_mismatch, all layers of the cell whatever `wanted`); run_download stops on them.
    """
    def want(l):
        return not wanted or l in wanted

    def have(l):
        return project.cell_raster(row, l).exists()
    if stale is not None:
        import rasterio
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):         # no listing of raw/ per file
            for l in cell_layers(project, row):
                why = settings_mismatch(project, row, l)
                if why:
                    stale.append(dict(cell=row["cell"], layer=l, file=str(project.raw(row["cell"], l)), reason=why))
    todo = []
    for l in cell_layers(project, row):
        if l == "GMTCH":                              # derived from Tolan_1m, downloaded only when needed
            if not have("Tolan_1m") and ((want("GMTCH") and not have("GMTCH")) or (wanted and "Tolan_1m" in wanted)):
                todo.append("Tolan_1m")
        elif want(l) and not have(l):
            todo.append(l)
    bundle = [l for l in todo if l in BUNDLE]
    s1s = [l for l in todo if l in S1_SEASONAL] if project["s1_method"] == "local" else []
    rest = [[l] for l in todo if l not in bundle and l not in s1s]
    return rest + ([s1s] if s1s else []) + ([bundle] if len(bundle) > 1 else [[l] for l in bundle])


def _gmtch_todo(project, row, wanted):
    return (row["role"] == "aoi" and "GMTCH" in project["benchmarks"] and (not wanted or "GMTCH" in wanted)
            and not project.cell_raster(row, "GMTCH").exists())


def _settings_error(stale):
    cells = list(dict.fromkeys(s["cell"] for s in stale))
    layers = sorted({s["layer"] for s in stale})
    reasons = sorted({f"{s['layer']}: {s['reason']}" for s in stale})
    shown = ", ".join(cells[:20]) + (f", ... ({len(cells)} cells in total)" if len(cells) > 20 else "")
    return SettingsChanged(
        f"{len(stale)} downloaded rasters were made with other settings than project.yaml "
        f"({'; '.join(reasons[:6])}{'; ...' if len(reasons) > 6 else ''}) in cells {shown}. They are not "
        f"downloaded again automatically. Start a new project for the new settings, or delete those rasters "
        f"({', '.join(f'raw/<cell>_{l}.tif' for l in layers)} of the cells listed) and run again from the "
        f"download stage.")


# ================================================================================================ files
def _write(dst, arr, dtype, bands, epsg, x0, y1, res, **tags):
    import rasterio
    from rasterio.transform import Affine
    H, W = arr.shape[1:]
    prof = dict(driver="GTiff", dtype=dtype, count=arr.shape[0], width=W, height=H, crs=f"EPSG:{epsg}",
                transform=Affine(res, 0, x0, 0, -res, y1), nodata=(np.nan if dtype == "float64" else None),
                tiled=True, blockxsize=256, blockysize=256, compress="deflate",
                predictor=(3 if dtype == "float64" else 2), BIGTIFF="IF_SAFER")
    dst = str(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with rasterio.open(dst + ".partial.tif", "w", **prof) as o:
        o.write(arr)
        for k, b in enumerate(bands, 1):
            o.set_band_description(k, b)
        o.update_tags(**tags)
    replace_retry(dst + ".partial.tif", dst)


def gmtch_array(src, W, H):
    """GMTCH = p90 of the 10 x 10 Tolan 1 m pixels (255 = no data) of every 10 m cell; NaN where all are empty."""
    import rasterio
    with rasterio.open(src) as s:
        a = s.read(1).astype("float64")
    a[a == 255] = np.nan
    b = a.reshape(H, 10, W, 10).transpose(0, 2, 1, 3).reshape(H, W, 100)
    empty = np.isnan(b).all(axis=2)
    b[empty] = 0
    g = np.nanpercentile(b, 90, axis=2).astype("float32")
    g[empty] = np.nan
    return g


def write_gmtch(src, dst, x0, y1, W=CELL_PX, H=CELL_PX):
    """Tolan_1m raster -> GMTCH GeoTIFF on the 10 m cell grid (profile of the Tolan file, float32, NaN)."""
    import rasterio
    from rasterio.transform import Affine
    g = gmtch_array(src, W, H)
    with rasterio.open(src) as s:
        prof = s.profile
    prof.update(dtype="float32", width=W, height=H, nodata=np.nan, predictor=3,
                transform=Affine(RES, 0, x0, 0, -RES, y1))
    dst = str(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with rasterio.open(dst + ".partial.tif", "w", **prof) as o:
        o.write(g, 1)
        o.set_band_description(1, "GMTCH_p90")
    replace_retry(dst + ".partial.tif", dst)


# ================================================================================================ download
def meta_key(epsg, x0, y1, year, W=CELL_PX, H=CELL_PX):
    """S1MetaCache key of a cell: its rectangle and the year."""
    return int(epsg), float(x0), float(y1), int(W), int(H), int(year)


def _meta_key(row, grid, year):
    return meta_key(*cell_geo(row, grid), year)


class S1MetaCache:
    """Seasonal S1 metadata of a cell (gee.layers.s1_seasonal_metadata), shared by the cell's four S1_asc_k jobs:
    the first job that needs it computes it, the others wait for it. uses = {key: number of jobs}: an entry is dropped
    once that many jobs have taken it. A failed computation is kept as well (the cell's other jobs fail at once
    instead of repeating the request; the next run of the stage tries again)."""

    def __init__(self, uses=None):
        self._lock = threading.Lock()
        self._keys, self._meta, self._uses = {}, {}, dict(uses or {})
        self.computed = 0

    def get(self, key, compute):
        from .gee import io
        with self._lock:
            klock = self._keys.setdefault(key, threading.Lock())
        while not klock.acquire(timeout=0.5):             # another job of the cell is computing it
            io._check_cancel()
        try:
            if key not in self._meta:
                io._check_cancel()
                try:
                    self._meta[key] = compute()
                except io.Cancelled:
                    raise
                except Exception as e:  # noqa: BLE001
                    self._meta[key] = e
                self.computed += 1
            meta = self._meta[key]
        finally:
            klock.release()
        with self._lock:
            n = self._uses.get(key)
            if n is not None:
                self._uses[key] = n - 1
                if n <= 1:
                    self._meta.pop(key, None)
                    self._keys.pop(key, None)
        if isinstance(meta, Exception):
            raise RuntimeError(f"S1 seasonal metadata failed: {meta!r}"[:300]) from meta
        return meta


def _export_seasonal(layer, dst, geom, epsg, x0, y1, year, base, W, H, s1_cache):
    """One seasonal layer: S1_asc_k / S2_k unmasked to SENTINEL, float64,
    bands VV, VH / B2 ... B12; a season without acquisitions (S1: none in the metadata, S2: no bands) -> NaN."""
    from .gee import io, layers as L
    kind, k = L.seasonal(layer)
    bands = SEASONAL_BANDS[kind]
    tags = dict(source=layer, **base, season=SEASONS[k], window=f"{year - 1}-12-01/{year}-12-01")
    img, info = None, ""
    if kind == "S1":
        meta = (s1_cache.get(meta_key(epsg, x0, y1, year, W, H), lambda: L.s1_seasonal_metadata(geom, year))
                if s1_cache is not None else L.s1_seasonal_metadata(geom, year))
        n = sum(L.in_season(m["t"], k) for m in meta)
        tags.update(n_s1_images=str(n), s1_method="gee")
        info = f", {n} of {len(meta)} S1 images in season"
        if n:
            img = L.layer_image(layer, geom, year, meta=meta)
    else:
        tags["n_s1_images"] = ""
        img = L.layer_image(layer, geom, year)
        got = img.bandNames().getInfo()
        if not got:
            img = None
        elif got != bands:
            raise RuntimeError(f"{layer}: bands {got}, expected {bands}")
    if img is None:
        a = np.full((len(bands), H, W), np.nan)
        info += ", no data in this season (NaN)"
    else:
        a = io.download_grid(img.unmask(io.SENTINEL), epsg, x0, y1, W, H, RES, len(bands), CHUNK[layer])
    _write(dst, a, "float64", bands, epsg, x0, y1, RES, **tags)
    return info


def _export_s1_local(layers, out, epsg, x0, y1, year, base, W, H, region):
    """S1 (or the listed S1_asc_k) of one cell by the local way: raw scenes from Earth Engine (gee.s1_raw), the
    chain in numpy (s1_local). Same files as the Earth Engine way (float64, NaN, bands VV, VH in dB), tagged
    s1_method=local; n_s1_images = acquisitions with data in the cell. Raises s1_raw.NotApplicable."""
    from .gee import s1_raw
    seasonal = layers[0] in S1_SEASONAL
    t0 = time.time()
    cell, meta, info = s1_raw.fetch_cell(region, epsg, x0, y1, W, H,
                                         seasons=[int(l[-1]) for l in layers] if seasonal else None)
    t1 = time.time()
    acq = cell.per_acquisition(meta)
    tags = dict(**base, s1_method="local")
    for l in layers:
        if seasonal:
            k = int(l[-1])
            a, n = cell.seasonal(meta, k, acq)
            _write(out[l], a, "float64", SEASONAL_BANDS["S1"], epsg, x0, y1, RES, source=l, **tags,
                   season=SEASONS[k], window=f"{year - 1}-12-01/{year}-12-01", n_s1_images=str(n))
        else:
            a, n = cell.annual(meta, acq)
            _write(out[l], a, "float64", SEASONAL_BANDS["S1"], epsg, x0, y1, RES, source=l, **tags,
                   n_s1_images=str(n))
    return (f", local S1: {len(meta)} acquisitions, {info['scenes']} scenes, {info['requests']} requests, "
            f"{info['mb']:.0f} MB in {t1 - t0:.0f}s, chain {time.time() - t1:.0f}s")


def export(layers, out, epsg, x0, y1, year, gedi_window, built_up_mask, gee_project, W=CELL_PX, H=CELL_PX,
           s1_cache=None, s1_method="gee", s1_region=None):
    """One job: one layer, DEM + GEDI together (one request, split into per-layer files), or the S1_asc_k of a
    cell together (local way).

    out: {layer: destination path}. s1_cache: S1MetaCache shared by the jobs of a run (seasonal S1: one metadata
    request per cell); without it every S1_asc_k job fetches the metadata itself. s1_method 'local' with
    s1_region (gee.s1_raw.S1Region of the S1 window, prepared by run_download) computes S1 / S1_asc_k locally
    from the raw scenes; where the local way does not apply (not a WGS84 UTM grid, ...) the cell falls back to
    the Earth Engine way. Earth Engine must be initialised. Returns a short log string.
    """
    from .gee import io, layers as L
    t0 = time.time()
    geom = L.rect(epsg, x0, y1, W, H, RES)
    gw = [str(d) for d in gedi_window]
    gtags = gedi_tags(gw, built_up_mask)
    base = dict(project=gee_project or "", year=str(year))
    note = ""
    if is_s1(layers[0]) and s1_method == "local":
        from .gee import s1_raw
        try:
            if s1_region is None:
                raise s1_raw.NotApplicable("no S1 metadata of the region")
            info = _export_s1_local(layers, out, epsg, x0, y1, year, base, W, H, s1_region)
            return f"ok {time.time() - t0:.0f}s{info}"
        except s1_raw.NotApplicable as e:
            note = f" (Earth Engine way: local S1 does not apply here, {e})"
    if all(l in SEASONAL_LAYERS for l in layers):
        cache = s1_cache if len(layers) == 1 else S1MetaCache()          # a fallen-back S1 bundle: one metadata
        info = "".join(_export_seasonal(l, out[l], geom, epsg, x0, y1, year, base, W, H, cache) for l in layers)
        return f"ok {time.time() - t0:.0f}s{info}{note}"
    if len(layers) > 1:                                   # DEM / GEDI bundle: all 10 m, same chunking
        imgs = [L.layer_image(k, geom, year, gw, built_up_mask).unmask(0 if k == "DEM" else io.SENTINEL)
                .toDouble().rename(k) for k in layers]
        a = io.download_grid(L.ee.Image.cat(imgs), epsg, x0, y1, W, H, RES, len(layers), CHUNK["DEM"])
        for i, k in enumerate(layers):
            if k == "DEM":
                _write(out[k], np.nan_to_num(a[i:i + 1], nan=0).astype("int16"), "int16", ["elevation"],
                       epsg, x0, y1, RES, source=k, **base, n_s1_images="")
            else:
                _write(out[k], a[i:i + 1], "float64", ["rh95"], epsg, x0, y1, RES, source=k, **base,
                       n_s1_images="", **gtags)
        return f"ok {time.time() - t0:.0f}s ({'+'.join(layers)} in 1 request)"
    layer = layers[0]
    meta = None
    if layer == "S1":
        meta = L.s1_metadata(geom, year)                  # per cell (cheaper than shared lists)
    img = L.layer_image(layer, geom, year, gw, built_up_mask, meta)
    bands = img.bandNames().getInfo()
    dtype, fill = INT_LAYERS.get(layer, ("float64", None))
    img = img.unmask(fill if fill is not None else io.SENTINEL)
    res = 1.0 if layer == "Tolan_1m" else RES
    k = int(round(RES / res))
    a = io.download_grid(img, epsg, x0, y1, W * k, H * k, res, len(bands), CHUNK[layer])
    if dtype != "float64":
        a = np.nan_to_num(a, nan=fill).astype(dtype)
    _write(out[layer], a, dtype, bands, epsg, x0, y1, res, source=layer, **base,
           n_s1_images=str(len(meta)) if meta else "", **(gtags if layer == LABEL_LAYER else {}),
           **(dict(s1_method="gee") if layer == "S1" else {}))
    return f"ok {time.time() - t0:.0f}s" + (f", {len(meta)} S1 images" if meta else "") + note


def s1_region_file(project, seasonal):
    """JSON file of the local way's S1 metadata of the project's year (annual or seasonal window)."""
    return project.path("raw", f"s1_local_{project['year']}{'_seasonal' if seasonal else ''}.json")


def s1_regions(project, grid, jobs):
    """{seasonal (bool): gee.s1_raw.S1Region} with the metadata of every acquisition over the cells of the S1 jobs
    [(row, bundle)] (requested once for all cells, only what raw/s1_local_*.json lacks). Cells not on a WGS84 UTM
    grid are left out (they take the Earth Engine way). Earth Engine must be initialised."""
    from .gee import s1_raw
    need = {}
    for r, b in jobs:
        epsg, x0, y1 = cell_geo(r, grid)
        if s1_raw.utm_wgs84(epsg):
            need.setdefault(b[0] in S1_SEASONAL, []).append((epsg, x0, y1, CELL_PX, CELL_PX))
    out = {}
    for seasonal in (False, True):
        if seasonal in need or any((b[0] in S1_SEASONAL) == seasonal for _, b in jobs):
            reg = s1_raw.S1Region(s1_region_file(project, seasonal), project["year"], seasonal)
            reg.ensure(need.get(seasonal, []), log=lambda m: project.log(f"download: {m}"))
            out[seasonal] = reg
    return out


def run_download(project, workers=None, cells=None, layers=None):
    """Download every missing layer of the used cells (optionally only `cells` / `layers`), then derive GMTCH.

    Existing rasters (raw/ or an imported cell's own folder) are skipped; raw/ rasters downloaded with other
    settings raise SettingsChanged before anything is downloaded. Ctrl+C cancels the queued jobs (the running
    ones stop at their next chunk or retry wait) and re-raises. Returns
    {'cells', 'jobs', 'ok', 'failed': [{'cell', 'layers', 'error'}], 'gmtch'}.
    """
    grid, P = load_plan(project)
    if cells:
        unknown = set(cells) - set(P["cell"])
        if unknown:
            raise ValueError(f"cells not in the plan (or not used): {sorted(unknown)}")
        P = P[P["cell"].isin(list(cells))]
    wanted = _wanted(layers)
    rows = P.to_dict("records")
    stale, jobs = [], []
    if len(rows) > CHECK_EVERY:          # opening every existing raster can take minutes on a network drive
        project.log(f"download: checking the existing files of {len(rows)} cells ...")
    for i, r in enumerate(rows, 1):
        jobs += [(r, b) for b in jobs_for(project, r, wanted, stale)]
        if i % CHECK_EVERY == 0 and i < len(rows):
            project.log(f"download: checked {i}/{len(rows)} cells, {len(jobs)} jobs missing so far")
    if stale:
        raise _settings_error(stale)
    workers = int(workers or project["workers"] or 1)
    project.log(f"download: {len(rows)} cells, {len(jobs)} jobs missing, {workers} worker(s)"
                + (f", Sentinel-1 the {project['s1_method']} way" if any(is_s1(b[0]) for _, b in jobs) else ""))
    fails = []
    if jobs:
        from .gee import io
        io.ee_init(project["gee_project"])
        year, gw, bm, gp = project["year"], project["gedi_window"], project["built_up_mask"], project["gee_project"]
        uses = {}
        for r, b in jobs:
            if b[0] in S1_SEASONAL:
                k = _meta_key(r, grid, year)
                uses[k] = uses.get(k, 0) + 1
        s1_cache = S1MetaCache(uses)
        s1_method = project["s1_method"]
        regions = s1_regions(project, grid, [(r, b) for r, b in jobs if is_s1(b[0])]) if s1_method == "local" else {}

        def run(r, b):
            epsg, x0, y1 = cell_geo(r, grid)
            kw = dict(s1_cache=s1_cache) if b[0] in S1_SEASONAL else {}
            if is_s1(b[0]):
                kw.update(s1_method=s1_method, s1_region=regions.get(b[0] in S1_SEASONAL))
            return export(b, {l: project.raw(r["cell"], l) for l in b}, epsg, x0, y1, year, gw, bm, gp, **kw)
        io.CANCEL.clear()
        ex = ThreadPoolExecutor(workers)
        futs = {ex.submit(run, r, b): (r, b) for r, b in jobs}
        pending, i = set(futs), 0
        try:
            while pending:
                done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)   # timeout: Ctrl+C on Windows
                for f in done:
                    i += 1
                    r, b = futs[f]
                    try:
                        msg = f.result()
                    except Exception as e:  # noqa: BLE001
                        msg = f"FAILED {e!r}"[:300]
                        fails.append(dict(cell=r["cell"], layers="+".join(b), error=msg))
                    project.log(f"download [{i}/{len(jobs)}] {r['cell']} {'+'.join(b)}: {msg}")
        except BaseException:
            io.CANCEL.set()
            n = sum(f.cancel() for f in futs)
            ex.shutdown(wait=False, cancel_futures=True)
            project.log(f"download interrupted: {n} queued jobs cancelled, {sum(f.running() for f in futs)} "
                        f"running ones stop at their next chunk or retry wait")
            raise
        ex.shutdown()
    made = 0
    for r in rows:
        if not _gmtch_todo(project, r, wanted):
            continue
        src = project.cell_raster(r, "Tolan_1m")
        if not src.exists():
            project.log(f"download: {r['cell']} GMTCH not derived (no Tolan_1m)")
            continue
        _, x0, y1 = cell_geo(r, grid)
        write_gmtch(src, project.raw(r["cell"], "GMTCH"), x0, y1)
        made += 1
    project.log(f"download done: {len(jobs) - len(fails)} ok, {len(fails)} failed, {made} GMTCH derived"
                + (f" - failed: {[(f['cell'], f['layers']) for f in fails[:20]]}" if fails else ""))
    return dict(cells=len(rows), jobs=len(jobs), ok=len(jobs) - len(fails), failed=fails, gmtch=made)


def chunks(layer):
    """getDownloadURL chunks of one cell (before any 2 x 2 split): Tolan_1m is one 2560 px chunk (1 m grid)."""
    k = int(round(RES)) if layer == "Tolan_1m" else 1
    return math.ceil(CELL_PX * k / CHUNK[layer]) ** 2


def estimate(project):
    """Size of the (remaining) download for the UI: cells, jobs, Earth Engine requests (HTTP calls), approx. MB
    on disk, MB of raw Sentinel-1 scenes passing through (local way, not stored) and EECU-seconds (typical and a
    rough low-high range; S1 dominates and scales with the number of Sentinel-1 acquisitions)."""
    _, P = load_plan(project)
    rows = P.to_dict("records")
    method = project["s1_method"]
    jobs, req = [], 0
    for r in rows:
        bs = jobs_for(project, r)
        jobs += bs
        if method == "gee":
            req += any(b[0] in S1_SEASONAL for b in bs)        # seasonal S1 metadata: once per cell
    n_layer, mb, transfer, eecu, lo, hi, local_jobs = {}, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    for b in jobs:
        if is_s1(b[0]) and method == "local":
            req += 3                                            # 2-3 computePixels of the raw scenes
            local_jobs += 1
        else:
            req += 2 * chunks(b[0])                             # getDownloadURL + GET per chunk
            if len(b) == 1:
                req += (b[0] not in S1_SEASONAL) + (b[0] == "S1")   # bandNames (known for seasonal S1), S1 metadata
        for l in b:
            n_layer[l] = n_layer.get(l, 0) + 1
            m, s, (a, z) = layer_cost(l, method)
            mb, eecu, lo, hi = mb + m, eecu + s, lo + a, hi + z
            transfer += TRANSFER_MB.get(l, 0) if method == "local" else 0
    if local_jobs:
        req += 2 + local_jobs // 50                             # acquisition metadata of the area, once
    roles = P["role"].value_counts().to_dict() if len(P) else {}
    return dict(cells=len(rows), aoi_cells=int(roles.get("aoi", 0)), ring_cells=int(roles.get("ring", 0)),
                jobs=len(jobs), layers=n_layer, requests=int(req), mb=round(mb, 1), transfer_mb=round(transfer),
                eecu_s=round(eecu), eecu_s_low=round(lo), eecu_s_high=round(hi), s1_method=method,
                note="requests = 2 HTTP calls per chunk + bandNames / S1 metadata (Earth Engine way; the local S1 "
                     "way: 2-3 requests per cell + the acquisition metadata once); chunks split after a memory "
                     "error and retries (429 / 5xx) come on top. EECU-s are rough: Sentinel-1 costs about 10 per "
                     "cell the local way (about 50 MB of raw scenes pass through per cell, not stored) and 100-3000 "
                     "the Earth Engine way, growing with the number of acquisitions (annual, or the four seasons "
                     "together); the stack, mosaics and models need further disk space.")

