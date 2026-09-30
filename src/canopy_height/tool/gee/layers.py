"""Earth Engine image recipes of every layer (the paper pipeline's gee_layers.py, config constants as parameters).

  Embedding  GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL, [year-01-01, year-12-31), filterBounds, median        64 bands
  DEM        USGS/SRTMGL1_003 elevation                                                                    int16
  S1         gee_s1_ard chain (IW, VV+VH, both orbits, border-noise mask, multi-temporal Quegan filter with a
             15 px boxcar and the 10 closest acquisitions, VOLUME terrain flattening on SRTM, dB), annual median
             of [year-01-01, year-12-31). Evaluated the 's1_fast' way: each image's filter set D_i and look
             direction are computed once per cell (s1_metadata) and passed in as constants - same mathematics as
             the library, a fraction of the EECU                                                           2 bands
  S2         S2_SR_HARMONIZED + S2_CLOUD_PROBABILITY (< 20), CLOUDY_PIXEL_PERCENTAGE < 40, QA60 bits 10/11,
             B8A/B9 edge mask, B2 B3 B4 B5 B6 B7 B8 B11 B12, annual median, not divided by 10000             9 bands
  GEDI       LARSE/GEDI/GEDI02_A_002_MONTHLY rh95, quality_flag 1, degrade_flag 0, median over [start, end);
             training labels additionally masked where WorldCover 2020 = built-up (class 50)
  ETH        users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1 (HRCH)                                         uint8
  UMD        users/potapovpeter/GEDI_V27 mosaic (GFCH)                                                     uint8
  Tolan_1m   projects/meta-forest-monitoring-okw37/assets/CanopyHeight mosaic (GMTCH source, 1 m, 255 = no data)
The expressions are those of the pipeline, unchanged: a re-download of a pipeline cell gives the pipeline file.

Seasonal composites (input representations T / TE), the paper's seasonal pipeline (seasonal_gee.py,
s1_fast.py): window [Dec (year - 1), Dec year), season k = months SEASON_MONTHS[k] (DJF, MAM, JJA, SON) by
calendarRange / UTC month of the acquisition.
  S2_k       as S2 but CLOUDY_PIXEL_PERCENTAGE < 50 (filtered before the masks), cloud probability < 50, median of
             the season's images / 10000 (0-1 reflectance), no final clip                                    9 bands
  S1_asc_k   the S1 chain above (ORBIT BOTH, the name is historical), the metadata of the whole window evaluated
             once per cell (s1_seasonal_metadata), median of the season's acquisitions in dB                 2 bands
The local S1 way (s1_method: local, gee.s1_raw + tool.s1_local) uses only s1_acquisitions / s1_metadata_of /
s1_scene_info / s1_raw_bands / s1_cell_bands below: pixels and metadata, the chain itself runs in numpy.
The S1 helpers (_get_filtered_collection, _heading, _Shared, _inner, _quegan, _correct) are those of s1_fast.py
(checked line by line: only docstrings and constant names differ), so the annual and the seasonal S1 share them.
border_noise_correction / helper / speckle_filter are the unmodified gee_s1_ard modules (MIT, Mullissa 2021) in
third_party/gee_s1_ard; they import each other as top-level modules, hence the sys.path entry.
"""
import datetime as dt
import math
import sys
from pathlib import Path

_S1_ARD = str(Path(__file__).resolve().parent / "third_party" / "gee_s1_ard")
if _S1_ARD not in sys.path:
    sys.path.insert(0, _S1_ARD)
import ee  # noqa: E402
import border_noise_correction as bnc  # noqa: E402
import helper  # noqa: E402
import speckle_filter as sf  # noqa: E402

from . import io  # noqa: E402

S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12"]
KERNEL, NR_OF_IMAGES = 15, 10
S1_BANDS = ["VV", "VH"]
MEAN_BANDS = [b + "_mean" for b in S1_BANDS]
RATIO_BANDS = [b + "_ratio" for b in S1_BANDS]
# seasonal composites (seasonal_gee.py): season k = months SEASON_MONTHS[k] of the window [Dec (year - 1), Dec year)
SEQ = [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
SEASON_MONTHS = [(SEQ[m - 1], SEQ[m + 1]) for m in (1, 4, 7, 10)]      # (12,2) (3,5) (6,8) (9,11)
SEASONAL_CLOUD = 50                         # seasonal S2: CLOUDY_PIXEL_PERCENTAGE < 50 and cloud probability < 50
SEASONAL_LAYERS = [f"S1_asc_{k}" for k in range(4)] + [f"S2_{k}" for k in range(4)]
LAYERS = ["Embedding", "DEM", "S1", "S2", "GEDI", "ETH", "UMD", "Tolan_1m"] + SEASONAL_LAYERS


def rect(epsg, x0, y1, W, H, res=10.0):
    """Cell / piece rectangle [x0, x0 + W*res] x [y1 - H*res, y1] in EPSG:epsg (planar)."""
    x0, y1, res = float(x0), float(y1), float(res)
    return ee.Geometry.Rectangle([x0, y1 - H * res, x0 + W * res, y1], proj=f"EPSG:{int(epsg)}", geodesic=False)


# ------------------------------------------------------------------------------------------------ optical / DEM
def embedding(geom, year):
    return (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
            .filterDate(ee.Date(f"{year}-01-01"), ee.Date(f"{year}-12-31")).filterBounds(geom).median().clip(geom))


def s2(geom, year):
    start, end = ee.Date(f"{year}-01-01"), ee.Date(f"{year}-12-31")
    criteria = ee.Filter.And(ee.Filter.bounds(geom), ee.Filter.date(start, end))

    def mask_edges(img):
        return img.updateMask(img.select("B8A").mask().updateMask(img.select("B9").mask()))

    def mask_clouds(img):
        return img.updateMask(ee.Image(img.get("cloud_mask")).select("probability").lt(20))

    def mask_qa60(img):
        qa = img.select("QA60")
        return img.updateMask(qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0)))

    col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filter(criteria).map(mask_edges)
    clouds = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filter(criteria)
    joined = ee.Join.saveFirst("cloud_mask").apply(
        primary=col, secondary=clouds, condition=ee.Filter.equals(leftField="system:index", rightField="system:index"))
    col = (ee.ImageCollection(joined).map(mask_clouds).map(lambda i: i.clip(geom))
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40)).map(mask_qa60).select(S2_BANDS))
    return col.median().clip(geom)


def seasonal_dates(year):
    """Window of the seasonal composites: [Dec (year - 1), Dec year)."""
    return ee.Date(f"{year - 1}-12-01"), ee.Date(f"{year}-12-01")


def s2_seasonal(geom, year, season):
    """seasonal_gee.s2_seasonal: season 0..3 of the window, median / 10000."""
    start, end = seasonal_dates(year)
    criteria = ee.Filter.And(ee.Filter.bounds(geom), ee.Filter.date(start, end))

    def mask_edges(img):
        return img.updateMask(img.select("B8A").mask().updateMask(img.select("B9").mask()))

    def mask_clouds(img):
        clouds = ee.Image(img.get("cloud_mask")).select("probability")
        return img.updateMask(clouds.lt(SEASONAL_CLOUD))

    def mask_qa60(img):
        qa = img.select("QA60")
        mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        return img.updateMask(mask)

    col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filter(criteria).map(mask_edges)
    clouds = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filter(criteria)
    joined = ee.Join.saveFirst("cloud_mask").apply(
        primary=col, secondary=clouds, condition=ee.Filter.equals(leftField="system:index", rightField="system:index"))
    col = (ee.ImageCollection(joined).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", SEASONAL_CLOUD))
           .map(mask_clouds).map(lambda i: i.clip(geom)).map(mask_qa60).select(S2_BANDS))
    t1, t2 = SEASON_MONTHS[season]
    return col.filter(ee.Filter.calendarRange(t1, t2, "month")).median().divide(10000)


def dem(geom):
    return ee.Image("USGS/SRTMGL1_003").clip(geom)


def gedi(geom, start, end, built_up_mask):
    """Median rh95 of the monthly GEDI L2A rasters in [start, end) (dates 'YYYY-MM-DD', end exclusive)."""
    def q(im):
        return im.updateMask(im.select("quality_flag").eq(1)).updateMask(im.select("degrade_flag").eq(0))
    img = (ee.ImageCollection("LARSE/GEDI/GEDI02_A_002_MONTHLY").map(q)
           .filter(ee.Filter.date(start, end)).select("rh95").median().clip(geom))
    if built_up_mask:
        img = img.updateMask(ee.Image("ESA/WorldCover/v100/2020").select("Map").neq(50))
    return img


# ------------------------------------------------------------------------------------------------ benchmarks
def eth(geom):
    return ee.Image("users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1").clip(geom)


def umd(geom):
    return ee.ImageCollection("users/potapovpeter/GEDI_V27").mosaic().clip(geom)


def tolan(geom):
    return ee.ImageCollection("projects/meta-forest-monitoring-okw37/assets/CanopyHeight").mosaic().clip(geom)


# ------------------------------------------------------------------------------------------------ S1 (s1_fast)
def _get_filtered_collection(image):
    """Verbatim speckle_filter.MultiTemporal_Filter.Quegan.get_filtered_collection (NR_OF_IMAGES = 10)."""
    s1_coll = ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT") \
        .filterBounds(image.geometry()) \
        .filter(ee.Filter.eq("instrumentMode", "IW")) \
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation",
                                       ee.List(image.get("transmitterReceiverPolarisation")).get(-1))) \
        .filter(ee.Filter.Or(ee.Filter.eq("relativeOrbitNumber_stop", image.get("relativeOrbitNumber_stop")),
                             ee.Filter.eq("relativeOrbitNumber_stop", image.get("relativeOrbitNumber_start")))) \
        .map(lambda im: im.resample())

    def check_overlap(_image):
        s1 = s1_coll.filterDate(_image.date(), _image.date().advance(1, "day"))
        intersect = image.geometry().intersection(s1.geometry().dissolve(), 10)
        valid_date = ee.Algorithms.If(intersect.area(10).divide(image.geometry().area(10)).gt(0.95),
                                      _image.date().format("YYYY-MM-dd"))
        return ee.Feature(None, {"date": valid_date})

    dates_before = s1_coll.filterDate("2014-01-01", image.date().advance(1, "day")) \
        .sort("system:time_start", False).limit(5 * NR_OF_IMAGES) \
        .map(check_overlap).distinct("date").aggregate_array("date")
    dates = ee.List(ee.Algorithms.If(
        dates_before.size().gte(NR_OF_IMAGES),
        dates_before.slice(0, NR_OF_IMAGES),
        s1_coll.filterDate(image.date(), "2100-01-01").sort("system:time_start", True).limit(5 * NR_OF_IMAGES)
        .map(check_overlap).distinct("date").aggregate_array("date")
        .cat(dates_before).distinct().sort().slice(0, NR_OF_IMAGES)))
    return ee.ImageCollection(dates.map(
        lambda date: s1_coll.filterDate(date, ee.Date(date).advance(1, "day")).toList(s1_coll.size())).flatten())


def _heading(image):
    """Verbatim terrain_flattening._correct look direction."""
    heading = ee.Terrain.aspect(image.select("angle")).reduceRegion(ee.Reducer.mean(), image.geometry(), 1000)
    heading = ee.Dictionary(heading).combine({"aspect": 0}, False).get("aspect")
    return ee.Algorithms.If(ee.Number(heading).gt(180), ee.Number(heading).subtract(360), ee.Number(heading))


class _Shared:
    """One ee.Image object per raw acquisition, so the serialized graph references each only once."""

    def __init__(self):
        self.raw, self.ratio = {}, {}

    def raw_img(self, sid):
        if sid not in self.raw:
            self.raw[sid] = ee.Image(sid).resample().select(S1_BANDS)
        return self.raw[sid]

    def ratio_img(self, sid):
        if sid not in self.ratio:
            self.ratio[sid] = _inner(self.raw_img(sid)).select(RATIO_BANDS)
        return self.ratio[sid]


def _inner(image):
    _filtered = sf.boxcar(image, KERNEL).select(S1_BANDS).rename(MEAN_BANDS)
    _ratio = image.select(S1_BANDS).divide(_filtered).rename(RATIO_BANDS)
    return _filtered.addBands(_ratio)


def _quegan(image, D, sh):
    count_img = ee.ImageCollection([sh.raw_img(s) for s in D]).reduce(ee.Reducer.count())
    isum = ee.ImageCollection([sh.ratio_img(s) for s in D]).reduce(ee.Reducer.sum())
    filtered = _inner(image).select(MEAN_BANDS)
    output = filtered.divide(count_img).multiply(isum).rename(S1_BANDS)
    return image.addBands(output, None, True)


def _correct(image, heading):
    """terrain_flattening._correct (VOLUME model, buffer 0) with the precomputed heading."""
    dem = ee.Image("USGS/SRTMGL1_003")
    ninetyRad = ee.Image.constant(90).multiply(math.pi / 180)
    bandNames = image.bandNames()
    geom = image.geometry()
    proj = image.select(1).projection()
    elevation = dem.resample("bilinear").reproject(proj, None, 10).clip(geom)
    theta_iRad = image.select("angle").multiply(math.pi / 180)
    phi_iRad = ee.Image.constant(heading).multiply(math.pi / 180)
    alpha_sRad = ee.Terrain.slope(elevation).select("slope").multiply(math.pi / 180)
    aspect = ee.Terrain.aspect(elevation).select("aspect").clip(geom)
    aspect_minus = aspect.updateMask(aspect.gt(180)).subtract(360)
    phi_sRad = aspect.updateMask(aspect.lte(180)).unmask().add(aspect_minus.unmask()) \
        .multiply(-1).multiply(math.pi / 180)
    phi_rRad = phi_iRad.subtract(phi_sRad)
    alpha_rRad = (alpha_sRad.tan().multiply(phi_rRad.cos())).atan()
    gamma0 = image.divide(theta_iRad.cos())
    scf = (ninetyRad.subtract(theta_iRad).add(alpha_rRad)).tan().divide((ninetyRad.subtract(theta_iRad)).tan())
    gamma0_flat = gamma0.multiply(scf)
    layover = alpha_rRad.lt(theta_iRad).rename("layover")
    shadow = alpha_rRad.gt(ee.Image.constant(-1).multiply(ninetyRad.subtract(theta_iRad))).rename("shadow")
    mask = layover.And(shadow).rename("no_data_mask")
    output = gamma0_flat.mask(mask).rename(bandNames).copyProperties(image)
    output = ee.Image(output).addBands(image.select("angle"), None, True)
    return output.set("system:time_start", image.get("system:time_start"))


def _meta_feature(img):
    """Metadata of one acquisition (after f_mask_edges): id, time, filter set D_i (ordered ids), heading."""
    return ee.Feature(None, {"id": img.get("system:id"), "t": img.get("system:time_start"),
                             "D": _get_filtered_collection(img).aggregate_array("system:id"),
                             "heading": _heading(img)})


def _meta_list(col):
    feats = ee.FeatureCollection(col.map(_meta_feature)).getInfo()["features"]
    return [{k: ft["properties"][k] for k in ("id", "t", "D", "heading")} for ft in feats]


def s1_metadata(geom, year):
    """Per acquisition of the year over geom: id, time, filter set D_i (ordered ids), heading - one getInfo."""
    col = (ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT")
           .filter(ee.Filter.eq("instrumentMode", "IW")).filter(ee.Filter.eq("resolution_meters", 10))
           .filterDate(ee.Date(f"{year}-01-01"), ee.Date(f"{year}-12-31")).filterBounds(geom)
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
           .select(["VV", "VH", "angle"])).map(bnc.f_mask_edges)
    return _meta_list(col)


def s1(geom, meta):
    sh = _Shared()
    imgs = []
    for m in meta:
        img = bnc.f_mask_edges(ee.Image(m["id"]).select(["VV", "VH", "angle"]))
        imgs.append(helper.lin_to_db(_correct(_quegan(img, m["D"], sh), m["heading"])).select(S1_BANDS))
    return ee.ImageCollection(imgs).median().clip(geom)


# ------------------------------------------------------------------------------------------------ S1 seasonal
def s1_seasonal_metadata(geom, year):
    """s1_fast.metadata: s1_metadata of the acquisitions of the seasonal window [Dec (year - 1), Dec year) (all four
    seasons; s1_seasonal picks a season's) - one getInfo per cell."""
    start, end = seasonal_dates(year)
    col = (ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT")
           .filter(ee.Filter.eq("instrumentMode", "IW")).filter(ee.Filter.eq("resolution_meters", 10))
           .filterDate(start, end).filterBounds(geom)
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
           .select(["VV", "VH", "angle"])).map(bnc.f_mask_edges)
    return _meta_list(col)


# ------------------------------------------------------------------------------------------------ S1 local way
def s1_window(year, seasonal=False):
    """Acquisition window of the annual S1 ([year-01-01, year-12-31)) or of the seasonal ones ([Dec (year - 1),
    Dec year)) as date strings."""
    return (f"{year - 1}-12-01", f"{year}-12-01") if seasonal else (f"{year}-01-01", f"{year}-12-31")


def _s1_acq_collection(start, end):
    """The acquisitions of s1_metadata / s1_seasonal_metadata, without the bounds filter."""
    return (ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT")
            .filter(ee.Filter.eq("instrumentMode", "IW")).filter(ee.Filter.eq("resolution_meters", 10))
            .filterDate(ee.Date(start), ee.Date(end))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))


def s1_acquisitions(region, start, end):
    """system:index of the acquisitions of [start, end) over region (a geometry)."""
    return _s1_acq_collection(start, end).filterBounds(region).aggregate_array("system:index").getInfo()


def s1_metadata_of(indices, start, end):
    """s1_metadata of the acquisitions with these system:index values: D_i and heading depend on the acquisition
    only, so this is the metadata of every cell they cover."""
    col = (_s1_acq_collection(start, end).filter(ee.Filter.inList("system:index", list(indices)))
           .select(["VV", "VH", "angle"])).map(bnc.f_mask_edges)
    return _meta_list(col)


def s1_scene_info(ids):
    """{id: {crs, t (affine of the VH grid), foot (footprint GeoJSON)}} of S1_GRD_FLOAT scenes."""
    got = ee.List([ee.Dictionary({"id": s, "proj": ee.Image(s).select("VH").projection(),
                                  "foot": ee.Image(s).geometry()}) for s in ids]).getInfo()
    return {g["id"]: {"crs": g["proj"]["crs"], "t": g["proj"]["transform"], "foot": g["foot"]} for g in got}


def s1_raw_bands(sid, k, main):
    """Bands of scene sid for the local way: VV_k, VH_k (linear, 0 where masked), m_k (both valid); for an
    acquisition (main) also elev_k (SRTM bilinear onto the scene grid, as _correct, not clipped) and tv_k (1 where
    ee.Terrain.slope of the clipped elevation is valid: the terrain algorithms run on the scene grid, where the
    clip mask is fractional at the footprint edge, and need the pixel and its 4 neighbours). Requested on a 10 m
    grid of the scene's CRS, nearest = an integer shift of the scene grid."""
    im = ee.Image(sid)
    out = [im.select(["VV", "VH"]).unmask(0).float().rename([f"VV_{k}", f"VH_{k}"]),
           im.select("VV").mask().And(im.select("VH").mask()).toByte().rename(f"m_{k}")]
    if main:
        elev = ee.Image("USGS/SRTMGL1_003").resample("bilinear").reproject(im.select(1).projection(), None, 10)
        out += [elev.float().rename(f"elev_{k}"),
                ee.Terrain.slope(elev.clip(im.geometry())).mask().gt(0).toByte().rename(f"tv_{k}")]
    return out


def s1_cell_bands(sid, k):
    """Bands of acquisition sid on the analysis grid: angle_k (nearest from its coarse grid, as the chain reads it,
    -9999 = masked) and foot_k (1 where the chain's aspect.clip(image.geometry()) keeps the pixel: that clip runs
    on the requested grid)."""
    im = ee.Image(sid)
    return [im.select("angle").unmask(-9999).float().rename(f"angle_{k}"),
            ee.Image(1).clip(im.geometry()).mask().gt(0).toByte().rename(f"foot_{k}")]


def in_season(t_ms, season):
    """s1_fast._in_season: does an acquisition time (ms, UTC) fall in the months of season 0..3."""
    m = dt.datetime.fromtimestamp(t_ms / 1000, tz=dt.timezone.utc).month
    t1, t2 = SEASON_MONTHS[season]
    return t1 <= m <= t2 if t1 <= t2 else (m >= t1 or m <= t2)


def s1_seasonal(geom, meta, season, shared=None):
    """s1_fast.s1_seasonal: median (dB) of the acquisitions of meta = s1_seasonal_metadata(geom, year) in season 0..3.
    shared: a _Shared of the cell (optional; the request graph is the same without it)."""
    sh = shared or _Shared()
    imgs = []
    for m in meta:
        if not in_season(m["t"], season):
            continue
        img = bnc.f_mask_edges(ee.Image(m["id"]).select(["VV", "VH", "angle"]))
        img = helper.lin_to_db(_correct(_quegan(img, m["D"], sh), m["heading"]))
        imgs.append(img.select(S1_BANDS))
    return ee.ImageCollection(imgs).median().clip(geom)


# ------------------------------------------------------------------------------------------------ dispatcher
def seasonal(layer):
    """('S1', k) for S1_asc_k, ('S2', k) for S2_k, else None."""
    if layer in SEASONAL_LAYERS:
        return layer[:2], int(layer[-1])
    return None


def layer_image(layer, geom, year, gedi_window=None, built_up_mask=True, meta=None):
    """Image of a file layer (LAYERS) over geom; S1 needs meta = s1_metadata(geom, year), S1_asc_k
    meta = s1_seasonal_metadata(geom, year), GEDI the window."""
    if layer == "Embedding":
        return embedding(geom, year)
    if layer == "DEM":
        return dem(geom)
    if layer == "S1":
        return s1(geom, meta)
    if layer == "S2":
        return s2(geom, year)
    if seasonal(layer):
        kind, k = seasonal(layer)
        return s1_seasonal(geom, meta, k) if kind == "S1" else s2_seasonal(geom, year, k)
    if layer == "GEDI":
        return gedi(geom, gedi_window[0], gedi_window[1], built_up_mask)
    if layer == "ETH":
        return eth(geom)
    if layer == "UMD":
        return umd(geom)
    if layer == "Tolan_1m":
        return tolan(geom)
    raise ValueError(f"unknown layer {layer!r}")


# ------------------------------------------------------------------------------------------------ land cover
def worldcover_shares(cells, epsg, cell_m=2560, gee_project=None):
    """WorldCover 2020 shares (water 80, built-up 50, tree 10) of the cells [(x0, y1), ...] (upper-left corners,
    cell_m squares in EPSG:epsg), 30 m frequency histogram, in batches of 250 -> [(water, built, tree), ...].

    Earth Engine must be initialised (gee.io.ee_init) or gee_project given.
    """
    from shapely.geometry import box
    io.ee_init(gee_project)
    blk = float(cell_m)
    wc = ee.Image("ESA/WorldCover/v100/2020")
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry(box(float(x), float(y) - blk, float(x) + blk, float(y))
                                                      .__geo_interface__, f"EPSG:{int(epsg)}", False), {"i": i})
                               for i, (x, y) in enumerate(cells)])
    out = {}
    for k in range(0, len(cells), 250):
        res = wc.reduceRegions(ee.FeatureCollection(fc.toList(250, k)), ee.Reducer.frequencyHistogram(),
                               30).getInfo()["features"]
        for f in res:
            h = f["properties"].get("histogram") or {}
            tot = sum(h.values()) or 1
            out[f["properties"]["i"]] = (h.get("80", 0) / tot, h.get("50", 0) / tot, h.get("10", 0) / tot)
    return [out.get(i, (0, 0, 0)) for i in range(len(cells))]
