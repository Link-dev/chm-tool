"""Map rendering of the canopy-height tool: rasters -> coloured PNG overlays for Leaflet (folium), the study area
and its cells as GeoJSON, the folium maps of the app's Study area and Results tabs, and the byte cache of the
app's download buttons.

Canopy-height layers (the local models, the benchmarks GMTCH / GFCH / HRCH, ALS, GEDI) are drawn on ONE common
0..max colour scale (viridis, no data transparent), so maps can be compared by eye as in the paper's figures.
Overlays are reprojected with rasterio.warp to Web Mercator, the grid Leaflet displays (an EPSG:4326 image is
stretched linearly in latitude by Leaflet and drifts by a few display pixels over large areas); their bounds are
given in EPSG:4326 (lat / lon). Rasters are read reduced to at most `max_px` pixels on the long side by block
means (block maxima for the sparse GEDI layer, so footprints stay visible), without holding the full-resolution
band in memory.

rasterio, matplotlib, pyproj and folium are imported inside the functions (the report can use the raster helpers
without folium, and the app starts without them).
"""
import base64
import contextlib
import functools
import io
import json
import math
import warnings
from pathlib import Path

import numpy as np

CELL_M = 2560.0
SENTINEL_BELOW = -900.0              # -999 (labels) / -9999 (downloads) no-data sentinels
CMAP = "viridis"
CATEGORY_COLORS = {"aoi": "#e6550d", "ring": "#3182bd", "dropped (water)": "#969696"}
AOI_STYLE = {"color": "#d62728", "weight": 2.5, "fill": False}
SATELLITE = ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
             "Esri, Maxar, Earthstar Geographics")


# ---------------------------------------------------------------------------------------------- rasters
def _clean(a, nodata):
    """float32 copy with the nodata value and <= -900 sentinels as NaN."""
    a = a.astype(np.float32)
    if nodata is not None and not np.isnan(nodata):
        a[a == np.float32(nodata)] = np.nan
    a[a <= SENTINEL_BELOW] = np.nan
    return a


def read_band(path, band=1, mask_path=None, max_px=None, how="mean"):
    """Band of a raster as float32 with no data as NaN (nodata value, NaN, <= -900 sentinels, and pixels where
    the optional mask raster of the same grid is 0) -> (array, transform, crs).

    With `max_px`, a raster longer than max_px pixels is read reduced by f = ceil(long side / max_px): a 'mean'
    raster with a nodata value by GDAL block averages (rasterio out_shape, which uses overviews when the file has
    them; the mask by majority), any other by exact f x f block means / maxima (`how`) computed strip by strip."""
    import rasterio
    with contextlib.ExitStack() as es:
        src = es.enter_context(rasterio.open(path))
        if src.crs is None:
            raise ValueError(f"{path} has no CRS")
        m = None
        if mask_path is not None and Path(mask_path).exists():
            m = es.enter_context(rasterio.open(mask_path))
            if (m.height, m.width) != (src.height, src.width) or not m.transform.almost_equals(src.transform):
                m = None
        f = max(1, math.ceil(max(src.height, src.width) / max_px)) if max_px else 1
        if f == 1:
            a, tr = _clean(src.read(band), src.nodata), src.transform
            if m is not None:
                a[m.read(1) == 0] = np.nan
        elif how == "mean" and src.nodata is not None:
            a, tr = _read_average(src, band, f, m)
        else:
            a, tr = _read_blocks(src, band, f, how, m)
        return a, tr, src.crs


def _read_average(src, band, f, m):
    from affine import Affine
    from rasterio.enums import Resampling
    h, w = -(-src.height // f), -(-src.width // f)
    a = _clean(src.read(band, out_shape=(h, w), resampling=Resampling.average), src.nodata)
    if m is not None:
        a[m.read(1, out_shape=(h, w), resampling=Resampling.mode) == 0] = np.nan
    tr, sx, sy = src.transform, src.width / w, src.height / h
    return a, Affine(tr.a * sx, tr.b * sy, tr.c, tr.d * sx, tr.e * sy, tr.f)     # same extent, w x h pixels


def _read_blocks(src, band, f, how, m, strip_px=1 << 22):
    from affine import Affine
    from rasterio.windows import Window
    H, W = src.height, src.width
    rows = max(1, strip_px // (W * f)) * f
    out = np.empty((-(-H // f), -(-W // f)), np.float32)
    for r0 in range(0, H, rows):
        win = Window(0, r0, W, min(rows, H - r0))
        a = _clean(src.read(band, window=win), src.nodata)
        if m is not None:
            a[m.read(1, window=win) == 0] = np.nan
        b = block_reduce(a, f, how)
        out[r0 // f:r0 // f + b.shape[0]] = b
    tr = src.transform
    return out, Affine(tr.a * f, tr.b * f, tr.c, tr.d * f, tr.e * f, tr.f)     # pixel size x f, same origin


def block_reduce(a, f, how="mean"):
    """f x f block mean (or max) ignoring NaN; the array is padded with NaN to a multiple of f."""
    if f <= 1:
        return a
    h, w = a.shape
    hp, wp = -(-h // f) * f, -(-w // f) * f
    p = np.full((hp, wp), np.nan, np.float32)
    p[:h, :w] = a
    b = p.reshape(hp // f, f, wp // f, f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = np.nanmax(b, axis=(1, 3)) if how == "max" else np.nanmean(b, axis=(1, 3))
    return r.astype(np.float32)


def warp(path, max_px=1024, how="mean", mask_path=None, dst_crs="EPSG:3857"):
    """Raster -> (float32 array on a `dst_crs` grid with <= ~max_px pixels on the long side, NaN = no data,
    bounds [[south, west], [north, east]] in EPSG:4326)."""
    from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds
    a, tr, crs = read_band(path, mask_path=mask_path, max_px=max_px, how=how)
    h, w = a.shape
    left, top = tr.c, tr.f
    right, bottom = left + w * tr.a, top + h * tr.e
    dtr, dw, dh = calculate_default_transform(crs, dst_crs, w, h, left=left, bottom=bottom, right=right, top=top)
    out = np.full((dh, dw), np.nan, np.float32)
    reproject(a, out, src_transform=tr, src_crs=crs, dst_transform=dtr, dst_crs=dst_crs,
              src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest)
    w84 = transform_bounds(dst_crs, "EPSG:4326", dtr.c, dtr.f + dh * dtr.e, dtr.c + dw * dtr.a, dtr.f)
    return out, [[w84[1], w84[0]], [w84[3], w84[2]]]


@functools.lru_cache(maxsize=2)
def file_bytes(path, mtime):
    """Bytes of a result file for the app's download buttons: read when a button is clicked and kept for the
    last two files by (path, modification time). Here, not in app.py, because Streamlit re-executes the app
    script (and would renew its caches) on every rerun."""
    return Path(path).read_bytes()


def common_vmax(arrays, q=99.5, step=5.0):
    """Upper end of the common colour scale: largest q-th percentile of the layers, rounded up to `step` m
    (a percentile, so single outliers such as odd GEDI shots do not wash out the colours)."""
    v = [float(np.nanpercentile(a, q)) for a in arrays if a is not None and np.isfinite(a).any()]
    v = max(v) if v else step
    return max(step, math.ceil(v / step) * step)


def colorize(a, vmax, vmin=0.0, cmap=CMAP):
    """float array -> RGBA uint8 [h, w, 4] on the vmin..vmax scale, NaN transparent."""
    import matplotlib
    ok = np.isfinite(a)
    z = np.clip((np.where(ok, a, vmin) - vmin) / max(float(vmax) - float(vmin), 1e-9), 0, 1)
    rgba = matplotlib.colormaps[cmap](z, bytes=True)
    rgba[~ok] = 0
    return rgba


def png_data_url(rgba):
    """RGBA uint8 array -> 'data:image/png;base64,...' (for folium.raster_layers.ImageOverlay)."""
    from matplotlib import image as mimage
    buf = io.BytesIO()
    mimage.imsave(buf, rgba, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def colormap_hex(n=11, cmap=CMAP):
    import matplotlib
    from matplotlib.colors import to_hex
    cm = matplotlib.colormaps[cmap]
    return [to_hex(cm(i / (n - 1))) for i in range(n)]


def save_quicklook(a, path, vmax, title="", cmap=CMAP, dpi=120):
    """Quick-look PNG of a canopy-height array on the 0..vmax scale with a colour bar (no pyplot state, so it is
    safe in the app's threads)."""
    from matplotlib.figure import Figure
    h, w = a.shape
    fig = Figure(figsize=(5.2, 5.2 * h / max(w, 1) + 0.4), dpi=dpi)
    ax = fig.subplots()
    im = ax.imshow(np.ma.masked_invalid(a), cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Canopy height (m)")
    fig.savefig(path, bbox_inches="tight")
    return Path(path)


# ---------------------------------------------------------------------------------------------- vectors
def aoi_geometry(src):
    """Study area of a project (aoi.geojson path or GeoJSON dict) as one shapely geometry in EPSG:4326."""
    from shapely.geometry import shape
    from shapely.ops import unary_union
    gj = json.loads(Path(src).read_text(encoding="utf-8")) if isinstance(src, (str, Path)) else src
    if gj.get("type") == "FeatureCollection":
        geoms = [shape(f["geometry"]) for f in gj["features"] if f.get("geometry")]
    elif gj.get("type") == "Feature":
        geoms = [shape(gj["geometry"])]
    else:
        geoms = [shape(gj)]
    return unary_union(geoms)


def area_km2(geom):
    """Geodesic area (km2, WGS84 ellipsoid) of a lon/lat geometry."""
    from pyproj import Geod
    return abs(Geod(ellps="WGS84").geometry_area_perimeter(geom)[0]) / 1e6


def _num(v, nd=None):
    """float of a table value, None for missing / NaN / empty."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else (round(f, nd) if nd is not None else f)


def _truthy(v):
    if isinstance(v, str):
        return v.strip().lower() not in ("false", "0", "no", "")
    return bool(v) if v == v else True


def cell_category(row):
    """'aoi', 'ring' or 'dropped (water)' (ring cells not used: water share >= water_max)."""
    if str(row.get("role", "ring")) == "aoi":
        return "aoi"
    return "ring" if _truthy(row.get("use", True)) else "dropped (water)"


def cell_counts(cells):
    """Number of cells per category, in the order aoi, ring, dropped (water)."""
    cats = [cell_category(r) for r in cells.to_dict("records")]
    return {c: cats.count(c) for c in CATEGORY_COLORS}


def cells_geojson(cells, epsg):
    """cells.csv (DataFrame: cell, x0, y1, role, use, ...; imported cells may carry their own epsg) ->
    GeoJSON FeatureCollection in EPSG:4326 with properties cell, role, category, dist_m, water."""
    from rasterio.warp import transform
    feats = []
    recs = cells.to_dict("records")
    groups = {}
    for r in recs:
        e = _num(r.get("epsg"))
        groups.setdefault(int(e) if e is not None else int(epsg), []).append(r)
    for e, rs in groups.items():
        xs, ys = [], []
        for r in rs:
            x0, y1 = float(r["x0"]), float(r["y1"])
            xs += [x0, x0 + CELL_M, x0 + CELL_M, x0, x0]
            ys += [y1, y1, y1 - CELL_M, y1 - CELL_M, y1]
        lon, lat = transform(f"EPSG:{e}", "EPSG:4326", xs, ys)
        for i, r in enumerate(rs):
            ring = [[lon[5 * i + k], lat[5 * i + k]] for k in range(5)]
            feats.append(dict(type="Feature", geometry=dict(type="Polygon", coordinates=[ring]), properties=dict(
                cell=str(r["cell"]), role=str(r.get("role", "")), category=cell_category(r),
                dist_m=_num(r.get("dist_m"), 0), water=_num(r.get("water"), 3))))
    return dict(type="FeatureCollection", features=feats)


# ---------------------------------------------------------------------------------------------- folium maps
def view(bounds, width=700, height=500, pad=0.85):
    """Centre (lat, lon) and zoom that show bounds [[s, w], [n, e]] in a width x height px Web Mercator map.
    Used instead of Leaflet's fitBounds, which zooms out to the whole world when the map is built in a hidden
    Streamlit tab (zero size)."""
    (s, w), (n, e) = bounds
    y = lambda lat: math.log(math.tan(math.pi / 4 + math.radians(max(-85.0, min(85.0, lat))) / 2))
    fx = max(e - w, 1e-6) / 360.0
    fy = max(y(n) - y(s), 1e-8) / (2 * math.pi)
    z = math.floor(min(math.log2(pad * width / (256 * fx)), math.log2(pad * height / (256 * fy))))
    return [(s + n) / 2, (w + e) / 2], int(min(18, max(1, z)))


def base_map(bounds=None, location=(20.0, 0.0), zoom=2, height=500):
    """folium map with satellite (shown) and OpenStreetMap base layers; bounds [[s, w], [n, e]] set the view."""
    import folium
    if bounds is not None:
        location, zoom = view(bounds, height=height)
    m = folium.Map(location=list(location), zoom_start=zoom, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=False).add_to(m)
    folium.TileLayer(tiles=SATELLITE[0], attr=SATELLITE[1], name="Satellite").add_to(m)
    return m


def _bounds(geom):
    w, s, e, n = geom.bounds
    return [[s, w], [n, e]]


def study_area_map(aoi, cells=None, epsg=None):
    """Study area outline and, once planned, its cells coloured by category (aoi / ring / dropped (water))."""
    import folium
    from shapely.geometry import mapping
    gj = cells_geojson(cells, epsg) if cells is not None and len(cells) else None
    b = _bounds(aoi)
    if gj is not None:
        c = _bounds_of_features(gj)
        b = [[min(b[0][0], c[0][0]), min(b[0][1], c[0][1])], [max(b[1][0], c[1][0]), max(b[1][1], c[1][1])]]
    m = base_map(b)
    if gj is not None:
        folium.GeoJson(
            gj, name="Cells",
            style_function=lambda f: {"color": CATEGORY_COLORS[f["properties"]["category"]], "weight": 1,
                                      "fillColor": CATEGORY_COLORS[f["properties"]["category"]],
                                      "fillOpacity": 0.25},
            tooltip=folium.GeoJsonTooltip(["cell", "category", "dist_m", "water"],
                                          aliases=["cell", "role", "distance (m)", "water share"]),
        ).add_to(m)
    folium.GeoJson(mapping(aoi), name="Study area", style_function=lambda f: AOI_STYLE).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    return m


def _bounds_of_features(gj):
    xy = np.array([p for f in gj["features"] for p in f["geometry"]["coordinates"][0]])
    return [[xy[:, 1].min(), xy[:, 0].min()], [xy[:, 1].max(), xy[:, 0].max()]]


def draw_map():
    """World map with the Draw plugin (polygon and rectangle only) and a place search."""
    import folium
    from folium import plugins
    m = base_map()
    plugins.Draw(export=False, position="topleft",
                 draw_options=dict(polyline=False, circle=False, marker=False, circlemarker=False,
                                   polygon=dict(allowIntersection=False, showArea=True), rectangle=True),
                 edit_options=dict(edit=False)).add_to(m)
    try:
        plugins.Geocoder(collapsed=True, add_marker=False, position="topright").add_to(m)
    except (AttributeError, TypeError):
        pass
    folium.LayerControl(collapsed=True).add_to(m)
    return m


LEGEND_CSS = ("<style>.legend.leaflet-control{background:rgba(255,255,255,0.92);padding:6px 8px 0 0;"
              "border-radius:4px;box-shadow:0 1px 4px rgba(0,0,0,0.3)}.legend .tick text{font-size:12px}"
              ".legend .caption{font-size:13px;font-weight:600}</style>")


def results_map(layers, vmax, aoi=None, show=1, opacity=0.85, height=600, caption="Canopy height (m)", vmin=0.0):
    """Canopy-height overlays [(name, png data url, bounds)] on one vmin..vmax scale: the first `show` layers are
    visible, the others can be switched on in the layer control; colour bar legend titled `caption` (the app adds
    the input representation) on a white panel, readable over the satellite image; study-area outline."""
    import branca.colormap as bcm
    import folium
    from folium.raster_layers import ImageOverlay
    from shapely.geometry import mapping
    bounds = _bounds(aoi) if aoi is not None else (layers[0][2] if layers else None)
    m = base_map(bounds, height=height)
    for i, (name, url, b) in enumerate(layers):
        ImageOverlay(image=url, bounds=b, name=name, opacity=opacity, show=i < show, interactive=False,
                     cross_origin=False, zindex=1).add_to(m)
    if aoi is not None:
        folium.GeoJson(mapping(aoi), name="Study area", style_function=lambda f: AOI_STYLE).add_to(m)
    cm = bcm.LinearColormap(colormap_hex(), vmin=float(vmin), vmax=float(vmax), caption=caption)
    cm.width = 500
    cm.add_to(m)
    m.get_root().header.add_child(folium.Element(LEGEND_CSS))
    folium.LayerControl(collapsed=False, position="bottomright").add_to(m)
    return m
