"""Raw Sentinel-1 inputs of the local S1 way (project setting s1_method: local; the chain is tool.s1_local).

S1Region  the acquisitions of the S1 window over the cells of a download run, their metadata (filter set D_i and
          heading: they depend on the acquisition only, so one request serves every cell) and the grid and
          footprint of every scene involved (acquisitions and filter-set members). Kept in a JSON file of the
          project (raw/s1_local_<year>[_seasonal].json) and extended when cells are added.
fetch_cell  the scenes of one cell (acquisitions whose footprint touches the cell, filter-set members that touch
          the cell + halo), fetched as band stacks on a few grids: scenes of the cell's CRS on the analysis grid +
          halo, scenes of another UTM zone on a 10 m grid of their CRS around the cell. S1 grids are 10 m north-up
          grids of a UTM zone, so Earth Engine's nearest resampling onto a 10 m grid of the same CRS is an integer
          shift and returns the scene pixels unchanged; the angle and footprint bands come on the analysis grid (the
          chain reads the angle by nearest from its coarse grid and clips on the requested grid). Footprints are
          compared in longitude / latitude with a margin of SLACK_DEG, so a few scenes that only come close to the
          cell may be fetched as well: they hold no valid pixel in the cell and change nothing.
NotApplicable  the cell is not on a WGS84 UTM grid (the tool's default CRS is), a scene grid is not a 10 m
          north-up UTM grid, or a grid offset is an exact half pixel (ambiguous nearest): the caller falls back to
          the Earth Engine way for that cell.
Cost (31 acquisitions / year, 40 scenes per cell): about 10 EECU-s per cell billed (the Earth Engine way:
about 140) and ~48 MB transferred per cell; the metadata about 0.6 EECU-s per acquisition, once.
"""
import json
import math
import os
import threading
import time

import numpy as np

from . import io, layers as L
from ..project import replace_retry
from ..s1_local import HALO, CellS1

SLACK_DEG = 0.03             # footprint test margin (degrees): planar vs geodesic footprint edges, a few km
META_BATCH = 25              # acquisitions per metadata request
INFO_BATCH = 150             # scenes per grid / footprint request
REQUEST_BYTES = 40_000_000   # per computePixels request (Earth Engine's limit is 48 MiB)
FORMAT = 1                   # version of the JSON file


class NotApplicable(Exception):
    """The local way cannot reproduce the Earth Engine way for this cell; use the Earth Engine way."""


def utm_wgs84(epsg):
    e = int(epsg)
    return 32601 <= e <= 32660 or 32701 <= e <= 32760


def cell_lonlat(epsg, x0, y1, W, H, res=10.0, pad_m=0.0, n=16):
    """The cell rectangle (+ pad_m) as a lon / lat shapely polygon, edges densified."""
    from pyproj import Transformer
    from shapely.geometry import Polygon
    xa, xb, ya, yb = x0 - pad_m, x0 + W * res + pad_m, y1 - H * res - pad_m, y1 + pad_m
    t = np.linspace(0, 1, n + 1)
    xs = np.concatenate([xa + (xb - xa) * t, np.full(n + 1, xb), xb - (xb - xa) * t, np.full(n + 1, xa)])
    ys = np.concatenate([np.full(n + 1, yb), yb - (yb - ya) * t, np.full(n + 1, ya), ya + (yb - ya) * t])
    lo, la = Transformer.from_crs(int(epsg), 4326, always_xy=True).transform(xs, ys)
    return Polygon(zip(lo, la))


class S1Region:
    """Acquisition metadata and scene grids of one S1 window (annual or seasonal) of a project, see the module."""

    def __init__(self, path, year, seasonal):
        self.path, self.year, self.seasonal = str(path), int(year), bool(seasonal)
        self.start, self.end = L.s1_window(self.year, self.seasonal)
        self.acq, self.scenes, self.cells = {}, {}, set()
        self._shapes = {}
        self._lock = threading.Lock()
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as fh:
                d = json.load(fh)
            if d.get("format") == FORMAT and d.get("window") == [self.start, self.end]:
                self.acq, self.scenes, self.cells = d["acq"], d["scenes"], set(d["cells"])

    def save(self):
        tmp = f"{self.path}.{os.getpid()}.tmp"
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(dict(format=FORMAT, window=[self.start, self.end], acq=self.acq, scenes=self.scenes,
                           cells=sorted(self.cells)), fh)
        replace_retry(tmp, self.path)

    @staticmethod
    def key(epsg, x0, y1, W, H):
        return f"{int(epsg)}:{float(x0):.1f}:{float(y1):.1f}:{int(W)}:{int(H)}"

    def ensure(self, cells, log=print):
        """Metadata of every acquisition over the cells [(epsg, x0, y1, W, H)] and the grids / footprints of their
        scenes; only what the file lacks is requested. Earth Engine must be initialised."""
        from shapely.geometry import mapping
        from shapely.ops import unary_union
        todo = [c for c in cells if self.key(*c) not in self.cells]
        if not todo:
            return 0
        t0 = time.time()
        area = unary_union([cell_lonlat(*c, pad_m=0.0) for c in todo]).buffer(0.002).simplify(0.001)
        region = L.ee.Geometry(json.loads(json.dumps(mapping(area))), None, False)
        idx = io.retry(lambda: L.s1_acquisitions(region, self.start, self.end))
        pre = "COPERNICUS/S1_GRD_FLOAT/"
        new = [i for i in idx if pre + i not in self.acq]
        log(f"S1 local: {len(idx)} acquisitions over {len(todo)} cells in [{self.start}, {self.end}), "
            f"{len(new)} without metadata yet")
        for k in range(0, len(new), META_BATCH):
            for m in io.retry(lambda b=new[k:k + META_BATCH]: L.s1_metadata_of(b, self.start, self.end)):
                self.acq[m["id"]] = {"t": m["t"], "D": m["D"], "heading": m["heading"]}
            log(f"S1 local: metadata {min(k + META_BATCH, len(new))}/{len(new)}")
        need = sorted({s for a in self.acq.values() for s in a["D"]} | set(self.acq))
        need = [s for s in need if s not in self.scenes]
        for k in range(0, len(need), INFO_BATCH):
            self.scenes.update(io.retry(lambda b=need[k:k + INFO_BATCH]: L.s1_scene_info(b)))
        self.cells |= {self.key(*c) for c in todo}
        self.save()
        log(f"S1 local: metadata of {len(new)} acquisitions and grids of {len(need)} scenes in "
            f"{time.time() - t0:.0f} s")
        return len(new)

    def _shape(self, sid):
        with self._lock:
            if sid not in self._shapes:
                from shapely.geometry import shape
                from shapely.prepared import prep
                self._shapes[sid] = prep(shape(self.scenes[sid]["foot"]))
            return self._shapes[sid]

    def select(self, epsg, x0, y1, W, H, halo=HALO, res=10.0, seasons=None):
        """(metadata of the cell's acquisitions sorted by time, ids of the scenes to fetch). seasons: only the
        acquisitions of these seasons (seasonal window)."""
        from ..s1_local import in_season
        cell = cell_lonlat(epsg, x0, y1, W, H, res).buffer(SLACK_DEG)
        ring = cell_lonlat(epsg, x0, y1, W, H, res, pad_m=(halo + 2) * res).buffer(SLACK_DEG)
        meta = [dict(id=i, **a) for i, a in self.acq.items() if self._shape(i).intersects(cell)
                and (seasons is None or any(in_season(a["t"], k) for k in seasons))]
        meta.sort(key=lambda m: (m["t"], m["id"]))
        ids = {m["id"] for m in meta}
        ids |= {d for m in meta for d in m["D"] if d in self.scenes and self._shape(d).intersects(ring)}
        return meta, sorted(ids)


# ================================================================================================ fetch
def _grid_for(crs, cell, margin_px=3):
    """(transform, w, h) of a 10 m north-up grid of crs covering the cell's analysis grid + halo."""
    if crs == f"EPSG:{cell.epsg}":
        return cell.transform, cell.m, cell.m
    from pyproj import Transformer
    tr = Transformer.from_crs(f"EPSG:{cell.epsg}", crs, always_xy=True)
    xa, yb = cell.gx0, cell.gy1
    xb, ya = xa + cell.m * cell.res, yb - cell.m * cell.res
    t = np.linspace(0, 1, 33)
    xs = np.concatenate([xa + (xb - xa) * t, np.full(33, xb), xb - (xb - xa) * t, np.full(33, xa)])
    ys = np.concatenate([np.full(33, yb), yb - (yb - ya) * t, np.full(33, ya), ya + (yb - ya) * t])
    X, Y = tr.transform(xs, ys)
    r = 10.0
    gx = math.floor(min(X) / r - margin_px) * r
    gy = math.ceil(max(Y) / r + margin_px) * r
    w = int(math.ceil(max(X) / r + margin_px) - math.floor(min(X) / r - margin_px))
    h = int(math.ceil(max(Y) / r + margin_px) - math.floor(min(Y) / r - margin_px))
    return [r, 0.0, gx, 0.0, -r, gy], w, h


def _window(t_scene, t_grid):
    """Affine of the scene-grid window that a same-CRS 10 m grid holds after nearest resampling (integer shift)."""
    a, b, ox, d, e, oy = [float(x) for x in t_scene]
    if (a, b, d, e) != (10.0, 0.0, 0.0, -10.0):
        raise NotApplicable(f"scene grid {t_scene} is not 10 m north-up")
    fx, fy = (t_grid[2] - ox) / 10.0, (oy - t_grid[5]) / 10.0
    for f in (fx, fy):
        if abs(abs(f - math.floor(f)) - 0.5) < 1e-6:
            raise NotApplicable(f"scene grid offset {f:.7f} px is a half pixel (ambiguous nearest)")
    C, R = int(round(fx)), int(round(fy))
    return [10.0, 0.0, ox + 10.0 * C, 0.0, -10.0, oy - 10.0 * R]


def _request(images, crs, transform, w, h):
    """Fetch a list of (single-band image, band name) on one grid, split into requests below REQUEST_BYTES (and
    further on TooBig) -> ({band: array}, number of requests)."""
    ee = L.ee
    sizes = [(im, 1 if name.startswith(("m_", "foot_", "tv_")) else 4) for im, name in images]
    parts, cur, nb = [], [], 0
    for im, bpp in sizes:
        if cur and nb + bpp * w * h > REQUEST_BYTES:
            parts.append(cur)
            cur, nb = [], 0
        cur.append(im)
        nb += bpp * w * h
    if cur:
        parts.append(cur)
    out, n = {}, [0]

    def get(ims):
        try:
            a = io.compute_pixels(ee.Image.cat(ims), crs, transform, w, h)
            n[0] += 1
            out.update({b: np.array(a[b]) for b in a.dtype.names})
        except io.TooBig:
            if len(ims) == 1:
                raise
            get(ims[:len(ims) // 2])
            get(ims[len(ims) // 2:])
    for p in parts:
        get(p)
    return out, n[0]


def _cell_band_list(ids, main):
    """(image, name) of the angle and footprint bands of the acquisitions, fetched on the analysis grid."""
    out = []
    for k, s in enumerate(ids):
        if s in main:
            a, f = L.s1_cell_bands(s, k)
            out += [(a, f"angle_{k}"), (f, f"foot_{k}")]
    return out


def _add_cell_bands(cell, ids, main, got):
    for k, s in enumerate(ids):
        if s in main:
            cell.add_angle(s, got[f"angle_{k}"], got[f"foot_{k}"])


def fetch_cell(region, epsg, x0, y1, W, H, halo=HALO, res=10.0, seasons=None):
    """Raw inputs of one cell -> (CellS1, metadata of its acquisitions, info dict: scenes, requests, MB).
    Raises NotApplicable when the local way does not apply (see the module)."""
    if not utm_wgs84(epsg):
        raise NotApplicable(f"EPSG:{epsg} is not a WGS84 UTM zone")
    if int(W) != int(H):
        raise NotApplicable(f"cell {W} x {H} px is not square")
    meta, ids = region.select(epsg, x0, y1, W, H, halo, res, seasons)
    cell = CellS1(epsg, x0, y1, n=int(W), halo=halo, res=res)
    if not meta:
        return cell, meta, dict(scenes=0, requests=0, mb=0.0)
    main = {m["id"] for m in meta}
    groups = {}
    for k, s in enumerate(ids):
        crs = region.scenes[s]["crs"]
        if not crs.startswith("EPSG:") or not utm_wgs84(crs[5:]):
            raise NotApplicable(f"scene {s} is in {crs}, not a WGS84 UTM zone")
        groups.setdefault(crs, []).append((k, s))
    n_req, nbytes = 0, 0
    cell_crs = f"EPSG:{int(epsg)}"
    for crs, members in sorted(groups.items(), key=lambda g: g[0] != cell_crs):
        t, w, h = _grid_for(crs, cell)
        wins = {s: _window(region.scenes[s]["t"], t) for _, s in members}
        imgs = []
        for k, s in members:
            bands = L.s1_raw_bands(s, k, s in main)
            imgs += [(bands[0].select(f"VV_{k}"), f"VV_{k}"), (bands[0].select(f"VH_{k}"), f"VH_{k}"),
                     (bands[1], f"m_{k}")] + ([(bands[2], f"elev_{k}"), (bands[3], f"tv_{k}")] if s in main else [])
        if crs == cell_crs:
            imgs += _cell_band_list(ids, main)
        got, nr = _request(imgs, crs, t, w, h)
        n_req += nr
        nbytes += sum(v.nbytes for v in got.values())
        for k, s in members:
            cell.add_scene(s, crs, wins[s], got[f"VV_{k}"], got[f"VH_{k}"], got[f"m_{k}"], got.get(f"elev_{k}"),
                           got.get(f"tv_{k}"))
        if crs == cell_crs:
            _add_cell_bands(cell, ids, main, got)
    if cell_crs not in groups:                      # every scene in another zone: the angles still come here
        got, nr = _request(_cell_band_list(ids, main), cell_crs, cell.transform, cell.m, cell.m)
        n_req += nr
        nbytes += sum(v.nbytes for v in got.values())
        _add_cell_bands(cell, ids, main, got)
    return cell, meta, dict(scenes=len(ids), requests=n_req, mb=nbytes / 1e6, zones=len(groups))
