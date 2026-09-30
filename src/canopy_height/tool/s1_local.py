"""Sentinel-1 composites computed locally from the raw scenes (project setting s1_method: local).

The Earth Engine way (gee.layers.s1 / s1_seasonal) runs the whole gee_s1_ard chain on Earth Engine; its 15 x 15
speckle-filter convolutions are most of the tool's compute quota. Here Earth Engine only serves pixels
(gee.s1_raw): every scene's VV / VH and validity on its own 10 m grid, the SRTM elevation resampled onto that grid
as the chain's terrain flattening does, the footprint and the 'angle' band; this module runs the same chain in
numpy on the analysis grid of a cell plus a halo:

  1. f_mask_edges      dB round trip, angle outside (30.63993, 45.23993) masked (float32 like Earth Engine)
  2. Quegan filter     boxcar(image) / count * sum over the filter set D of raw / boxcar(raw)
  3. terrain           VOLUME model on SRTM (bilinear onto the scene grid), slope / aspect by ee.Terrain
  4. dB, median        of the acquisitions of the year (or of a season)

Earth Engine's semantics that matter (measured on synthetic images and step by step against the chain's
intermediates):
  nearest      the source pixel containing the target pixel centre (round of the fractional index); the main
               acquisition is read unresampled (nearest), the filter-set members with .resample() (bilinear)
  bilinear     sum over the valid neighbours of w * v / sum of their w; mask = sum of w * mask (8 bit)
  convolve     Kernel.square(7.5, normalize) = 15 x 15 weights 1/225; masked pixels count as 0 (no
               renormalisation); the output mask is the centre pixel's
  count / sum  count = images with mask > 0; sum = sum of value * mask (fractional masks weight)
  divide       x / 0 = 0 (unmasked)
  mask(m)      replaces the mask by m; pixels unmasked that way hold 0
  log10        of 0 (or less) is masked: the dB round trip of f_mask_edges drops valid zero pixels
  median       exact up to 64 valid values (even count: mean of the middle two); above that the median of the
               values replaced by the means of their histogram buckets (ee_median; the annual composite of areas with
               two orbits and two satellites, e.g. 2019 in Guatemala with 109-173 acquisitions, is in that regime)
  ee.Terrain   works on the requested grid (the scene-grid DEM arrives by nearest resampling): central
               differences of the 4 neighbours over their haversine distances on a 6371008.8 m sphere; aspect is
               the downslope bearing minus the mean meridian convergence of the grid axes
The result is not bit-identical to the Earth Engine way (float32 / float64 rounding): identical NaN masks, about
1e-5 dB per raster, at most ~1e-2 dB at a few pixels (layover edges where the terrain factor is in the thousands, values
on a median histogram-bucket edge); < 0.01 % of the model's uint16 S1 codes change, by one.
"""
import math
import warnings

import numpy as np
from scipy.ndimage import uniform_filter

ANG_LO, ANG_HI = 30.63993, 45.23993          # border_noise_correction.maskAngGT30 / maskAngLT452
KERNEL = 15                                   # speckle_filter.boxcar(KERNEL_SIZE = 15): Kernel.square(7.5)
BANDS = ("VV", "VH")
HALO = 10                                     # px around the cell: boxcar 7 + bilinear 1 + terrain 1, rounded up
R_EARTH = 6371008.8                           # ee.Terrain neighbour distances (haversine), fitted
MAX_RAW, MAX_BUCKETS = 64, 256                # ImageCollection.median(): exact up to 64 values, then a histogram
SEASON_MONTHS = [(12, 2), (3, 5), (6, 8), (9, 11)]      # = gee.layers.SEASON_MONTHS (DJF, MAM, JJA, SON)


def in_season(t_ms, season):
    """gee.layers.in_season: does an acquisition time (ms, UTC) fall in the months of season 0..3."""
    import datetime as dt
    m = dt.datetime.fromtimestamp(t_ms / 1000, tz=dt.timezone.utc).month
    t1, t2 = SEASON_MONTHS[season]
    return t1 <= m <= t2 if t1 <= t2 else (m >= t1 or m <= t2)


# ================================================================================================ resampling
def centres(x0, y1, nx, ny, res=10.0):
    """Pixel-centre coordinates (X, Y) [ny, nx] of a north-up grid with upper-left corner (x0, y1)."""
    j, i = np.mgrid[0:ny, 0:nx]
    return x0 + (i + 0.5) * res, y1 - (j + 0.5) * res


def frac_index(t, X, Y):
    """Fractional (col, row) of points in the pixel-centre convention of the affine t = [a, b, c, d, e, f]."""
    a, b, c, d, e, f = t
    inv = np.linalg.inv(np.array([[a, b], [d, e]], float))
    dx, dy = X - c, Y - f
    return inv[0, 0] * dx + inv[0, 1] * dy - 0.5, inv[1, 0] * dx + inv[1, 1] * dy - 0.5


def _take(a, r, c, fill):
    ok = (r >= 0) & (r < a.shape[0]) & (c >= 0) & (c < a.shape[1])
    out = np.full(r.shape, fill, dtype=np.float64)
    out[ok] = a[r[ok], c[ok]]
    return out


def nearest(val, mask, t, X, Y):
    """(value, mask) of val / mask (grid t) at the points X, Y by nearest resampling; outside -> masked."""
    u, v = frac_index(t, X, Y)
    c, r = np.round(u).astype(np.int64), np.round(v).astype(np.int64)
    return _take(val, r, c, 0.0), _take(mask, r, c, 0.0)


def bilinear(val, mask, t, X, Y):
    """Earth Engine bilinear: renormalised over the valid neighbours; mask = sum(w * mask) in 1/255 steps."""
    u, v = frac_index(t, X, Y)
    c0, r0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    fu, fv = u - c0, v - r0
    num, den, msum = np.zeros(X.shape), np.zeros(X.shape), np.zeros(X.shape)
    for w, dr, dc in (((1 - fu) * (1 - fv), 0, 0), (fu * (1 - fv), 0, 1), ((1 - fu) * fv, 1, 0), (fu * fv, 1, 1)):
        m = _take(mask, r0 + dr, c0 + dc, 0.0)
        ok = m > 0
        num += np.where(ok, w * _take(val, r0 + dr, c0 + dc, 0.0), 0.0)
        den += np.where(ok, w, 0.0)
        msum += w * m
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0), np.round(msum * 255.0) / 255.0


def boxcar(val, mask):
    """convolve(Kernel.square(7.5, normalize)): masked pixels -> 0, sum / 225 (zero outside the array)."""
    return uniform_filter(np.where(mask > 0, val, 0.0), size=KERNEL, mode="constant", cval=0.0)


def safe_div(a, b):
    """Earth Engine divide: x / 0 = 0."""
    return np.where(b != 0, a / np.where(b != 0, b, 1.0), 0.0)


def ee_median(stack):
    """ImageCollection.median() of a stack [N, ...] (NaN = masked) as Earth Engine computes it: the exact median of
    up to MAX_RAW valid values; above that a histogram of power-of-2 buckets (aligned to multiples of the width, the
    smallest width that needs at most MAX_BUCKETS buckets) that keeps each bucket's mean, and the median of the values
    replaced by their bucket means (even count: mean of the two middle ones). Measured on 2026-09-29 (33 value sets
    of 64-300 values: all equal to Earth Engine's result); order-independent."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmedian(stack, axis=0)
        cnt = np.isfinite(stack).sum(axis=0)
        big = cnt > MAX_RAW
        if not big.any():
            return out
        v = stack[:, big]
        c = cnt[big]
        lo, hi = np.nanmin(v, axis=0), np.nanmax(v, axis=0)
        w = np.exp2(np.ceil(np.log2(np.maximum((hi - lo) / MAX_BUCKETS, 1e-300))))
        for _ in range(3):                                   # floor-aligned buckets may need one more doubling
            over = np.floor(hi / w) - np.floor(lo / w) + 1 > MAX_BUCKETS
            w = np.where(over, 2 * w, w)
        s = np.sort(v, axis=0)                               # NaN last
        cols = np.arange(v.shape[1])
        idx = np.floor(v / w)

        def bucket_mean(x):
            k = np.floor(x / w)
            return np.nanmean(np.where(idx == k, v, np.nan), axis=0)
        out[big] = (bucket_mean(s[(c - 1) // 2, cols]) + bucket_mean(s[c // 2, cols])) / 2
        return out


def terrain_geometry(epsg, X, Y, res=10.0):
    """Per pixel of a north-up grid of EPSG:epsg: half the haversine distance between the E / W and the N / S
    neighbour centres (sx, sy, m) and the rotation ee.Terrain.aspect subtracts from the grid aspect (degrees): the
    mean of the true bearings of grid north and of grid east - 90 (fitted against ee.Terrain on SRTM:
    slope within 1e-5 deg, aspect within 2e-5 deg)."""
    from pyproj import Transformer
    tr = Transformer.from_crs(int(epsg), 4326, always_xy=True)

    def lonlat(x, y):
        lo, la = tr.transform(x, y)
        return np.radians(lo), np.radians(la)

    def hav(p, q):
        (l1, f1), (l2, f2) = p, q
        a = np.sin((f2 - f1) / 2) ** 2 + np.cos(f1) * np.cos(f2) * np.sin((l2 - l1) / 2) ** 2
        return 2 * R_EARTH * np.arcsin(np.sqrt(a))

    def bearing(p, q):
        (l1, f1), (l2, f2) = p, q
        return np.degrees(np.arctan2(np.sin(l2 - l1) * np.cos(f2),
                                     np.cos(f1) * np.sin(f2) - np.sin(f1) * np.cos(f2) * np.cos(l2 - l1)))

    E, W, N, S = lonlat(X + res, Y), lonlat(X - res, Y), lonlat(X, Y + res), lonlat(X, Y - res)
    return hav(W, E) / 2, hav(S, N) / 2, (bearing(S, N) + bearing(W, E) - 90.0) / 2


def terrain(elev, valid, geo):
    """ee.Terrain.slope / aspect (degrees) of a DEM on the analysis grid: 4-connected central differences over the
    haversine neighbour distances; aspect = downslope bearing (0 = north, clockwise) minus geo's rotation; masked
    where the pixel or one of its 4 neighbours is masked. -> (slope, aspect, valid)."""
    sx, sy, rot = geo
    zp = np.pad(np.where(valid, elev, np.nan), 1, constant_values=np.nan)
    gx = (zp[1:-1, 2:] - zp[1:-1, :-2]) / 2 / sx          # east - west
    gy = (zp[:-2, 1:-1] - zp[2:, 1:-1]) / 2 / sy          # north - south
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspect = (np.degrees(np.arctan2(-gx, -gy)) - rot) % 360.0
    ok = np.isfinite(slope) & valid
    return np.where(ok, slope, 0.0), np.where(ok, aspect, 0.0), ok


# ================================================================================================ chain
class CellS1:
    """Raw S1 inputs of one cell and the chain on its analysis grid (cell + halo px on every side).

    add_scene: a scene's window of its own grid (crs, affine t: a 10 m north-up grid), VV, VH, validity, and for
    the cell's acquisitions also elev (SRTM bilinear onto that grid) and tv (ee.Terrain's validity there: the
    terrain algorithms run on the scene grid). add_angle: an acquisition's angle band (nearest, -9999 = masked)
    and footprint (1 = inside: the chain clips the aspect again on the requested grid) on the analysis grid
    (+ halo). Scenes of another CRS than the cell's are resampled through
    pyproj (the Earth Engine way resamples them the same way).
    """

    def __init__(self, epsg, x0, y1, n=256, halo=HALO, res=10.0):
        self.epsg, self.x0, self.y1, self.n, self.halo, self.res = int(epsg), float(x0), float(y1), n, halo, res
        self.gx0, self.gy1 = self.x0 - halo * res, self.y1 + halo * res
        self.m = n + 2 * halo
        self.X, self.Y = centres(self.gx0, self.gy1, self.m, self.m, res)
        self.scenes, self.angle = {}, {}
        self._xy, self._ratio, self._geo = {f"EPSG:{self.epsg}": (self.X, self.Y)}, {}, None

    @property
    def transform(self):
        """Affine of the analysis grid + halo."""
        return [self.res, 0.0, self.gx0, 0.0, -self.res, self.gy1]

    @property
    def geo(self):
        if self._geo is None:
            self._geo = terrain_geometry(self.epsg, self.X, self.Y, self.res)
        return self._geo

    def add_scene(self, sid, crs, t, VV, VH, valid, elev=None, tv=None):
        self.scenes[sid] = dict(crs=crs, t=[float(x) for x in t], VV=VV, VH=VH, m=valid, elev=elev, tv=tv)

    def add_angle(self, sid, a, foot):
        for name, x in (("angle", a), ("footprint", foot)):
            if x.shape != self.X.shape:
                raise ValueError(f"{name} of {sid}: shape {x.shape}, expected {self.X.shape}")
        self.angle[sid] = (a, foot)

    def xy(self, crs):
        """Analysis-grid pixel centres in another CRS."""
        if crs not in self._xy:
            from pyproj import Transformer
            tr = Transformer.from_crs(f"EPSG:{self.epsg}", crs, always_xy=True)
            self._xy[crs] = tuple(np.asarray(v).reshape(self.X.shape) for v in tr.transform(self.X, self.Y))
        return self._xy[crs]

    def crop(self, a):
        h = self.halo
        return a[..., h:h + self.n, h:h + self.n]

    # --- per scene on the analysis grid
    def _resample(self, sid, name, how):
        s = self.scenes[sid]
        X, Y = self.xy(s["crs"])
        mask = s["m"] if name in BANDS else np.ones(s[name].shape)
        return how(np.asarray(s[name], np.float64), np.asarray(mask, np.float64), s["t"], X, Y)

    def ratio(self, sid, b):
        """(ratio * weight, weight > 0) of a filter-set member: raw_bil / boxcar(raw_bil), weighted by its mask
        (ImageCollection sum); a member without pixels here contributes nothing."""
        k = (sid, b)
        if k not in self._ratio:
            if sid not in self.scenes:
                return None
            v, w = self._resample(sid, b, bilinear)
            r = safe_div(np.where(w > 0, v, 0.0), boxcar(v, w))
            self._ratio[k] = (r * w, w > 0)
        return self._ratio[k]

    def acquisition(self, m, debug=False):
        """dB VV, VH [m, m] (NaN = masked) of acquisition m ({'id', 'D', 'heading', ...}) = one image of
        layers.s1 before the median; debug -> (out, intermediates)."""
        sid = m["id"]
        ang = np.asarray(self.angle[sid][0], np.float64)
        edge_ok = (ang != -9999.0) & (ang > ANG_LO) & (ang < ANG_HI)       # f_mask_edges: every band, angle too
        dbg = {"angle": ang, "edge_ok": edge_ok}
        q, q_ok = {}, {}
        for b in BANDS:
            v, mk = self._resample(sid, b, nearest)
            ok = (mk > 0) & edge_ok & (v > 0)                # log10 of 0 is masked in the dB round trip
            with np.errstate(divide="ignore"):
                db = np.float32(10) * np.log10(np.where(ok, v, 0.0).astype(np.float32))
                lin = np.where(ok, np.power(np.float32(10), db / np.float32(10)).astype(np.float64), 0.0)
            cnt, isum = np.zeros(self.X.shape), np.zeros(self.X.shape)
            for d in m["D"]:
                rw = self.ratio(d, b)
                if rw is not None:
                    isum += rw[0]
                    cnt += rw[1]
            filt = boxcar(lin, ok)
            q[b] = safe_div(filt, cnt) * isum
            q_ok[b] = ok                      # boxcar mask = centre mask; count / sum are valid wherever it is
            if debug:
                dbg.update({f"fme_{b}": lin, f"filt_{b}": filt, f"cnt_{b}": cnt, f"isum_{b}": isum, f"q_{b}": q[b]})
        # terrain flattening (terrain_flattening._correct, VOLUME, heading from the metadata)
        # ee.Terrain runs on the scene grid (clip mask there, pixel + 4 neighbours: tv) and arrives by nearest; the
        # aspect is clipped again on the requested grid; phi_s is 0 where the aspect is masked (unmask)
        elev, _ = self._resample(sid, "elev", nearest)
        slope, aspect, ok_g = terrain(elev, np.ones(elev.shape, bool), self.geo)
        t_ok = (self._resample(sid, "tv", nearest)[0] > 0) if self.scenes[sid]["tv"] is not None else ok_g
        a_ok = t_ok & (np.asarray(self.angle[sid][1]) > 0)
        theta = np.radians(ang)
        phi_s = -np.radians(np.where(a_ok & (aspect <= 180), aspect, 0.0)
                            + np.where(a_ok & (aspect > 180), aspect - 360.0, 0.0))
        alpha_r = np.arctan(np.tan(np.radians(slope)) * np.cos(math.radians(m["heading"]) - phi_s))
        ninety = math.pi / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            scf = np.tan(ninety - theta + alpha_r) / np.tan(ninety - theta)
            keep = (alpha_r < theta) & (alpha_r > -(ninety - theta)) & t_ok & edge_ok    # layover, shadow
            out = {}
            for b in BANDS:
                g = np.where(q_ok[b], q[b] / np.cos(theta) * scf, 0.0)      # mask(m): unmasked pixels hold 0
                ok_db = keep & (g > 0)                                  # log10(0): masked, not -inf
                out[b] = np.where(ok_db, 10.0 * np.log10(np.where(ok_db, g, 1.0)), np.nan)
        if debug:
            dbg.update({"slope": slope, "aspect": aspect, "t_ok": t_ok, "phi_s": phi_s, "alpha_r": alpha_r, "scf": scf,
                        "keep": keep})
            return out, dbg
        return out

    def per_acquisition(self, meta):
        """{id: dB [2, n, n]} of the acquisitions of meta, cropped to the cell."""
        out = {}
        for m in meta:
            o = self.acquisition(m)
            out[m["id"]] = np.stack([self.crop(o[b]) for b in BANDS])
        return out

    def median(self, arrs):
        """Earth Engine's median of the valid values (ee_median) -> float64 [2, n, n] (NaN = no data; no
        acquisition -> all NaN)."""
        if not arrs:
            return np.full((len(BANDS), self.n, self.n), np.nan)
        return ee_median(np.stack(arrs))

    def annual(self, meta, acq=None):
        """layers.s1: median dB of the acquisitions of meta -> ([2, n, n], acquisitions with data here)."""
        acq = acq if acq is not None else self.per_acquisition(meta)
        arrs = [acq[m["id"]] for m in meta]
        return self.median(arrs), sum(bool(np.isfinite(a).any()) for a in arrs)

    def seasonal(self, meta, season, acq=None):
        """layers.s1_seasonal: median dB of the acquisitions of meta in season 0..3 -> ([2, n, n], count)."""
        sel = [m for m in meta if in_season(m["t"], season)]
        acq = acq if acq is not None else self.per_acquisition(sel)
        arrs = [acq[m["id"]] for m in sel]
        return self.median(arrs), sum(bool(np.isfinite(a).any()) for a in arrs)
