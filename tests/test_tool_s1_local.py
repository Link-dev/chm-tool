"""Canopy-height tool: Sentinel-1 the local way (s1_method: local) without Earth Engine.

- Earth Engine semantics: tests/data/s1_gee_synthetic.npz holds a synthetic image (binary and fractional masks) on a
  10 m grid offset by (0.67, 0.61) px and what Earth Engine made of it on the analysis grid (nearest, bilinear,
  Kernel.square(7.5) convolve, their ratio; probe run on 2026-09-29): tool.s1_local must give the same.
- ee.Terrain's slope / aspect geometry, and the chain on synthetic scenes (constant backscatter, flat and tilted
  terrain, the angle mask, a scene of the neighbouring UTM zone, the median of an even number of acquisitions).
- gee.s1_raw: the integer-shift window, the per-cell scene selection and the metadata file (offline).
- download.export: the local way writes the Earth Engine way's files (tag s1_method=local; the four seasons from
  one fetch) and falls back to the Earth Engine way where the local way does not apply.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("pyproj")
pytest.importorskip("shapely")

from canopy_height.tool import s1_local as S  # noqa: E402

DATA = Path(__file__).parent / "data" / "s1_gee_synthetic.npz"
EPSG, X0, Y1 = 32650, 584670.0, 550110.0


# ---------------------------------------------------------------------------------------------- Earth Engine semantics
def test_resampling_and_boxcar_match_earth_engine():
    D = np.load(DATA)
    TP, TG = D["TP"], D["TG"]
    X, Y = S.centres(TG[2], TG[5], 40, 40)
    inner = (slice(7, -7), slice(7, -7))
    for n in "ABC":                                  # A, C: binary masks; B: fractional mask
        v, m = D[f"P_{n}"], D[f"P_{n}_m"]
        nv, nm = S.nearest(v, m, TP, X, Y)
        assert np.array_equal(np.where(nm > 0, nv, -9999), D[f"G_{n}_nn"]) and np.array_equal(nm, D[f"G_{n}_nn_m"])
        bv, bm = S.bilinear(v, m, TP, X, Y)
        ok = D[f"G_{n}_bil_m"] > 0
        assert np.abs(bv - D[f"G_{n}_bil"])[ok].max() < 1e-9 and np.abs(bm - D[f"G_{n}_bil_m"]).max() < 1e-6
        for src, (sv, sm) in (("nn", (nv, nm)), ("bil", (bv, bm))):
            conv = S.boxcar(sv, sm)
            okc = D[f"G_{n}_{src}_conv_m"][inner] > 0
            assert np.abs(conv - D[f"G_{n}_{src}_conv"])[inner][okc].max() < 1e-9
            assert np.allclose(D[f"G_{n}_{src}_conv_m"][inner], sm[inner], atol=1e-6)        # mask = centre mask
        ratio = S.safe_div(np.where(bm > 0, bv, 0), S.boxcar(bv, bm))
        okr = D[f"G_{n}_ratio_m"][inner] > 0
        assert np.abs(ratio - D[f"G_{n}_ratio"])[inner][okr].max() < 1e-9
    # ImageCollection count / sum / median of the three bilinear images; divide by zero
    bl = [S.bilinear(D[f"P_{n}"], D[f"P_{n}_m"], TP, X, Y) for n in "ABC"]
    assert np.array_equal(sum((m > 0) for _, m in bl), D["G_cnt"])
    assert np.abs(sum(np.where(m > 0, v * m, 0) for v, m in bl) - D["G_sum"]).max() < 1e-5      # mask-weighted
    med = np.nanmedian(np.stack([np.where(m > 0, v, np.nan) for v, m in bl]), axis=0)
    assert np.abs(med - D["G_med"]).max() < 1e-9
    assert set(np.unique(D["G_div0"][D["G_div0_m"] > 0])) == {0.0}


def test_median_matches_earth_engine_above_64_values():
    """ImageCollection.median() is exact up to 64 values and a bucket-mean histogram median above (Earth Engine
    results of 33 value sets of 64-300 values in tests/data/ee_median_cases.json)."""
    cases = json.load(open(DATA.parent / "ee_median_cases.json"))["cases"]
    n = max(len(c["values"]) for c in cases)
    st = np.full((n, len(cases)), np.nan)
    for j, c in enumerate(cases):
        st[:len(c["values"]), j] = np.random.default_rng(j).permutation(c["values"])     # order-independent
    got = S.ee_median(st)
    gee = np.array([c["gee_median"] for c in cases])
    assert np.abs(got - gee).max() < 1e-12
    exact = np.array([np.median(c["values"]) for c in cases])
    big = np.array([len(c["values"]) > 64 for c in cases])
    assert np.array_equal(got[~big], exact[~big]) and np.abs(got[big] - exact[big]).max() > 0.01
    assert np.isnan(S.ee_median(np.full((70, 3), np.nan))).all()


def test_season_months_match_the_earth_engine_layers():
    L = pytest.importorskip("canopy_height.tool.gee.layers")
    assert S.SEASON_MONTHS == L.SEASON_MONTHS
    t = [int(1575158400000 + d * 86400000) for d in range(0, 400, 3)]
    assert all(S.in_season(x, k) == L.in_season(x, k) for x in t for k in range(4))


def test_terrain_geometry_and_plane():
    X, Y = S.centres(X0, Y1, 8, 8)
    sx, sy, rot = S.terrain_geometry(EPSG, X, Y)
    # 117.8 E, 5 N (zone 50): fitted against ee.Terrain on 2026-09-29
    assert abs(sx.mean() - 9.99166) < 2e-4 and abs(sy.mean() - 10.05849) < 2e-4 and abs(rot.mean() - 0.0663) < 2e-4
    z = 0.1 * (X - X0)                                               # rises to the east: faces west
    slope, aspect, ok = S.terrain(z, np.ones(z.shape, bool), (sx, sy, rot))
    assert not ok[0].any() and not ok[:, -1].any() and ok[1:-1, 1:-1].all()   # 4 neighbours needed
    inner = (slice(1, -1), slice(1, -1))
    assert np.allclose(slope[inner], np.degrees(np.arctan(0.1 * 10 / sx[inner])))
    assert np.allclose(aspect[inner], 270 - rot[inner])


# ---------------------------------------------------------------------------------------------- the chain
def _scene_grid(cell, crs=None, off=(3.3, 6.1), margin=6):
    """A scene-grid window (10 m, north-up, offset by a fraction of a pixel) covering the cell grid + halo."""
    if crs is None or crs == f"EPSG:{cell.epsg}":
        x0, y1, n = cell.gx0 - margin * 10 - off[0], cell.gy1 + margin * 10 + off[1], cell.m + 2 * margin + 1
        return [10.0, 0.0, x0, 0.0, -10.0, y1], n, n
    from pyproj import Transformer
    X, Y = Transformer.from_crs(f"EPSG:{cell.epsg}", crs, always_xy=True).transform(cell.X.ravel(), cell.Y.ravel())
    x0, y1 = math.floor(min(X) / 10) * 10 - margin * 10 - off[0], math.ceil(max(Y) / 10) * 10 + margin * 10 + off[1]
    return [10.0, 0.0, x0, 0.0, -10.0, y1], int((max(X) - x0) / 10) + margin + 2, int((y1 - min(Y)) / 10) + margin + 2


def _add(cell, sid, vv, vh=0.02, crs=None, elev=None, angle=38.0):
    crs = crs or f"EPSG:{cell.epsg}"
    t, w, h = _scene_grid(cell, crs)
    X, Y = S.centres(t[2], t[5], w, h)
    z = np.full((h, w), 100.0) if elev is None else elev(X, Y)
    cell.add_scene(sid, crs, t, np.full((h, w), vv, np.float32), np.full((h, w), vh, np.float32),
                   np.ones((h, w), np.uint8), z.astype(np.float32))
    cell.add_angle(sid, np.full(cell.X.shape, angle, np.float32), np.ones(cell.X.shape, np.uint8))


def _meta(ids, D=None, heading=-10.0, month=6):
    import datetime as dt
    t0 = int(dt.datetime(2020, month, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
    return [dict(id=s, t=t0 + i * 86400000, D=list(D or ids), heading=heading) for i, s in enumerate(ids)]


def test_chain_constant_backscatter_flat_terrain():
    cell = S.CellS1(EPSG, X0, Y1, n=24)
    for s, vv in (("a", 0.1), ("b", 0.1), ("c", 0.1)):
        _add(cell, s, vv)
    meta = _meta(["a", "b", "c"])
    out, dbg = cell.acquisition(meta[0], debug=True)
    expect = 10 * np.log10(np.array([0.1, 0.02]) / np.cos(np.radians(38.0)))
    for i, b in enumerate(S.BANDS):
        assert np.allclose(cell.crop(out[b]), expect[i], atol=1e-5)                # Quegan of constants = constant
        assert np.allclose(cell.crop(dbg[f"cnt_{b}"]), 3)
    assert np.allclose(cell.crop(dbg["scf"]), 1) and not np.isnan(cell.crop(out["VV"])).any()
    med, n = cell.annual(meta)
    assert n == 3 and med.shape == (2, 24, 24) and np.allclose(med[0], expect[0], atol=1e-5)
    # a scene of the neighbouring UTM zone (EPSG:32651) with the same values gives the same result
    _add(cell, "z51", 0.1, crs="EPSG:32651")
    out51 = cell.acquisition(_meta(["z51"], D=["a", "b", "z51"])[0])
    assert np.allclose(cell.crop(out51["VV"]), expect[0], atol=1e-5)


def test_chain_masks_median_and_layover():
    cell = S.CellS1(EPSG, X0, Y1, n=24)
    _add(cell, "a", 0.1)
    _add(cell, "b", 0.4)
    _add(cell, "edge", 0.1, angle=29.0)                              # outside (30.64, 45.24): f_mask_edges
    meta = _meta(["a", "b", "edge"])
    acq = cell.per_acquisition(meta)
    assert np.isnan(acq["edge"]).all() and not np.isnan(acq["a"]).any()
    med, n = cell.annual(meta, acq)
    assert n == 2 and np.allclose(med, (acq["a"] + acq["b"]) / 2)     # even count: mean of the middle two
    assert np.allclose(acq["b"][0] - acq["a"][0], 10 * np.log10(4), atol=1e-5)
    s, _ = cell.seasonal(meta, 2, acq)                               # June = JJA
    assert np.allclose(s, med, equal_nan=True) and np.isnan(cell.seasonal(meta, 0, acq)[0]).all()
    # terrain: a slope whose downslope bearing is the heading (0 = north: the terrain falls to the north) is seen at
    # alpha_r = slope; steeper than the incidence angle -> layover (masked), gentler -> corrected (scf != 1)
    for grad, masked in ((0.2, False), (2.0, True)):
        c2 = S.CellS1(EPSG, X0, Y1, n=24)
        _add(c2, "t", 0.1, elev=lambda X, Y, g=grad: 100 - g * (Y - Y1))
        o, d = c2.acquisition(_meta(["t"], heading=0.0)[0], debug=True)
        v = c2.crop(o["VV"])
        assert np.isnan(v).all() if masked else (np.isfinite(v).all() and abs(c2.crop(d["scf"]).mean() - 1) > 0.05)


# ---------------------------------------------------------------------------------------------- gee.s1_raw offline
def test_integer_shift_window_and_region_selection(tmp_path):
    R = pytest.importorskip("canopy_height.tool.gee.s1_raw")
    tg = [10.0, 0.0, 584570.0, 0.0, -10.0, 550210.0]
    ox, oy = 346938.43845, 614396.36434
    assert R._window([10, 0, ox, 0, -10, oy], tg) == [10.0, 0.0, ox + 10 * round((tg[2] - ox) / 10), 0.0, -10.0,
                                                        oy - 10 * round((oy - tg[5]) / 10)]
    with pytest.raises(R.NotApplicable, match="half pixel"):
        R._window([10, 0, 584575.0, 0, -10, 550210.0], tg)
    with pytest.raises(R.NotApplicable, match="north-up"):
        R._window([-12493.1, -4266.1, 630565.9, 2579.4, -20048.4, 562780.9], tg)
    assert R.utm_wgs84(32650) and R.utm_wgs84(32733) and not R.utm_wgs84(31983) and not R.utm_wgs84(3857)
    # scene selection: footprints in lon / lat; acquisitions touching the cell, filter-set members touching it + halo
    cell_ll = R.cell_lonlat(EPSG, X0, Y1, 256, 256)
    lo0, la0, lo1, la1 = cell_ll.bounds

    def box(a, b, c, d):
        return {"type": "Polygon", "coordinates": [[[a, b], [c, b], [c, d], [a, d], [a, b]]]}
    reg = R.S1Region(tmp_path / "raw" / "s1_local_2020.json", 2020, False)
    reg.acq = {"in": dict(t=1590969600000, D=["in", "near", "far"], heading=-10.0),
               "out": dict(t=1591969600000, D=["out"], heading=-10.0)}
    reg.scenes = {k: dict(crs="EPSG:32650", t=[10, 0, 0.5, 0, -10, 0.3], foot=f) for k, f in (
        ("in", box(lo0 - 1, la0 - 1, lo1 + 1, la1 + 1)), ("near", box(lo1 + 0.001, la0, lo1 + 1, la1)),
        ("far", box(lo1 + 0.5, la0, lo1 + 1, la1)), ("out", box(lo1 + 0.6, la0, lo1 + 1, la1)))}
    meta, ids = reg.select(EPSG, X0, Y1, 256, 256)
    assert [m["id"] for m in meta] == ["in"] and ids == ["in", "near"]
    assert reg.select(EPSG, X0, Y1, 256, 256, seasons=[0])[0] == []          # June: not DJF
    reg.cells = {reg.key(EPSG, X0, Y1, 256, 256)}
    reg.save()
    again = R.S1Region(tmp_path / "raw" / "s1_local_2020.json", 2020, False)
    assert again.acq == reg.acq and again.cells == reg.cells
    assert again.ensure([(EPSG, X0, Y1, 256, 256)]) == 0                     # nothing to request
    other = R.S1Region(tmp_path / "raw" / "s1_local_2020.json", 2020, True)   # another window: not reused
    assert other.acq == {} and other.start == "2019-12-01"


# ---------------------------------------------------------------------------------------------- download.export
def test_export_local_files_and_fallback(tmp_path, monkeypatch):
    from canopy_height.tool import download
    from canopy_height.tool.gee import io as gee_io, layers as L, s1_raw as R

    def fake_fetch(region, epsg, x0, y1, W, H, halo=S.HALO, res=10.0, seasons=None):
        if x0 != X0:
            raise R.NotApplicable("test: other cell")
        cell = S.CellS1(epsg, x0, y1, n=W, halo=halo)
        for s in ("a", "b"):
            _add(cell, s, 0.1 if s == "a" else 0.4)
        meta = [m for m in _meta(["a", "b"], month=6) if seasons is None or any(S.in_season(m["t"], k)
                                                                                 for k in seasons)]
        return cell, meta, dict(scenes=2, requests=1, mb=0.1)
    monkeypatch.setattr(R, "fetch_cell", fake_fetch)
    monkeypatch.setattr(L, "rect", lambda *a, **k: ("rect", a[1]))
    out = {l: tmp_path / f"c0_{l}.tif" for l in ["S1"] + [f"S1_asc_{k}" for k in range(4)]}
    msg = download.export(["S1"], out, EPSG, X0, Y1, 2020, ["2019-01-01", "2021-12-31"], True, "p", W=24, H=24,
                          s1_method="local", s1_region=object())
    assert msg.startswith("ok") and "local S1: 2 acquisitions" in msg
    with rasterio.open(out["S1"]) as d:
        a, t = d.read(), d.tags()
        assert d.dtypes[0] == "float64" and d.descriptions == ("VV", "VH") and (d.transform.c, d.transform.f) == (X0, Y1)
    assert (t["s1_method"], t["n_s1_images"], t["year"], t["source"]) == ("local", "2", "2020", "S1")
    expect = 10 * np.log10(np.array([0.1, 0.4]) / np.cos(np.radians(38.0))).mean()
    assert np.allclose(a[0], expect, atol=1e-5)
    seas = [f"S1_asc_{k}" for k in range(4)]
    msg = download.export(seas, out, EPSG, X0, Y1, 2020, ["2019-01-01", "2021-12-31"], True, "p", W=24, H=24,
                          s1_method="local", s1_region=object())
    for k, l in enumerate(seas):
        with rasterio.open(out[l]) as d:
            a, t = d.read(), d.tags()
        assert (t["season"], t["window"], t["s1_method"]) == (download.SEASONS[k], "2019-12-01/2020-12-01", "local")
        assert (t["n_s1_images"], np.isnan(a).all()) == (("2", False) if k == 2 else ("0", True))
    # a cell where the local way does not apply takes the Earth Engine way (one metadata request for the bundle)
    calls = []

    def fake_meta(geom, year):
        calls.append(geom)
        return [dict(id="x", t=_meta(["x"])[0]["t"], D=["x"], heading=0.0)]

    class Img:
        def __init__(self, layer):
            self.layer = layer

        def unmask(self, v):
            return self
    monkeypatch.setattr(L, "s1_seasonal_metadata", fake_meta)
    monkeypatch.setattr(L, "layer_image", lambda layer, *a, **k: Img(layer))
    monkeypatch.setattr(gee_io, "download_grid", lambda img, epsg, x0, y1, W, H, *a: np.full((2, H, W), -9.0))
    out2 = {l: tmp_path / f"c1_{l}.tif" for l in seas}
    msg = download.export(seas, out2, EPSG, X0 + 240, Y1, 2020, ["2019-01-01", "2021-12-31"], True, "p", W=24, H=24,
                          s1_method="local", s1_region=object())
    assert "Earth Engine way: local S1 does not apply here" in msg and len(calls) == 1
    with rasterio.open(out2["S1_asc_2"]) as d:
        assert d.tags()["s1_method"] == "gee" and np.allclose(d.read(), -9.0)


def test_project_setting():
    from canopy_height.tool.project import DEFAULTS, validate
    assert DEFAULTS["s1_method"] == "local"
    validate({**DEFAULTS, "s1_method": "gee"})
    with pytest.raises(ValueError, match="s1_method"):
        validate({**DEFAULTS, "s1_method": "snap"})
