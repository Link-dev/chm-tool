"""Canopy-height tool: download stage without Earth Engine (download.export and gee.io.ee_init are stubbed).

Covers the request / EECU estimate, the settings check of existing downloads (year, GEDI window, built-up mask),
imported cells whose folder lacks a layer (downloaded into raw/ and found there), Ctrl+C cancelling the queued
jobs, and the input representations: only the layers of project['input'] are downloaded; the seasonal layers
S1_asc_k / S2_k (T, TE) in the format of make_outside_seasonal, with one S1 metadata request per cell shared by its
four S1 jobs (download.export with gee.layers / gee.io.download_grid stubbed).
"""
import csv
import datetime as dt
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

rasterio = pytest.importorskip("rasterio")

from canopy_height.tool import download  # noqa: E402
from canopy_height.tool.gee import io as gee_io  # noqa: E402
from canopy_height.tool.project import SEASONAL_LAYERS, Project  # noqa: E402

EPSG, X0, Y1, PX = 32650, 584670.0, 550110.0, 256
S1S = [f"S1_asc_{k}" for k in range(4)]
S2S = [f"S2_{k}" for k in range(4)]
COLS = ["cell", "x0", "y1", "col", "row", "role", "dist_m", "water", "built", "tree", "use"]


def _project(root, cells, **settings):
    """Project folder with project.yaml, plan/grid.json and plan/cells.csv; cells = [(name, col, role, extra)]."""
    root = Path(root)
    (root / "plan").mkdir(parents=True, exist_ok=True)
    cfg = dict(name="download test", gee_project="stub-project", workers=2)
    cfg.update(settings)
    (root / "project.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    json.dump(dict(epsg=EPSG, x0=X0, y1=Y1, res=10.0, cell_px=PX), open(root / "plan" / "grid.json", "w"))
    extra_cols = sorted({k for *_, e in cells for k in e})
    with open(root / "plan" / "cells.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS + extra_cols)
        for name, col, role, extra in cells:
            w.writerow([name, X0 + col * PX * 10, Y1, col, 0, role, 0 if role == "aoi" else 300, 0.0, 0.0, 0.5,
                        True] + [extra.get(k, "") for k in extra_cols])
    return Project(root)


def _set(project, **settings):
    cfg = yaml.safe_load(project.path("project.yaml").read_text(encoding="utf-8"))
    cfg.update(settings)
    project.path("project.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    project.close_log()
    return Project(project.root)


def _layer_array(layer):
    if layer == "Tolan_1m":
        a = np.full((1, PX * 10, PX * 10), 255, np.uint8)
        a[0, :1000] = 20
        return a
    nb = {"Embedding": 64, "S1": 2, "S2": 9}.get(layer, 2 if layer in S1S else 9 if layer in S2S else 1)
    return np.full((nb, PX, PX), 7.0)


def _write_layer(dst, layer, x0, tags):
    dtype = download.INT_LAYERS.get(layer, ("float64", None))[0]
    res = 1.0 if layer == "Tolan_1m" else 10.0
    a = _layer_array(layer)
    download._write(dst, a.astype(dtype), dtype, [f"b{i}" for i in range(a.shape[0])], EPSG, x0, Y1, res, **tags)


class StubExport:
    """download.export without Earth Engine: writes every layer with the tags the real export writes."""

    def __init__(self, sleep=0.0):
        self.calls, self.sleep, self.lock = [], sleep, threading.Lock()

    def __call__(self, layers, out, epsg, x0, y1, year, gedi_window, built_up_mask, gee_project, **kw):
        with self.lock:
            self.calls.append(tuple(layers))
        if self.sleep:
            gee_io._sleep(self.sleep)
        base = dict(project=gee_project, year=str(year), source="stub")
        for l in layers:
            tags = dict(base, **(download.gedi_tags(gedi_window, built_up_mask) if l == "GEDI" else {}))
            _write_layer(out[l], l, x0, tags)
        return "ok (stub)"


@pytest.fixture
def stub(monkeypatch):
    s = StubExport()
    monkeypatch.setattr(download, "export", s)
    monkeypatch.setattr(download, "s1_regions", lambda *a, **k: {})          # the local way's area metadata
    monkeypatch.setattr(gee_io, "ee_init", lambda *a, **k: None)
    yield s
    gee_io.CANCEL.clear()


# ---------------------------------------------------------------------------------------------- estimate
def test_estimate_counts_http_calls_and_eecu_range(tmp_path):
    p = _project(tmp_path / "p", [("a0", 0, "aoi", {}), ("r0", 1, "ring", {})], s1_method="gee")
    assert [download.chunks(l) for l in ("Embedding", "S1", "S2", "DEM", "Tolan_1m")] == [1, 1, 1, 1, 1]
    est = download.estimate(p)
    # aoi: Embedding 3, S1 4, S2 3, DEM+GEDI 2, ETH 3, UMD 3, Tolan_1m 3 = 21; ring: 3 + 4 + 3 + 2 = 12
    assert (est["cells"], est["jobs"], est["requests"]) == (2, 11, 33)
    assert est["layers"]["S1"] == 2 and est["layers"]["Tolan_1m"] == 1
    assert est["eecu_s_low"] < est["eecu_s"] < est["eecu_s_high"]
    assert 2 * 100 <= est["eecu_s_low"] < 2 * 200 and est["eecu_s_high"] >= 2 * 3000
    assert "2 x 2" not in est["note"] and "split" in est["note"] and est["transfer_mb"] == 0
    _write_layer(p.raw("r0", "S1"), "S1", X0 + 2560, dict(year="2020"))
    after = download.estimate(p)
    assert (after["jobs"], after["requests"]) == (10, 29) and after["layers"]["S1"] == 1
    # the local way (default): 3 requests per S1 job and the area metadata once; ~50 MB of scenes pass per cell
    p = _set(p, s1_method="local")
    loc = download.estimate(p)
    assert (loc["jobs"], loc["requests"], loc["transfer_mb"], loc["s1_method"]) == (10, 28 + 2, 50, "local")
    assert loc["eecu_s"] == after["eecu_s"] - 150 + 10 and loc["eecu_s_high"] < after["eecu_s_high"]


# ---------------------------------------------------------------------------------------------- settings check
def test_changed_settings_stop_the_download(tmp_path, stub):
    imp = tmp_path / "pipeline_inputs"
    for l in ("Embedding", "DEM", "S1", "S2", "GEDI"):                 # pipeline raster: other year, no GEDI tags
        _write_layer(imp / f"imp0_{l}.tif", l, X0 + 2560, dict(year="2019", source=l))
    p = _project(tmp_path / "p", [("c0", 0, "ring", {}), ("imp0", 1, "ring", dict(src_dir=str(imp), lab_dir=str(imp),
                                                                                   epsg=EPSG))],
                 benchmarks=[])
    res = download.run_download(p)
    assert res["jobs"] == 4 and res["failed"] == [] and len(stub.calls) == 4        # c0 only
    assert download.run_download(p)["jobs"] == 0

    p = _set(p, year=2021)
    with pytest.raises(download.SettingsChanged) as e:
        download.run_download(p, layers=["HRCH"])                    # every layer is checked, not only `layers`
    msg = str(e.value)
    assert "3 downloaded rasters" in msg and "c0" in msg and "imp0" not in msg
    assert "S1: year 2020 (project: 2021)" in msg and "Start a new project" in msg and "raw/<cell>_S2.tif" in msg
    assert "DEM" not in msg and "GEDI" not in msg and len(stub.calls) == 4
    assert download.estimate(p)["jobs"] == 0                         # the estimate does not read tags

    p = _set(p, year=2020, gedi_window=["2019-01-01", "2020-01-01"], built_up_mask=False)
    with pytest.raises(download.SettingsChanged, match=r"GEDI: gedi_end 2021-12-31 \(project: 2020-01-01\), "
                                                        r"built_up_mask True \(project: False\)"):
        download.run_download(p)

    p.raw("c0", "GEDI").unlink()                                      # deleted -> downloaded with the new settings
    assert download.run_download(p)["jobs"] == 1 and stub.calls[-1] == ("GEDI",)
    with rasterio.open(p.raw("c0", "GEDI")) as s:
        assert (s.tags()["gedi_end"], s.tags()["built_up_mask"]) == ("2020-01-01", "False")
    _write_layer(p.raw("c0", "GEDI"), "GEDI", X0, {})                   # no tags (e.g. copied in): accepted
    p = _set(p, gedi_window=["2019-01-01", "2021-12-31"])
    assert download.run_download(p)["jobs"] == 0


def test_unquoted_yaml_dates_match_the_tags(tmp_path, stub):
    p = _project(tmp_path / "p", [("c0", 0, "ring", {})], benchmarks=[])
    text = p.path("project.yaml").read_text(encoding="utf-8")
    p.path("project.yaml").write_text(text + "gedi_window: [2019-01-01, 2021-12-31]\n", encoding="utf-8")
    p = Project(p.root)
    assert not isinstance(p["gedi_window"][0], str)                   # datetime.date from YAML
    assert download.run_download(p)["jobs"] == 4
    assert download.settings_mismatch(p, dict(cell="c0"), "GEDI") is None
    assert download.run_download(p)["jobs"] == 0


# ---------------------------------------------------------------------------------------------- imported cells
def test_imported_cell_missing_layers_land_in_raw(tmp_path, stub, monkeypatch):
    imp = tmp_path / "pipeline_inputs"
    for l in ("Embedding", "DEM", "S2", "GEDI"):                     # S1 and the benchmarks are missing
        _write_layer(imp / f"imp0_{l}.tif", l, X0, dict(year="2020", source=l))
    p = _project(tmp_path / "p", [("imp0", 0, "aoi", dict(src_dir=str(imp), lab_dir=str(imp), epsg=EPSG))],
                 benchmarks=["GMTCH", "HRCH"])
    row = download.load_plan(p)[1].to_dict("records")[0]
    assert download.jobs_for(p, row) == [["S1"], ["Tolan_1m"], ["ETH"]]
    replaced = []
    monkeypatch.setattr(download, "replace_retry",
                        lambda a, b: (replaced.append(Path(b).name), download.os.replace(a, b)))
    res = download.run_download(p)
    assert (res["jobs"], res["gmtch"], res["failed"]) == (3, 1, [])
    assert {"imp0_S1.tif", "imp0_ETH.tif", "imp0_Tolan_1m.tif", "imp0_GMTCH.tif"} <= set(replaced)
    for l in ("S1", "ETH", "Tolan_1m", "GMTCH"):
        assert p.cell_raster(row, l) == p.raw("imp0", l) and p.raw("imp0", l).exists()
    assert p.cell_raster(row, "S2") == imp / "imp0_S2.tif"
    again = download.run_download(p)
    assert (again["jobs"], again["gmtch"]) == (0, 0) and len(stub.calls) == 3
    with rasterio.open(p.raw("imp0", "GMTCH")) as s:
        g = s.read(1)
    assert np.isclose(g[0, 0], 20) and np.isnan(g[-1, -1])


# ---------------------------------------------------------------------------------------------- Ctrl+C
def test_ctrl_c_cancels_queued_jobs(tmp_path, monkeypatch):
    p = _project(tmp_path / "p", [(f"c{i}", i, "ring", {}) for i in range(10)], benchmarks=[])
    starts, ends, lock = [], [], threading.Lock()

    def slow_export(layers, out, *a, **k):
        with lock:
            starts.append(time.time())
            first = len(starts) == 1
        try:
            gee_io._sleep(0.2 if first else 30)                     # a long chunk / retry wait
        finally:
            with lock:
                ends.append(time.time())
        return "ok (stub)"
    monkeypatch.setattr(download, "export", slow_export)
    monkeypatch.setattr(download, "s1_regions", lambda *a, **k: {})
    monkeypatch.setattr(gee_io, "ee_init", lambda *a, **k: None)
    log, lines, hit = p.log, [], []

    def interrupting_log(msg):
        lines.append(msg)
        log(msg)
        if msg.startswith("download [1/") and not hit:
            hit.append(time.time())
            raise KeyboardInterrupt
    p.log = interrupting_log
    try:
        with pytest.raises(KeyboardInterrupt):
            download.run_download(p, workers=2)
        assert gee_io.CANCEL.is_set()
        t_end = time.time() + 5
        while len(ends) < len(starts) and time.time() < t_end:
            time.sleep(0.05)
        assert len(ends) == len(starts) <= 3                         # 40 jobs planned: the queue was cancelled
        assert sum(t > hit[0] for t in starts) <= 1
        assert max(ends) - hit[0] < 2                                # the running job stopped at its wait
        assert any(m.startswith("download interrupted: ") and "queued jobs cancelled" in m for m in lines)
    finally:
        gee_io.CANCEL.clear()
        p.close_log()


def test_cancel_stops_worker_retries_only(monkeypatch):
    calls = []

    def busy():
        calls.append(1)
        raise RuntimeError("429 Too Many Requests")

    class Image:
        def getDownloadURL(self, *a):
            calls.append("url")
            raise RuntimeError("429")
    out = {}

    def worker():
        try:
            gee_io.retry(busy)
        except Exception as e:  # noqa: BLE001
            out["retry"] = e
        try:
            gee_io.download_grid(Image(), EPSG, X0, Y1, 8, 8, 10.0, 1, 8)
        except Exception as e:  # noqa: BLE001
            out["grid"] = e
    gee_io.CANCEL.set()
    try:
        t0 = time.time()
        th = threading.Thread(target=worker)
        th.start()
        th.join(10)
        assert time.time() - t0 < 2 and calls == [1]                  # no back-off wait, no download request
        assert isinstance(out["retry"], gee_io.Cancelled) and isinstance(out["grid"], gee_io.Cancelled)
        gee_io._sleep(0.01)                                          # the main thread is not cancelled
        gee_io._check_cancel()
    finally:
        gee_io.CANCEL.clear()


# ---------------------------------------------------------------------------------------------- input representations
def test_cell_layers_follow_the_input_representation(tmp_path):
    p = _project(tmp_path / "p", [("a0", 0, "aoi", {}), ("r0", 1, "ring", {})], benchmarks=["GMTCH", "HRCH"])
    aoi, ring = download.load_plan(p)[1].to_dict("records")
    want = {"AE": ["Embedding", "DEM", "S1", "S2"], "A": ["DEM", "S1", "S2"], "E": ["Embedding", "DEM"],
            "T": ["DEM"] + SEASONAL_LAYERS, "TE": ["Embedding", "DEM"] + SEASONAL_LAYERS}
    assert SEASONAL_LAYERS == S1S + S2S
    for inp, lay in want.items():
        p = _set(p, input=inp)
        assert download.cell_layers(p, ring) == lay + ["GEDI"]
        assert download.cell_layers(p, aoi) == lay + ["GEDI", "GMTCH", "ETH"]
    # the local way (default) fetches the raw scenes once for the four S1 seasons of a cell: one job
    assert download.jobs_for(p, ring) == [["Embedding"]] + [[l] for l in S2S] + [S1S] + [["DEM", "GEDI"]]
    p.raw("r0", "S1_asc_1").parent.mkdir(parents=True, exist_ok=True)
    p.raw("r0", "S1_asc_1").write_bytes(b"")
    assert [S1S[0]] + S1S[2:] in download.jobs_for(p, ring)                     # only the missing seasons
    p.raw("r0", "S1_asc_1").unlink()
    p = _set(p, s1_method="gee")
    assert download.jobs_for(p, ring) == [["Embedding"]] + [[l] for l in SEASONAL_LAYERS] + [["DEM", "GEDI"]]
    p = _set(p, input="E")
    assert download.jobs_for(p, aoi) == [["Embedding"], ["Tolan_1m"], ["ETH"], ["DEM", "GEDI"]]
    assert download.jobs_for(p, ring, wanted=download._wanted(["S2_0", "S1"])) == []     # not layers of E
    assert download.cell_cost("E", ["GMTCH", "GFCH", "HRCH"]) == ((25.63, 11), (1.23, 7))
    assert download.cell_cost("AE") == ((27.33, 81), (0, 0))                   # S1 the local way
    assert download.cell_cost("T") == ((17.23, 98), (0, 0))
    assert download.cell_cost("AE", s1_method="gee") == ((27.33, 221), (0, 0))
    assert download.cell_cost("T", s1_method="gee") == ((17.23, 246), (0, 0))
    with pytest.raises(ValueError, match="choose from"):
        download.cell_cost("X")


def test_seasonal_estimate(tmp_path):
    p = _project(tmp_path / "p", [("a0", 0, "aoi", {}), ("r0", 1, "ring", {})], input="T", s1_method="gee")
    assert [download.chunks(l) for l in SEASONAL_LAYERS] == [1] * 8                  # 512 / 768 px: one chunk
    assert [download.CHUNK[l] for l in ("S1_asc_0", "S1_asc_3", "S2_0", "S2_3")] == [512, 512, 768, 768]
    est = download.estimate(p)
    # per cell: 4 S1_asc x 2 + 4 S2_k x 3 (bandNames) + DEM+GEDI 2 + 1 S1 metadata = 23; aoi + ETH, UMD, Tolan_1m 9
    assert (est["cells"], est["jobs"], est["requests"]) == (2, 21, 55)
    assert est["layers"]["S1_asc_0"] == 2 and "S1" not in est["layers"] and "Embedding" not in est["layers"]
    assert est["eecu_s"] == 2 * (4 * 40 + 4 * 20 + 1 + 5) + 1 + 1 + 5 and est["mb"] == 35.7
    assert est["eecu_s_low"] >= 2 * 4 * 25 and est["eecu_s_high"] >= 2 * 3000         # like the annual S1
    for k in range(3):
        _write_layer(p.raw("r0", f"S1_asc_{k}"), f"S1_asc_{k}", X0 + 2560, dict(year="2020"))
    after = download.estimate(p)
    assert (after["jobs"], after["requests"]) == (18, 49)             # the last S1 season still needs the metadata
    _write_layer(p.raw("r0", "S1_asc_3"), "S1_asc_3", X0 + 2560, dict(year="2020"))
    assert download.estimate(p)["requests"] == 46
    # the local way: a0's four seasons are one job of 3 requests, + the area metadata
    p = _set(p, s1_method="local")
    loc = download.estimate(p)
    assert (loc["jobs"], loc["requests"], loc["transfer_mb"]) == (17 - 4 + 1, 46 - 4 * 2 - 1 + 3 + 2, 50)
    assert loc["layers"]["S1_asc_0"] == 1 and loc["eecu_s"] == est["eecu_s"] - 2 * 4 * 40 + 4 * 3


def test_seasonal_download_and_settings_check(tmp_path, stub):
    p = _project(tmp_path / "p", [("c0", 0, "ring", {})], benchmarks=[], input="TE")
    res = download.run_download(p)
    assert (res["jobs"], res["failed"]) == (7, [])                              # the four S1 seasons: one job
    assert sorted(stub.calls) == sorted([("Embedding",), ("DEM", "GEDI"), tuple(S1S)] + [(l,) for l in S2S])
    assert not p.raw("c0", "S1").exists() and not p.raw("c0", "S2").exists()
    assert download.run_download(p)["jobs"] == 0
    p = _set(p, year=2021)
    with pytest.raises(download.SettingsChanged) as e:
        download.run_download(p)
    msg = str(e.value)
    assert "9 downloaded rasters" in msg and "S1_asc_0: year 2020 (project: 2021)" in msg
    assert "raw/<cell>_S2_3.tif" in msg


class _Info:
    def __init__(self, v):
        self.v = v

    def getInfo(self):
        return self.v


class _FakeImage:
    def __init__(self, layer, bands):
        self.layer, self.bands, self.unmasked = layer, bands, None

    def bandNames(self):
        return _Info(self.bands)

    def unmask(self, v):
        self.unmasked = v
        return self


def _ms(y, m, d):
    return int(dt.datetime(y, m, d, 3, tzinfo=dt.timezone.utc).timestamp() * 1000)


def test_seasonal_export_format_and_shared_s1_metadata(tmp_path, monkeypatch):
    from canopy_height.tool.gee import layers as L
    p = _project(tmp_path / "p", [("c0", 0, "ring", {}), ("c1", 1, "ring", {})], benchmarks=[], input="T",
                 s1_method="gee")
    # c0: acquisitions in Dec 2019 (DJF) and April, July 2020; none in SON. c1: every season
    meta = {X0: [dict(id="S1_a", t=_ms(2019, 12, 20), D=["S1_a"], heading=-10.0),
                 dict(id="S1_b", t=_ms(2020, 4, 2), D=["S1_a", "S1_b"], heading=-10.0),
                 dict(id="S1_c", t=_ms(2020, 7, 9), D=["S1_b", "S1_c"], heading=-10.0)],
            X0 + 2560: [dict(id=f"S1_{m}", t=_ms(2020, m, 1), D=[], heading=0.0) for m in (1, 2, 3, 6, 9, 11)]}
    calls, grids, lock = [], [], threading.Lock()

    def fake_meta(geom, year):
        with lock:
            calls.append((geom[1], year))
        time.sleep(0.3)                                                # the cell's other S1 jobs wait for it
        return meta[geom[1]]

    def fake_image(layer, geom, year, gedi_window=None, built_up_mask=True, meta=None):
        kind, k = L.seasonal(layer)
        if kind == "S1":
            assert meta is not None and year == 2020
            return _FakeImage(layer, ["VV", "VH"])
        return _FakeImage(layer, [] if (layer == "S2_2" and geom[1] == X0) else download.SEASONAL_BANDS["S2"])

    def fake_grid(img, epsg, x0, y1, W, H, res, nb, chunk):
        with lock:
            grids.append((img.layer, x0, chunk, nb, img.unmasked, (W, H, res)))
        a = np.full((nb, H, W), -7.5 if img.layer.startswith("S1") else 0.0312)
        a[:, 0, 0] = np.nan
        return a
    monkeypatch.setattr(gee_io, "ee_init", lambda *a, **k: None)
    monkeypatch.setattr(gee_io, "download_grid", fake_grid)
    monkeypatch.setattr(L, "rect", lambda epsg, x0, y1, W, H, res=10.0: ("rect", float(x0)))
    monkeypatch.setattr(L, "s1_seasonal_metadata", fake_meta)
    monkeypatch.setattr(L, "layer_image", fake_image)
    res = download.run_download(p, workers=4, layers=SEASONAL_LAYERS)
    assert (res["jobs"], res["failed"]) == (16, [])
    assert sorted(calls) == [(X0, 2020), (X0 + 2560, 2020)]            # one metadata request per cell
    got = {(g[0], g[1]): g for g in grids}
    assert len(grids) == 14 and ("S1_asc_3", X0) not in got and ("S2_2", X0) not in got    # empty seasons
    assert {g[2] for g in grids if g[0] in S1S} == {512} and {g[2] for g in grids if g[0] in S2S} == {768}
    assert {g[4] for g in grids} == {gee_io.SENTINEL} and {g[5] for g in grids} == {(PX, PX, 10.0)}
    n_img = {"c0": ["1", "1", "1", "0"], "c1": ["2", "1", "1", "2"]}
    for cell, x0 in (("c0", X0), ("c1", X0 + 2560)):
        for l in SEASONAL_LAYERS:
            with rasterio.open(p.raw(cell, l)) as s:
                a, t = s.read(), s.tags()
                assert (s.dtypes[0], s.crs.to_epsg(), s.transform.c, s.transform.f) == ("float64", EPSG, x0, Y1)
                assert np.isnan(s.nodata) and s.descriptions == tuple(download.SEASONAL_BANDS[l[:2]])
                assert (t["year"], t["source"], t["window"]) == ("2020", l, "2019-12-01/2020-12-01")
                assert t["season"] == download.SEASONS[int(l[-1])]
            empty = (cell, l) in (("c0", "S1_asc_3"), ("c0", "S2_2"))
            assert np.isnan(a).all() if empty else (np.isnan(a).sum() == a.shape[0] and a[0, 1, 1] in (-7.5, 0.0312))
            assert t.get("n_s1_images", "") == (n_img[cell][int(l[-1])] if l in S1S else "")   # "": not stored
            assert t.get("s1_method", "") == ("gee" if l in S1S else "")
    lines = p.log_file.read_text(encoding="utf-8")
    assert "S1_asc_3: ok" in lines and "0 of 3 S1 images in season, no data in this season (NaN)" in lines
    p.close_log()


def test_s1_meta_cache(monkeypatch):
    cache = download.S1MetaCache({"a": 2})
    calls, out = [], []

    def compute():
        calls.append(1)
        time.sleep(0.2)
        return ["meta"]
    ths = [threading.Thread(target=lambda: out.append(cache.get("b", compute))) for _ in range(4)]
    for th in ths:
        th.start()
    for th in ths:
        th.join(5)
    assert out == [["meta"]] * 4 and len(calls) == 1
    assert cache.get("a", compute) == cache.get("a", compute) == ["meta"] and len(calls) == 2
    assert cache.get("a", compute) == ["meta"] and len(calls) == 3          # dropped after its 2 uses: fetched again

    def bad():
        calls.append("x")
        raise RuntimeError("User memory limit exceeded")
    for _ in range(3):                                                  # the error is shared, not repeated
        with pytest.raises(RuntimeError, match="S1 seasonal metadata failed.*memory limit"):
            cache.get("c", bad)
    assert calls.count("x") == 1

    err = {}

    def slow():
        time.sleep(1.5)
        return ["late"]

    def waiter():
        try:
            cache.get("d", slow)
        except Exception as e:  # noqa: BLE001
            err["e"] = e
    first = threading.Thread(target=lambda: cache.get("d", slow))
    first.start()
    time.sleep(0.2)
    second = threading.Thread(target=waiter)
    second.start()
    try:
        time.sleep(0.2)
        t0 = time.time()
        gee_io.CANCEL.set()                                             # Ctrl+C: a job waiting for the metadata stops
        second.join(3)
        assert isinstance(err.get("e"), gee_io.Cancelled) and time.time() - t0 < 1.2
    finally:
        first.join(5)
        gee_io.CANCEL.clear()


def test_seasonal_recipes_offline():
    L = pytest.importorskip("canopy_height.tool.gee.layers")
    assert L.SEASONAL_LAYERS == SEASONAL_LAYERS and L.SEASON_MONTHS == [(12, 2), (3, 5), (6, 8), (9, 11)]
    assert [L.seasonal(l) for l in ("S1_asc_0", "S2_3", "S1", "S2", "DEM")] == [("S1", 0), ("S2", 3), None, None, None]
    months = {k: [m for m in range(1, 13) if L.in_season(_ms(2020, m, 1), k)] for k in range(4)}
    assert months == {0: [1, 2, 12], 1: [3, 4, 5], 2: [6, 7, 8], 3: [9, 10, 11]}
    t = int(dt.datetime(2020, 3, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)          # UTC month
    assert L.in_season(t - 1, 0) and not L.in_season(t - 1, 1) and L.in_season(t, 1)
    assert set(download.SEASONS) == {"DJF", "MAM", "JJA", "SON"}
