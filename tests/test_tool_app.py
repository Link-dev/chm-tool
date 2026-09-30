"""Streamlit interface and map rendering of the canopy-height tool, on synthetic rasters and a fake project folder
(no Earth Engine, no torch)."""
import base64
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
pd = pytest.importorskip("pandas")
from rasterio.transform import from_origin  # noqa: E402
from rasterio.warp import transform, transform_bounds  # noqa: E402
from shapely.geometry import Polygon, box  # noqa: E402

from canopy_height.tool import viz  # noqa: E402
from canopy_height.tool.project import INPUTS, Project  # noqa: E402

APP = Path(viz.__file__).with_name("app.py")
EPSG, X0, Y1 = 32650, 584670.0, 550110.0          # grid anchor of a pipeline cell set
PNG = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(autouse=True)
def weights(tmp_path_factory, monkeypatch):
    """Every UNet-ALS checkpoint present ($CHM_WEIGHTS/source/, empty files: the page only checks that they
    exist), so the tests do not depend on downloaded weights."""
    d = tmp_path_factory.mktemp("weights") / "source"
    d.mkdir()
    for v in INPUTS.values():
        (d / v["checkpoint"]).write_bytes(b"")
    monkeypatch.setenv("CHM_WEIGHTS", str(d.parent))
    monkeypatch.delenv("CHM_UNET_ALS", raising=False)
    return d


def _tif(path, a, x0=X0, y1=Y1, nodata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=a.shape[1], height=a.shape[0], count=1, dtype=a.dtype,
                       crs=f"EPSG:{EPSG}", transform=from_origin(x0, y1, 10.0, 10.0), nodata=nodata) as dst:
        dst.write(a, 1)
    return path


def _cells():
    return pd.DataFrame(dict(cell=["x584670y550110", "x587230y550110", "x582110y550110"],
                             x0=[584670.0, 587230.0, 582110.0], y1=[Y1] * 3, col=[0, 1, -1], row=[0, 0, 0],
                             role=["aoi", "ring", "ring"], dist_m=[0, 0, 0], water=[0.0, 0.1, 0.99],
                             built=[0.0] * 3, tree=[0.9, 0.8, 0.0], use=[True, True, False]))


def _aoi_lonlat():
    xs = [X0 + 300, X0 + 2200, X0 + 2200, X0 + 300]
    ys = [Y1 - 300, Y1 - 300, Y1 - 2200, Y1 - 2200]
    lon, lat = transform(f"EPSG:{EPSG}", "EPSG:4326", xs, ys)
    return Polygon(list(zip(lon, lat)))


# ---------------------------------------------------------------------------------------------- viz
def test_warp_colorize_png(tmp_path):
    g = np.random.default_rng(0)
    a = g.uniform(0, 40, (300, 200)).astype(np.float32)
    a[:50, :50] = np.nan
    a[-20:, -20:] = -999
    f = _tif(tmp_path / "m.tif", a)
    w, b = viz.warp(f, max_px=100)
    assert w.dtype == np.float32 and max(w.shape) <= 110
    west, south, east, north = transform_bounds(f"EPSG:{EPSG}", "EPSG:4326", X0, Y1 - 3000, X0 + 2000, Y1)
    assert np.allclose([b[0][0], b[0][1], b[1][0], b[1][1]], [south, west, north, east], atol=1e-3)
    fin = w[np.isfinite(w)]
    assert fin.size and fin.min() >= 0 and fin.max() <= 40 and np.isnan(w).any()
    rgba = viz.colorize(w, 40)
    assert rgba.shape == w.shape + (4,) and rgba.dtype == np.uint8
    assert (rgba[..., 3][np.isnan(w)] == 0).all() and (rgba[..., 3][np.isfinite(w)] == 255).all()
    url = viz.png_data_url(rgba)
    assert url.startswith("data:image/png;base64,") and base64.b64decode(url.split(",", 1)[1])[:8] == PNG


def test_mask_and_block_reduce(tmp_path):
    a = np.full((8, 8), np.nan, np.float32)
    a[1, 1], a[6, 5] = 30, 12
    m = viz.block_reduce(a, 4, "max")
    assert m[0, 0] == 30 and m[1, 1] == 12 and np.isnan(m[0, 1])
    assert np.array_equal(viz.block_reduce(np.arange(16, dtype=np.float32).reshape(4, 4), 2),
                          [[2.5, 4.5], [10.5, 12.5]])
    f = _tif(tmp_path / "h.tif", np.full((64, 64), 20, np.float32))
    mask = np.ones((64, 64), np.uint8)
    mask[:, :32] = 0
    fm = _tif(tmp_path / "aoi_mask.tif", mask)
    band, _, _ = viz.read_band(f, mask_path=fm)
    assert np.isnan(band[:, :32]).all() and (band[:, 32:] == 20).all()
    assert viz.common_vmax([np.array([1.0, 2.0, 33.0]), np.array([np.nan])], q=100) == 35
    assert viz.common_vmax([np.array([np.nan])]) == 5


def test_read_band_decimated(tmp_path):
    g = np.random.default_rng(3)
    a = g.uniform(0, 40, (1000, 600)).astype(np.float32)
    a[:10, :10] = np.nan                                # a whole 10 x 10 block without data
    a[10:20, :5] = np.nan                               # half a block
    f = _tif(tmp_path / "nd.tif", a, nodata=np.nan)
    full, tr, _ = viz.read_band(f)
    d, dtr, crs = viz.read_band(f, max_px=100)         # GDAL block averages (out_shape)
    assert d.shape == (100, 60) and crs.to_epsg() == EPSG and np.isclose(dtr.a, 100) and (dtr.c, dtr.f) == (X0, Y1)
    assert np.allclose(d, viz.block_reduce(full, 10), atol=1e-3, equal_nan=True) and np.isnan(d[0, 0])
    odd, otr, _ = viz.read_band(_tif(tmp_path / "odd.tif", a[:997, :593], nodata=np.nan), max_px=100)
    assert odd.shape == (100, 60) and np.isclose(otr.a * 60, 5930) and np.isclose(otr.e * 100, -9970)  # same extent
    mask = np.zeros(a.shape, np.uint8)
    mask[:, :300] = 1
    fm = _tif(tmp_path / "aoi_mask.tif", mask)
    dm, _, _ = viz.read_band(f, mask_path=fm, max_px=100)
    assert np.isnan(dm[:, 30:]).all() and np.isfinite(dm[1:, :30]).all()
    b = a.copy()                                        # no nodata value: exact block means, sentinels dropped
    b[-25:, -25:] = -999
    f2 = _tif(tmp_path / "sentinel.tif", b)
    ref = viz.block_reduce(viz.read_band(f2)[0], 10)
    d2, t2, _ = viz.read_band(f2, max_px=100)
    assert np.array_equal(d2, ref, equal_nan=True) and np.isclose(t2.a, 100) and np.nanmin(d2) >= 0
    with rasterio.open(f2) as s:                        # strip by strip, strips not a multiple of the rows
        for k in (7, 10):
            m, _ = viz._read_blocks(s, 1, k, "mean", None, strip_px=600 * k * 3)
            assert np.array_equal(m, viz.block_reduce(viz.read_band(f2)[0], k), equal_nan=True)
    gd = np.full(a.shape, np.nan, np.float32)
    gd[5, 5], gd[500, 300] = 33, 12
    gm, _, _ = viz.read_band(_tif(tmp_path / "gedi.tif", gd, nodata=np.nan), max_px=100, how="max")
    assert gm[0, 0] == 33 and gm[50, 30] == 12 and np.isfinite(gm).sum() == 2
    assert viz.file_bytes(str(f), f.stat().st_mtime) == f.read_bytes()


def test_cells_area_quicklook(tmp_path):
    cells = _cells()
    gj = viz.cells_geojson(cells, EPSG)
    assert [f["properties"]["category"] for f in gj["features"]] == ["aoi", "ring", "dropped (water)"]
    assert viz.cell_counts(cells) == {"aoi": 1, "ring": 1, "dropped (water)": 1}
    ring = np.array(gj["features"][0]["geometry"]["coordinates"][0])
    assert ring.shape == (5, 2) and np.allclose(ring[0], ring[-1]) and 117 < ring[:, 0].mean() < 118.5
    cells.to_csv(tmp_path / "cells.csv", index=False)           # bools as written by pandas
    assert viz.cell_counts(pd.read_csv(tmp_path / "cells.csv")) == {"aoi": 1, "ring": 1, "dropped (water)": 1}
    assert abs(viz.area_km2(box(0, 0, 1, 1)) - 12308) < 30
    assert abs(viz.area_km2(_aoi_lonlat()) - 1.9 ** 2) < 0.05
    q = viz.save_quicklook(np.random.default_rng(1).uniform(0, 30, (50, 80)), tmp_path / "q.png", 30, "test")
    assert q.read_bytes()[:8] == PNG


def test_folium_maps(tmp_path):
    pytest.importorskip("folium")
    aoi = _aoi_lonlat()
    assert "Study area" in viz.study_area_map(aoi, _cells(), EPSG).get_root().render()
    a, b = viz.warp(_tif(tmp_path / "m.tif", np.full((256, 256), 25, np.float32)))
    html = viz.results_map([("UNet-SLS", viz.png_data_url(viz.colorize(a, 30)), b)], 30, aoi).get_root().render()
    assert "UNet-SLS" in html and "Canopy height (m)" in html
    assert "draw" in viz.draw_map().get_root().render().lower()


# ---------------------------------------------------------------------------------------------- app helpers
def _app():
    pytest.importorskip("streamlit")
    return importlib.import_module("canopy_height.tool.app")


def _fake_project(root):
    """project.yaml + aoi.geojson + plan/ + a fake status.json + maps/ + mosaic/ + report/ + logs/."""
    from shapely.geometry import mapping
    root.mkdir(parents=True)
    (root / "project.yaml").write_text("name: fake\nyear: 2020\ngee_project: my-cloud-project\n", encoding="utf-8")
    (root / "aoi.geojson").write_text(json.dumps(dict(type="FeatureCollection", features=[dict(
        type="Feature", properties={}, geometry=mapping(_aoi_lonlat()))])), encoding="utf-8")
    (root / "plan").mkdir()
    (root / "plan" / "grid.json").write_text(json.dumps(dict(epsg=EPSG, x0=X0, y1=Y1, res=10.0, cell_px=256)))
    _cells().to_csv(root / "plan" / "cells.csv", index=False)
    (root / "status.json").write_text(json.dumps(dict(
        plan=dict(state="done", message="3 cells", started="2026-09-27 10:00:00", finished="2026-09-27 10:01:00"),
        download=dict(state="running", message="12 / 40 files", started="2026-09-27 10:01:00"))))
    g = np.random.default_rng(2)
    m = g.uniform(0, 45, (256, 256)).astype(np.float32)
    m[:20] = np.nan
    _tif(root / "maps" / "UNet-SLS.tif", m)
    _tif(root / "maps" / "RF-SLS.tif", m * 0.9)
    gedi = np.full((256, 256), -999, np.float32)
    gedi[g.integers(0, 256, 300), g.integers(0, 256, 300)] = g.uniform(5, 50, 300)
    _tif(root / "mosaic" / "GEDI.tif", gedi)
    _tif(root / "mosaic" / "HRCH.tif", m + 2)
    mask = np.zeros((256, 256), np.uint8)
    mask[30:220, 30:220] = 1
    _tif(root / "mosaic" / "aoi_mask.tif", mask)
    (root / "report").mkdir()
    pd.DataFrame(dict(product=["UNet-SLS", "HRCH"], reference=["GEDI", "GEDI"], rmse=[5.1, 7.3], n=[300, 300])) \
        .to_csv(root / "report" / "metrics.csv", index=False)
    viz.save_quicklook(m, root / "report" / "UNet-SLS.png", 50, "UNet-SLS")
    (root / "logs").mkdir()
    (root / "logs" / "run.log").write_text("".join(f"2026-09-27 10:0{i % 10}:00 line {i}\n" for i in range(100)),
                                           encoding="utf-8")
    return root


def test_pick_aoi_file_and_tail(tmp_path):
    app = _app()
    names = ["a.dbf", "a.prj", "a.shp", "a.shx", "b.geojson"]
    assert app.pick_aoi_file([tmp_path / n for n in names]).name == "a.shp"
    assert app.pick_aoi_file([tmp_path / "x.zip", tmp_path / "y.kml"]).name == "x.zip"
    assert app.pick_aoi_file([tmp_path / "a.dbf"]) is None
    f = tmp_path / "log.txt"
    f.write_text("".join(f"line {i}\n" for i in range(100)), encoding="utf-8")
    t = app.tail(f).splitlines()
    assert len(t) == app.TAIL and t[-1] == "line 99"
    assert app.tail(tmp_path / "missing.log") == ""
    cmd = app.run_command(tmp_path, "train")
    assert cmd[1:4] == ["-m", "canopy_height.tool", "run"] and cmd[-4:] == ["--from", "train", "--to", "train"]
    assert app.run_command(tmp_path)[-1] == str(tmp_path.resolve())


def test_background_run_start_stop(tmp_path):
    app = _app()
    root = tmp_path / "proj"
    root.mkdir()
    (root / "project.yaml").write_text("name: t\n", encoding="utf-8")
    assert app.current_run(root) is None and app.stop_run(root) is None
    sleeper = [sys.executable, "-c", "import time; time.sleep(120)", app.MARKER]
    pid = app.start_run(root, cmd=sleeper)
    try:
        assert app.pid_file(root).read_text().strip() == str(pid)
        assert app.current_run(root) == pid
        with pytest.raises(RuntimeError):
            app.start_run(root, cmd=sleeper)
        Project(root).set_status("train", "running")
        assert app.stop_run(root, timeout=10) == pid
        assert app.current_run(root) is None
        assert Project(root).status()["train"]["state"] == "failed"
        assert "stopped from the app" in app.tail(app.process_log(root))
    finally:
        if app.current_run(root) is not None:
            app.stop_run(root)


def test_pid_of_another_process_is_not_ours(tmp_path):
    app = _app()
    app.pid_file(tmp_path).parent.mkdir(parents=True)
    app.pid_file(tmp_path).write_text(f"{os.getpid()}\n")
    later = time.time() + 1000                        # PID file written long after this process started
    os.utime(app.pid_file(tmp_path), (later, later))
    assert app.current_run(tmp_path) is None
    assert app.stop_run(tmp_path) is None             # never touches a process that is not ours


def test_start_run_detaches_from_the_console(tmp_path, monkeypatch):
    app = _app()
    root = tmp_path / "proj"
    root.mkdir()
    (root / "project.yaml").write_text("name: t\n", encoding="utf-8")
    calls = []

    class Done:
        pid = 4_000_001

        def poll(self):
            return 0

    def popen(cmd, **kw):
        calls.append(kw)
        if os.name == "nt" and len(calls) == 1:       # a job object that does not allow breakaway
            raise PermissionError(13, "Access is denied", None, 5)
        return Done()
    monkeypatch.setattr(app.subprocess, "Popen", popen)
    assert app.start_run(root, cmd=["x", app.MARKER]) == Done.pid
    app._PROCS.pop(Done.pid, None)
    if os.name == "nt":
        import subprocess
        first, second = calls[0]["creationflags"], calls[1]["creationflags"]
        assert first & subprocess.CREATE_BREAKAWAY_FROM_JOB and not second & subprocess.CREATE_BREAKAWAY_FROM_JOB
        for fl in (first, second):
            assert fl & subprocess.CREATE_NEW_PROCESS_GROUP and fl & subprocess.CREATE_NO_WINDOW
            assert not fl & subprocess.DETACHED_PROCESS   # would give every DataLoader worker a console window
    else:
        assert calls[0]["start_new_session"]


def test_run_from_a_terminal_blocks_the_app(tmp_path):
    psutil = pytest.importorskip("psutil")
    import subprocess
    app = _app()
    root = _fake_project(tmp_path / "fake")
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        lock = root / "logs" / "run.lock"
        lock.write_text(json.dumps(dict(pid=proc.pid, create_time=psutil.Process(proc.pid).create_time(),
                                        started="2026-09-27 10:00:00", command="run")), encoding="utf-8")
        run = app.active_run(root)
        assert run["pid"] == proc.pid and not run["ours"] and run["what"] == "run"
        with pytest.raises(RuntimeError, match=str(proc.pid)):
            app.start_run(root, cmd=[sys.executable, "-c", "pass", app.MARKER])
        at = _apptest()
        at.session_state["project_dir"] = str(root)
        at.run()
        assert not at.exception
        assert any(f"outside this app (PID {proc.pid}" in i.value for i in at.info)
        b = {x.label: x for x in at.button}
        assert b["Run all"].disabled and b["Plan"].disabled and not b["Stop"].disabled
        lock.unlink()                                  # status.json of a stage run by a live process
        st = json.loads((root / "status.json").read_text())
        st["download"].update(pid=proc.pid, started=time.strftime("%Y-%m-%d %H:%M:%S"))
        (root / "status.json").write_text(json.dumps(st))
        assert app.active_run(root)["pid"] == proc.pid
        st["download"]["started"] = "2020-01-01 00:00:00"   # the PID was reused by a later process
        (root / "status.json").write_text(json.dumps(st))
        assert app.active_run(root) is None
        st["download"]["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
        (root / "status.json").write_text(json.dumps(st))
        assert app.stop_run(root, timeout=10) == proc.pid
        proc.wait(10)
        assert app.active_run(root) is None and Project(root).status()["download"]["state"] == "failed"
    finally:
        if proc.poll() is None:
            proc.kill()


def test_download_size_and_result_files(tmp_path):
    app = _app()
    assert app.YEARS == list(range(2019, 2025))
    est = app.rough_size(3.6, 400, ["GMTCH", "GFCH", "HRCH"])                 # S1 the local way (default)
    assert est["cells"] == 400 and est["aoi_cells"] == 1 and not app.needs_confirmation(est)
    assert 7 < est["eecu_s"] / 3600 < 12 and 10_000 < est["mb"] < 12_000
    gee = app.rough_size(3.6, 400, ["GMTCH", "GFCH", "HRCH"], "AE", "gee")    # S1 on Earth Engine
    assert 20 < gee["eecu_s"] / 3600 < 30 and gee["mb"] == est["mb"]
    assert app.needs_confirmation(app.rough_size(3.6, 400, [], "AE", "gee") | dict(eecu_s=60 * 3600))
    assert set(app.S1_METHOD_LABELS) == {"local", "gee"}
    assert not app.needs_confirmation(app.rough_size(3.6, 16, []))
    assert not app.needs_confirmation(dict(planned=True, cells=500, aoi_cells=200, jobs=0, mb=0, eecu_s=0))
    assert "Downloads complete" in app.size_text(dict(planned=True, cells=5, aoi_cells=1, ring_cells=4, jobs=0))
    from canopy_height.tool.project import DEFAULTS
    assert app.invalid_settings(DEFAULTS) is None and "2019" in app.invalid_settings({**DEFAULTS, "year": 2017})
    assert app.invalid_settings({**DEFAULTS, "epsg": 4326}) and app.invalid_settings({**DEFAULTS, "epsg": 99999})
    root = _fake_project(tmp_path / "fake")
    _tif(root / "maps" / "UNet-SLS.partial.tif", np.zeros((8, 8), np.float32))
    _tif(root / "maps" / "notes.tif", np.zeros((8, 8), np.float32))
    layers = app.result_layers(Project(root))
    assert [n for n, _, _ in layers][:2] == ["RF-SLS", "UNet-SLS"]
    assert not any("partial" in str(f) or f.name == "notes.tif" for _, f, _ in layers)
    items = app.download_items(layers + [("gone", root / "maps" / "missing.tif", "mean")])
    assert len(items) == len(layers) and all(ok for *_, ok in items)
    assert not any(ok for *_, ok in app.download_items(layers, max_mb=0.01))


def test_input_representations_size_and_checkpoints(tmp_path, weights):
    app = _app()
    assert list(INPUTS)[0] == "AE"
    assert app.cell_layers("E") == ["Embedding", "DEM", "GEDI"]
    assert app.cell_layers("T")[0] == "DEM" and "S1_asc_3" in app.cell_layers("T") and "S2_0" in app.cell_layers("TE")
    size = {k: app.rough_size(3.6, 400, [], k) for k in INPUTS}
    assert size["E"]["eecu_s"] < 0.2 * size["AE"]["eecu_s"]                    # no Sentinel-1
    gee = {k: app.rough_size(3.6, 400, [], k, "gee") for k in INPUTS}
    assert size["E"]["eecu_s"] == gee["E"]["eecu_s"] < 0.06 * gee["AE"]["eecu_s"]
    assert 0.8 < gee["T"]["eecu_s"] / gee["AE"]["eecu_s"] < 1.25
    assert 0.8 < size["T"]["eecu_s"] / size["AE"]["eecu_s"] < 1.25             # four S1 seasons ~ the annual S1
    assert size["T"]["mb"] > size["A"]["mb"] and size["TE"]["mb"] > size["AE"]["mb"]
    assert app.stack_mb("T") > app.stack_mb("AE") == app.stack_mb("E")
    t = app.input_table()
    assert list(t["input"]) == list(INPUTS) and list(t["UNet-ALS checkpoint"]) == [v["checkpoint"] for v in
                                                                                    INPUTS.values()]
    assert list(t["four seasons S1 / S2"]) == ["no", "no", "no", "yes", "yes"]
    assert t.set_index("input")["model channels"].to_dict() == dict(AE=76, A=12, E=64, T=45, TE=109)
    assert app.checkpoint("E") == (weights / "UNet-E-ALS.pth", True) and app.checkpoint("XX") == (None, False)
    (weights / "UNet-T-ALS.pth").unlink()
    assert not app.checkpoint("T")[1] and "UNet-T-ALS.pth (missing)" in list(app.input_table()["UNet-ALS checkpoint"])
    own = tmp_path / "mine.pth"
    own.write_bytes(b"")
    assert app.checkpoint("T", own) == (own, True)                            # project.yaml base_model wins
    assert "input T" in app.size_text(dict(planned=False, cells=5, aoi_cells=1, ring_cells=4, jobs=None, mb=1,
                                           eecu_s=10), "T")


def test_report_summary_and_model_inputs(tmp_path):
    app = _app()
    root = _fake_project(tmp_path / "fake")
    p = Project(root)
    rep = root / "report"
    extra = viz.save_quicklook(np.ones((8, 8)), rep / "old.png", 10)
    assert app.quicklook_files(rep) == sorted([extra, rep / "UNet-SLS.png"])           # no summary: every PNG
    (rep / "summary.json").write_text(json.dumps(dict(quicklooks=["UNet-SLS.png", "gone.png", "UNet-SLS.png"],
                                                      notes=["a note"], stale_models={"KG-UNet1": "stack rebuilt"})))
    s, err = app.read_summary(rep)
    assert err is None and app.quicklook_files(rep, s) == [rep / "UNet-SLS.png"]
    assert app.report_notes(s) == ([("KG-UNet1", "stack rebuilt")], ["a note"])
    (rep / "summary.json").write_text("[1, 2]")
    assert app.read_summary(rep)[0] is None and "object" in app.read_summary(rep)[1]
    assert app.results_input(p) == ("AE", None)                             # no trained model: the project's
    for m, res in (("rf-sls", dict(input="AE")), ("unet-sls", dict(model=dict(input="AE"), recipe=dict(input="AE")))):
        p.model_dir(m).mkdir(parents=True)
        (p.model_dir(m) / "result.json").write_text(json.dumps(res))
    assert app.model_inputs(p) == {"rf-sls": "AE", "unet-sls": "AE"}
    (root / "project.yaml").write_text("name: fake\ninput: E\n", encoding="utf-8")
    shown, warn = app.results_input(Project(root))
    assert shown == "AE" and "RF-SLS (AE)" in warn and "with E" in warn


def test_authenticate_in_a_child_process(monkeypatch):
    import subprocess
    app = _app()
    assert app._authenticate(cmd=[sys.executable, "-c", "print('ok')"])[0]
    ok, msg = app._authenticate(cmd=[sys.executable, "-c", "import sys; sys.exit('no browser here')"])
    assert not ok and "no browser here" in msg
    t0 = time.time()
    ok, msg = app._authenticate(timeout=1, cmd=[sys.executable, "-c", "import time; time.sleep(60)"])
    assert not ok and "given up" in msg and time.time() - t0 < 30
    seen = []
    monkeypatch.setattr(app.subprocess, "run", lambda cmd, **kw: seen.append(cmd) or subprocess.CompletedProcess(
        cmd, 0, "", ""))
    assert app._authenticate()[0] and "auth_mode='localhost', force=True" in seen[0][-1]


# ---------------------------------------------------------------------------------------------- the page
def _apptest():
    pytest.importorskip("streamlit")
    pytest.importorskip("folium")
    pytest.importorskip("streamlit_folium")
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(APP), default_timeout=120)


def test_page_without_project():
    at = _apptest()
    at.run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["1 Study area", "2 Settings", "3 Run", "4 Results", "5 Help"]
    assert at.sidebar.radio[0].value == "Open existing"
    help_text = " ".join(m.value for m in at.markdown)
    assert "**AE** (the default)" in help_text and "UNet-TE-ALS.pth" in help_text
    assert "No agreement with GEDI in the report is an independent accuracy" in help_text
    assert "HRCH (ETH) and GFCH (UMD) were calibrated" in help_text


def _selectbox(at, label):
    return [s for s in at.selectbox if s.label == label][0]


def test_create_project_with_an_input_representation(tmp_path, weights):
    at = _apptest()
    at.session_state["mode"] = "Create new"
    at.session_state["new_parent"] = str(tmp_path)
    at.session_state["new_name"] = "newproj"
    at.session_state["aoi"] = _aoi_lonlat()
    at.session_state["aoi_label"] = "test"
    at.run()
    assert not at.exception
    inp = _selectbox(at, "Input representation")
    assert inp.value == "AE" and inp.options[0].startswith("AE:") and len(inp.options) == 5
    assert not _button(at, "Create project").disabled
    (weights / "UNet-TE-ALS.pth").unlink()
    inp.set_value("TE")
    at.run()
    assert any("UNet-TE-ALS.pth" in w.value and "cannot be created" in w.value for w in at.warning)
    assert _button(at, "Create project").disabled
    _selectbox(at, "Input representation").set_value("E")
    at.run()
    assert not _button(at, "Create project").disabled
    _button(at, "Create project").click()
    at.run()
    assert not at.exception
    p = Project(tmp_path / "newproj")
    assert p["input"] == "E" and p["name"] == "newproj"
    assert any("Input representation:" in m.value and "E: Earth embedding only" in m.value for m in at.sidebar.markdown)
    assert _selectbox(at, "Input representation").value == "E"          # settings tab of the new project


def test_page_with_fake_project(tmp_path):
    root = _fake_project(tmp_path / "fake")
    at = _apptest()
    at.session_state["project_dir"] = str(root)
    at.run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Open project:" in text and "fake" in text
    assert any("status.json marks download as running" in w.value for w in at.warning)
    labels = [b.label for b in at.button]
    assert "Run all" in labels and "Stop" in labels
    # settings form -> project.yaml via Project.save
    [n for n in at.number_input if n.label == "Training cells"][0].set_value(16)
    [b for b in at.button if b.label == "Save settings"][0].click()
    at.run()
    assert not at.exception
    p = Project(root)
    assert p["train_cells"] == 16 and p["year"] == 2020 and p["gee_project"] == "my-cloud-project"
    assert p["models"] == ["rf-sls", "unet-sls", "kg-unet1", "kg-unet2"]


def _button(at, label):
    return [b for b in at.button if b.label == label][0]


def test_settings_keep_stored_values(tmp_path):
    root = _fake_project(tmp_path / "fake")
    (root / "project.yaml").write_text("name: fake\nyear: 2025\ngee_project: my-cloud-project\ntrain_cells: 6000\n",
                                       encoding="utf-8")
    raw = _tif(root / "raw" / "x584670y550110_S2.tif", np.zeros((4, 4), np.float32))
    at = _apptest()
    at.session_state["project_dir"] = str(root)
    at.run()
    assert not at.exception
    year = [s for s in at.selectbox if s.label == "Year of the annual inputs"][0]
    assert year.value == 2025 and year.disabled
    assert [n for n in at.number_input if n.label == "Training cells"][0].value == 6000
    assert any("year 2025" in e.value for e in at.error)
    before = (root / "project.yaml").read_text(encoding="utf-8")
    _button(at, "Save settings").click()
    at.run()
    assert not at.exception and (root / "project.yaml").read_text(encoding="utf-8") == before
    assert any("not saved" in e.value for e in at.error)
    raw.unlink()                                      # nothing downloaded: the year can be corrected
    at.run()
    [s for s in at.selectbox if s.label == "Year of the annual inputs"][0].set_value(2021)
    _button(at, "Save settings").click()
    at.run()
    assert not at.exception
    p = Project(root)
    assert p["year"] == 2021 and p["train_cells"] == 6000


def test_settings_form_gee_project_and_als_units(tmp_path):
    root = _fake_project(tmp_path / "fake")
    at = _apptest()
    at.session_state["project_dir"] = str(root)
    at.run()
    [t for t in at.text_input if t.label.startswith("Google Cloud project")][0].set_value("other-project")
    [s for s in at.selectbox if s.label == "Units of the ALS values"][0].set_value(0.01)
    _button(at, "Save settings").click()
    at.run()
    assert not at.exception
    p = Project(root)
    assert p["gee_project"] == "other-project" and p["als_scale"] == 0.01 and p["year"] == 2020
    assert any("other-project" in m.value for m in at.markdown)


def test_replan_after_changing_the_training_region(tmp_path):
    root = _fake_project(tmp_path / "fake")
    (root / "plan" / "landcover.csv").write_text("cell,water,built,tree\n", encoding="utf-8")
    at = _apptest()
    at.session_state["project_dir"] = str(root)
    at.run()
    assert not any("training-region settings changed" in w.value for w in at.warning)
    [n for n in at.number_input if n.label == "Training cells"][0].set_value(16)
    _button(at, "Save settings").click()
    at.run()
    assert any("training-region settings changed" in w.value for w in at.warning)
    assert _button(at, "Re-plan").disabled
    [c for c in at.checkbox if c.label == "Delete the current plan"][0].check()
    at.run()
    _button(at, "Re-plan").click()
    at.run()
    assert not at.exception
    plan = root / "plan"
    assert not (plan / "cells.csv").exists() and not (plan / "grid.json").exists()
    assert (plan / "landcover.csv").exists() and (root / "maps" / "UNet-SLS.tif").exists()
    assert Project(root).status()["plan"]["state"] == "pending"
    assert any("Removed plan/cells.csv" in s.value for s in at.success)


def test_large_download_needs_confirmation(tmp_path):
    root = _fake_project(tmp_path / "fake")
    n = [(i, j) for j in range(10) for i in range(11)]
    pd.DataFrame(dict(cell=[f"x{X0 + 2560 * i:.0f}y{Y1 - 2560 * j:.0f}" for i, j in n],
                      x0=[X0 + 2560 * i for i, _ in n], y1=[Y1 - 2560 * j for _, j in n], col=[i for i, _ in n],
                      row=[j for _, j in n], role="aoi", dist_m=0, water=0.0, built=0.0, tree=0.9, use=True)) \
        .to_csv(root / "plan" / "cells.csv", index=False)
    at = _apptest()
    at.session_state["project_dir"] = str(root)
    at.run()
    assert not at.exception
    assert any("Still to download" in m.value and "110 cells" in m.value for m in at.markdown)
    assert _button(at, "Run all").disabled and _button(at, "Download").disabled and not _button(at, "Stack").disabled
    [c for c in at.checkbox if c.label.startswith("I have checked the size")][0].check()
    at.run()
    assert not _button(at, "Run all").disabled and not _button(at, "Download").disabled
    (root / "plan" / "cells.csv").unlink()            # before planning: rough size of 3000 training cells
    at.run()                                          # 400: about 9 EECU-hours with the local S1 way, no question
    assert not _button(at, "Run all").disabled
    with open(root / "project.yaml", "a", encoding="utf-8") as fh:
        fh.write("train_cells: 3000\n")
    at.run()
    assert _button(at, "Run all").disabled and not _button(at, "Plan").disabled
    assert any(c.label.startswith("I have checked the size") and "3,000 cells" in c.label for c in at.checkbox)


def test_results_with_als_in_centimetres(tmp_path):
    root = _fake_project(tmp_path / "fake")
    _tif(root / "mosaic" / "ALS.tif", np.random.default_rng(4).uniform(0, 4000, (256, 256)).astype(np.float32),
         nodata=np.nan)
    at = _apptest()
    at.session_state["project_dir"] = str(root)
    at.run()
    assert not at.exception
    assert [x for x in at.number_input if x.label == "Colour scale maximum (m)"][0].value > 150
    assert any("ALS reference" in w.value and "Units of the ALS values" in w.value for w in at.warning)
    assert any(b.label.startswith("UNet-SLS (UNet-SLS.tif") for b in at.get("download_button"))


def test_settings_input_representation(tmp_path):
    root = _fake_project(tmp_path / "fake")
    _tif(root / "raw" / "x584670y550110_S2.tif", np.zeros((4, 4), np.float32))    # downloads exist
    at = _apptest()
    at.session_state["project_dir"] = str(root)
    at.run()
    assert not at.exception
    assert _selectbox(at, "Input representation").value == "AE"
    assert any("Input representation:" in m.value and "AE:" in m.value for m in at.sidebar.markdown)
    assert any("Sentinel-1 takes more than 90 %" in c.value for c in at.caption)
    _selectbox(at, "Input representation").set_value("T")
    at.run()
    assert not at.exception
    assert any("from AE to T after downloading is allowed" in i.value and "S1_asc_0" in i.value
               and "trains the models again" in i.value for i in at.info)
    assert not _button(at, "Save settings").disabled
    [n for n in at.number_input if n.label == "Training cells"][0].set_value(16)
    _button(at, "Save settings").click()
    at.run()
    assert not at.exception
    p = Project(root)
    assert p["input"] == "T" and p["train_cells"] == 16 and p["year"] == 2020
    assert any("AE -> T" in s.value and "trains the models again" in s.value for s in at.success)
    assert any("T: four-season Sentinel-1/2" in m.value for m in at.sidebar.markdown)
    assert _selectbox(at, "Input representation").value == "T"
    assert any(s.value == "Canopy-height maps: input T" for s in at.subheader)
    assert any(s.value == "Report: input T" for s in at.subheader)
    (root / "project.yaml").write_text("name: fake\ninput: E\n", encoding="utf-8")   # changed outside the app
    at.run()
    assert _selectbox(at, "Input representation").value == "E"
    (root / "project.yaml").write_text("name: fake\ninput: XX\n", encoding="utf-8")  # not a representation
    at.run()
    assert not at.exception and _selectbox(at, "Input representation").value == "XX"
    assert any("Unknown input representation 'XX'" in w.value for w in at.warning)
    assert any("input 'XX'" in e.value for e in at.error) and _button(at, "Save settings").disabled
    _selectbox(at, "Input representation").set_value("AE")
    at.run()
    _button(at, "Save settings").click()
    at.run()
    assert not at.exception and Project(root)["input"] == "AE"
    assert "input" not in (root / "project.yaml").read_text(encoding="utf-8")        # the default is not written


def test_settings_refused_without_the_checkpoint(tmp_path, weights):
    root = _fake_project(tmp_path / "fake")
    (weights / "UNet-E-ALS.pth").unlink()
    at = _apptest()
    at.session_state["project_dir"] = str(root)
    at.run()
    assert not _button(at, "Save settings").disabled
    _selectbox(at, "Input representation").set_value("E")
    at.run()
    assert not at.exception
    assert any("UNet-E-ALS.pth" in w.value and "cannot be saved" in w.value for w in at.warning)
    assert _button(at, "Save settings").disabled
    before = (root / "project.yaml").read_text(encoding="utf-8")
    (weights / "UNet-E-ALS.pth").write_bytes(b"")
    at.run()
    assert not _button(at, "Save settings").disabled
    (weights / "UNet-E-ALS.pth").unlink()                 # gone between showing the page and pressing Save
    _button(at, "Save settings").click()
    at.run()
    assert not at.exception and (root / "project.yaml").read_text(encoding="utf-8") == before
    assert Project(root)["input"] == "AE" and _button(at, "Save settings").disabled
    _selectbox(at, "Input representation").set_value("A")
    at.run()
    _button(at, "Save settings").click()
    at.run()
    assert Project(root)["input"] == "A"
    (weights / "UNet-A-ALS.pth").unlink()                 # the project's own checkpoint goes missing
    at.run()
    assert any("KG-UNet1/2 cannot be trained" in w.value for w in at.warning)


def test_results_report_summary_notes_and_quicklooks(tmp_path):
    root = _fake_project(tmp_path / "fake")
    rep = root / "report"
    viz.save_quicklook(np.ones((8, 8)), rep / "stale.png", 10)                 # not listed in summary.json
    (rep / "summary.json").write_text(json.dumps(dict(
        input="AE", quicklooks=["UNet-SLS.png"], stale_models={"KG-UNet2": "trained on an older stack"},
        notes=["study_area: 12 GEDI labels > 80 m left out"])), encoding="utf-8")
    at = _apptest()
    at.session_state["project_dir"] = str(root)
    at.run()
    assert not at.exception
    assert any("KG-UNet2: trained on an older stack" in w.value for w in at.warning)
    assert any("Notes of the report" in i.value and "12 GEDI labels > 80 m" in i.value for i in at.info)
    assert any("No agreement with GEDI in this report is an independent accuracy" in c.value
               and "HRCH (ETH) and GFCH (UMD)" in c.value for c in at.caption)
    assert len(at.get("image")) == 1                   # the quick-looks summary.json lists, not every PNG
    assert any(s.value == "Report: input AE" for s in at.subheader)


def test_page_while_a_run_is_going(tmp_path):
    app = _app()
    root = _fake_project(tmp_path / "fake")
    pid = app.start_run(root, cmd=[sys.executable, "-c", "import time; time.sleep(120)", app.MARKER])
    try:
        at = _apptest()
        at.session_state["project_dir"] = str(root)
        at.run()
        assert not at.exception
        assert any(f"Run in progress (PID {pid})" in i.value for i in at.info)
        b = {x.label: x for x in at.button}
        assert b["Run all"].disabled and b["Stack"].disabled and not b["Stop"].disabled
        b["Stop"].click()
        at.run()
        assert not at.exception and app.current_run(root) is None
        assert any(f"stopped the run (PID {pid})" in s.value for s in at.success)
        assert not {x.label: x for x in at.button}["Run all"].disabled
    finally:
        if app.current_run(root) is not None:
            app.stop_run(root)
