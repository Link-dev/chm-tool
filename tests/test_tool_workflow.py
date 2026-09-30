"""Canopy-height tool: stage orchestration, report and command line on a tiny synthetic project.

Six 2.56 km cells (2 x 2 study-area cells + 2 surrounding cells) with plausible inputs derived from one smooth
synthetic canopy-height field; every model is made tiny through train_overrides and trained on the CPU. The
stack module is exercised from raw rasters when it is importable; the other tests write the training stack and
the mosaics themselves.
"""
import csv
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from canopy_height.tool import MODELS, Project, cli, workflow  # noqa: E402
from canopy_height.tool.report import COLUMNS  # noqa: E402

EPSG, X0, Y1, PX = 32650, 584670.0, 550110.0, 256
CELLS = [(0, 0, "aoi"), (1, 0, "aoi"), (0, 1, "aoi"), (1, 1, "aoi"), (2, 0, "ring"), (-1, 1, "ring")]
TINY = {"rf-sls": dict(n_estimators=5, max_depth=8, n_jobs=1),
        "unet-sls": dict(epochs=1),                                  # default val_workers: one spawned worker
        "kg-unet1": dict(epochs=1, val_workers=0),
        "kg-unet2": dict(epochs=1, val_workers=0, kg2_gate=-1)}      # gate -1: teacher terms in epoch 0
BENCH = ["GMTCH", "GFCH", "HRCH"]


# ---------------------------------------------------------------------------------------------- synthetic data
def _height(x, y):
    return 18.0 + 12.0 * np.sin(x / 900.0) * np.cos(y / 700.0)


def _cell(col, row, seed):
    """Raw layers of a cell in the download formats (float64 NaN, DEM int16, ETH / UMD uint8 0 fill)."""
    xs = X0 + col * PX * 10 + 5 + 10 * np.arange(PX)
    ys = Y1 - row * PX * 10 - 5 - 10 * np.arange(PX)
    X, Y = np.meshgrid(xs, ys)
    h = _height(X, Y)
    g = np.random.default_rng(seed)
    k = np.arange(64)[:, None, None]
    lab = np.full((PX, PX), np.nan)
    m = g.random((PX, PX)) < 0.03
    lab[m] = np.clip(h[m] + g.normal(0, 2, m.sum()), 0, None)
    return dict(
        Embedding=0.25 * np.sin(h[None] / 8.0 + k) + 0.01 * g.standard_normal((64, PX, PX)),
        DEM=(150 + (X - X0) / 50 + (Y1 - Y) / 80).astype(np.int16)[None],
        S1=np.stack([-9 - h / 8, -15 - h / 6]) + 0.3 * g.standard_normal((2, PX, PX)),
        S2=np.stack([200 + 3000 * np.exp(-h / (10.0 + b)) for b in range(9)]) + 20 * g.standard_normal((9, PX, PX)),
        GEDI=lab[None],
        ETH=np.clip(0.9 * h + g.normal(0, 2, h.shape), 1, 254).astype(np.uint8)[None],
        UMD=np.clip(0.8 * h + g.normal(0, 3, h.shape), 1, 254).astype(np.uint8)[None],
        GMTCH=(0.85 * h + g.normal(0, 2, h.shape)).astype(np.float32)[None],
        ALS=(h + g.normal(0, 1, h.shape)).astype(np.float32),
    )


def _encode_x(c):
    """76-band uint16 input as the paper's pipeline encodes it (common.encode_x)."""
    x = np.concatenate([(c["Embedding"] + 1) * 1e4, c["DEM"], (c["S1"] + 50) * 100, c["S2"]])
    return np.clip(np.nan_to_num(x, nan=0.0), 0, 65535).astype(np.uint16)


def _enc999(a):
    return np.nan_to_num(a.astype(np.float32), nan=-999)


def _write(path, a, dtype, nodata, x0, y1, crs=EPSG):
    a = np.asarray(a)
    a = a[None] if a.ndim == 2 else a
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=a.shape[2], height=a.shape[1], count=a.shape[0],
                       dtype=dtype, nodata=nodata, crs=f"EPSG:{crs}", transform=from_origin(x0, y1, 10, 10)) as d:
        d.write(a.astype(dtype))


def _aoi_lonlat():
    from pyproj import Transformer
    t = Transformer.from_crs(EPSG, 4326, always_xy=True)
    xa, xb, ya, yb = X0 + 300, X0 + 2 * PX * 10 - 300, Y1 - 2 * PX * 10 + 300, Y1 - 300
    return [list(t.transform(x, y)) for x, y in ((xa, ya), (xb, ya), (xb, yb), (xa, yb), (xa, ya))]


def _project(root, **settings):
    """Project folder with project.yaml, aoi.geojson, plan/grid.json and plan/cells.csv (no GEE)."""
    root = Path(root)
    (root / "plan").mkdir(parents=True, exist_ok=True)
    cfg = dict(name="synthetic", device="cpu", train_overrides=TINY, benchmarks=BENCH)
    cfg.update(settings)
    (root / "project.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    try:
        ring = _aoi_lonlat()
    except ImportError:
        ring = [[117.77, 4.96], [117.81, 4.96], [117.81, 4.99], [117.77, 4.99], [117.77, 4.96]]
    gj = dict(type="FeatureCollection", features=[dict(type="Feature", properties={},
                                                        geometry=dict(type="Polygon", coordinates=[ring]))])
    (root / "aoi.geojson").write_text(json.dumps(gj), encoding="utf-8")
    json.dump(dict(epsg=EPSG, x0=X0, y1=Y1, res=10.0, cell_px=PX), open(root / "plan" / "grid.json", "w"))
    with open(root / "plan" / "cells.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "x0", "y1", "col", "row", "role", "dist_m", "water", "built", "tree", "use"])
        for col, row, role in CELLS:
            x0, y1 = X0 + col * PX * 10, Y1 - row * PX * 10
            w.writerow([f"x{round(x0)}y{round(y1)}", x0, y1, col, row, role, 0 if role == "aoi" else 300,
                        0.0, 0.0, 0.9, True])
    return Project(root)


def _write_stacks_and_mosaics(p):
    """What the stack stage produces: training stack (cells.csv order) and 2 x 2-cell study-area mosaics."""
    cells = [_cell(c, r, i) for i, (c, r, _) in enumerate(CELLS)]
    p.path("stacks").mkdir(exist_ok=True)
    np.save(p.stack_part("x", 1), np.stack([_encode_x(c) for c in cells]))
    np.save(p.stack_part("GEDI", 1), np.stack([_enc999(c["GEDI"][0]) for c in cells]))
    # train_index.csv and hashes.json complete the stack (stack.stack_id), against which the report checks models
    with open(p.path("stacks", "train_index.csv"), "w", newline="") as fh:
        csv.writer(fh).writerows([["chip", "part", "row_in_part", "cell"]] +
                                 [[i, 1, i, f"c{i}"] for i in range(len(cells))])
    files = {p.stack_part(k, 1).name: dict(data_sha256=f"synthetic-{k}") for k in ("x", "GEDI")}
    p.path("stacks", "hashes.json").write_text(json.dumps(dict(chips=len(cells), part_size=400, files=files)))
    big = lambda k: np.block([[cells[0][k], cells[1][k]], [cells[2][k], cells[3][k]]])  # noqa: E731
    ann = np.concatenate([np.concatenate([_encode_x(cells[0]), _encode_x(cells[1])], 2),
                          np.concatenate([_encode_x(cells[2]), _encode_x(cells[3])], 2)], 1)
    _write(p.mosaic("annual"), ann, "uint16", 0, X0, Y1)
    _write(p.mosaic("GEDI"), big("GEDI").astype(np.float32), "float32", np.nan, X0, Y1)
    for b, k in (("HRCH", "ETH"), ("GFCH", "UMD"), ("GMTCH", "GMTCH")):
        _write(p.mosaic(b), big(k).astype(np.float32), "float32", np.nan, X0, Y1)
    als = big("ALS").astype(np.float32)
    als[:100] = np.nan                                                   # ALS covers part of the area
    _write(p.mosaic("ALS"), als, "float32", np.nan, X0, Y1)
    yy, xx = np.mgrid[:2 * PX, :2 * PX]
    _write(p.mosaic("aoi_mask"), ((yy - 256) ** 2 + (xx - 256) ** 2 < 230 ** 2).astype(np.uint8), "uint8", None,
           X0, Y1)


def _release(root):
    """Close the log handlers of a project (Windows keeps open files locked)."""
    lg = logging.getLogger(f"chm-tool:{Path(root).resolve()}")
    for h in list(lg.handlers):
        h.close()
        lg.removeHandler(h)


def _log(p):
    return p.log_file.read_text(encoding="utf-8")


def _check_maps_and_report(p, models=MODELS):
    with rasterio.open(p.mosaic("annual")) as s:
        grid = (s.width, s.height, s.transform, s.crs)
    with rasterio.open(p.mosaic("aoi_mask")) as s:
        inside = s.read(1) == 1
    for m in models:
        with rasterio.open(p.map_file(m)) as s:
            a = s.read(1)
            assert (s.width, s.height, s.transform, s.crs) == grid and s.count == 1 and s.dtypes[0] == "float32"
            assert s.crs.to_epsg() == EPSG
        assert np.isnan(a[~inside]).all() and np.isfinite(a[inside]).all()
    assert not list(p.path("maps").glob("*.partial.tif"))
    with open(p.path("report", "metrics.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0].keys()) == COLUMNS
    names = [workflow.MODEL_NAMES[m] for m in models]
    got = {(r["section"], r["layer"]) for r in rows}
    assert {("gedi_val", n) for n in names} <= got
    assert {("study_area", n) for n in names + BENCH + ["GEDI"]} <= got
    for r in rows:
        if r["section"] in ("gedi_val", "als"):
            assert int(r["n"]) > 0 and float(r["rmse"]) >= 0
    summary = json.loads(p.path("report", "summary.json").read_text(),
                         parse_constant=lambda c: pytest.fail(f"non-JSON constant {c}"))
    assert summary["colour_scale"]["vmax"] >= 40
    for f in summary["quicklooks"]:
        assert p.path("report", f).stat().st_size > 1000
    assert len(summary["quicklooks"]) == len(models) + len(BENCH) + int(p.mosaic("ALS").exists())
    return rows, summary


# ---------------------------------------------------------------------------------------------- tests
def test_status_failure_cli_and_empty_report(tmp_path, capsys):
    """No PyTorch needed: stage status on failure, plan kept, argument checks, CLI, report without inputs."""
    root = tmp_path / "proj"
    p = _project(root)
    try:
        rows = workflow.plan(p)
        assert len(rows) == 6 and p.status()["plan"]["state"] == "done"
        assert p.status()["plan"]["message"].startswith("4 study-area cells, 2 surrounding cells")
        with pytest.raises(FileNotFoundError, match="training stack is missing"):
            workflow.train(p)
        st = p.status()["train"]
        assert st["state"] == "failed" and "training stack is missing" in st["message"] and st["pid"] == os.getpid()
        assert "Traceback" in _log(p) and "== train: FAILED" in _log(p)
        assert workflow.model_order(["kg-unet2", "rf-sls", "unet-sls"]) == ["rf-sls", "unet-sls", "kg-unet2"]
        with pytest.raises(ValueError):
            workflow.model_order(["unet-als"])
        with pytest.raises(ValueError):
            workflow.run(p, "report", "plan")

        summary = workflow.report(p)                    # nothing to report: notes, header-only metrics.csv
        assert summary["n_rows"] == 0 and summary["notes"] and p.status()["report"]["state"] == "done"
        assert p.path("report", "metrics.csv").read_text().strip() == ",".join(COLUMNS)

        # benchmarks only (no model yet): study-area rows and quick-looks
        yy, xx = np.mgrid[:2 * PX, :2 * PX]
        _write(p.mosaic("aoi_mask"), (xx > 100).astype(np.uint8), "uint8", None, X0, Y1)
        _write(p.mosaic("HRCH"), (20 + np.sin(yy / 30.0)).astype(np.float32), "float32", np.nan, X0, Y1)
        summary = workflow.report(p)
        assert summary["quicklooks"] == ["HRCH.png"] and summary["study_area"]["layers"]["HRCH"]["n_px"] == 2 * PX * (2 * PX - 101)
        assert any("GFCH" in n for n in summary["notes"])
    finally:
        _release(root)

    capsys.readouterr()
    assert cli.main(["status", str(root)]) == 0
    out = capsys.readouterr().out
    assert "train" in out and "failed" in out
    assert cli.main(["train", str(root)]) == 1
    assert "training stack is missing" in capsys.readouterr().err
    _release(root)
    src = str(Path(workflow.__file__).resolve().parents[2])
    env = dict(os.environ, PYTHONPATH=src + os.pathsep + os.environ.get("PYTHONPATH", ""), PYTHONNOUSERSITE="1")
    r = subprocess.run([sys.executable, "-m", "canopy_height.tool", "status", str(root)], capture_output=True,
                       text=True, env=env)
    assert r.returncode == 0 and "report" in r.stdout and "done" in r.stdout


def test_download_stage_and_import_cells(tmp_path, monkeypatch):
    """Download stage with a stubbed download.run_download (no Earth Engine); import-cells via the CLI."""
    dl = pytest.importorskip("canopy_height.tool.download")
    p = _project(tmp_path / "proj")
    try:
        calls = []

        def fake(project, workers=None, **kw):
            calls.append(workers)
            return dict(cells=6, jobs=2, ok=1, gmtch=0,
                        failed=[dict(cell="x584670y550110", layers="S1", error="FAILED HttpError 429")])
        monkeypatch.setattr(dl, "run_download", fake)
        with pytest.raises(RuntimeError, match="1 download jobs failed.*x584670y550110 S1"):
            workflow.download(p, workers=1)
        assert calls == [1] and p.status()["download"]["state"] == "failed"
        monkeypatch.setattr(dl, "run_download", lambda project, workers=None: dict(cells=6, jobs=0, ok=0, gmtch=0,
                                                                                  failed=[]))
        workflow.download(p)
        st = p.status()["download"]
        assert st["state"] == "done" and st["message"] == "6 cells, 0 downloads, 0 GMTCH derived"
    finally:
        _release(p.root)

    # import-cells: two existing pieces, the pipeline's rows table layout
    pytest.importorskip("canopy_height.tool.grid")
    q = _project(tmp_path / "imported")
    q.cells_file.unlink()
    rows = tmp_path / "rows.csv"
    with open(rows, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["piece", "tile", "kind", "min_dist_m", "epsg", "tile_x0", "tile_y1", "padded", "x_src", "lab_src"])
        w.writerow(["Px584670y550110_2020", 1, "test", 0, EPSG, X0, Y1, False, tmp_path, tmp_path])
        w.writerow(["Px587230y550110_2020", 1, "near", 10, EPSG, X0 + 2560, Y1, False, tmp_path, tmp_path])
    try:
        assert cli.main(["import-cells", str(q.root), "--rows", str(rows)]) == 0
        cells = workflow.read_cells(q)
        assert [c["role"] for c in cells] == ["aoi", "ring"] and cells[0]["cell"] == "Px584670y550110_2020"
        st = q.status()["plan"]
        assert st["state"] == "done" and st["message"] == "imported from rows.csv: 1 study-area cells, 1 surrounding cells"
    finally:
        _release(q.root)

    # init: settings from the command line, paper defaults otherwise
    new = tmp_path / "new"
    assert cli.main(["init", str(new), "--aoi", str(p.aoi_file), "--year", "2021", "--train-cells", "16",
                     "--models", "unet-sls", "rf-sls", "--benchmarks", "HRCH", "--gee-project", "my-project"]) == 0
    n = Project(new)
    assert (n["year"], n["train_cells"], n["models"], n["benchmarks"], n["gee_project"]) == \
           (2021, 16, ["unet-sls", "rf-sls"], ["HRCH"], "my-project")
    assert n["ring_max_km"] == 50 and n.aoi_file.exists()
    assert set(yaml.safe_load(n.path("project.yaml").read_text())) == {"year", "gee_project", "train_cells",
                                                                        "models", "benchmarks"}
    assert cli.main(["init", str(new), "--aoi", str(p.aoi_file)]) == 1          # refuses an existing project


def test_train_predict_report(tmp_path):
    """train -> predict -> report on a written stack and mosaics, all four models on the CPU; then resume."""
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    root = tmp_path / "proj"
    p = _project(root)
    try:
        if not _base_model_available(p):
            pytest.skip("UNet-ALS checkpoint not found (weights/source/UNet-ALS.pth or $CHM_UNET_ALS)")
        _write_stacks_and_mosaics(p)
        d = p.model_dir("unet-sls")
        d.mkdir(parents=True)
        (d / "history.csv").write_text("partial run")                 # files of a crashed run ...
        (d / "notes.txt").write_text("user file")                      # ... and a file that is not the recipe's
        st = workflow.run(p, "train", "report")
        assert all(st[s]["state"] == "done" for s in ("train", "predict", "report"))
        for m in MODELS:
            assert (p.model_dir(m) / "result.json").exists() and p.model_file(m).exists()
        assert (d / "notes.txt").read_text() == "user file"
        assert (d / "history.csv").read_text().startswith("epoch")
        log = _log(p)
        assert "removed files of an unfinished run: history.csv" in log
        assert "batch size set to" in log and "not meaningful" in log
        res = json.load(open(p.model_dir("kg-unet2") / "result.json"))
        assert res["teacher_sls"] and Path(res["teacher_sls"]).name == "model.pth"
        rows, summary = _check_maps_and_report(p)
        assert {("als", n) for n in ["RF-SLS", "UNet-SLS", "KG-UNet1", "KG-UNet2"] + BENCH} <= \
               {(r["section"], r["layer"]) for r in rows}
        assert summary["gedi_validation"]["split"] == "models/unet-sls/split.npz"

        # resume: finished models and up-to-date maps are kept
        mt = {m: p.map_file(m).stat().st_mtime for m in MODELS}
        workflow.run(p, "train", "predict")
        assert {m: p.map_file(m).stat().st_mtime for m in MODELS} == mt
        assert "finished run in" in _log(p) and "is up to date, kept" in _log(p)
    finally:
        _release(root)


def test_stack_to_report_from_raw(tmp_path):
    """stack -> train -> predict -> report from raw rasters in the download formats (needs tool.stack)."""
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    st_mod = pytest.importorskip("canopy_height.tool.stack")
    if not all(hasattr(st_mod, f) for f in ("build_train_stack", "build_mosaics")):
        pytest.skip("canopy_height.tool.stack is incomplete")
    root = tmp_path / "proj"
    p = _project(root, models=["rf-sls", "unet-sls", "kg-unet1", "kg-unet2"])
    try:
        if not _base_model_available(p):
            pytest.skip("UNet-ALS checkpoint not found")
        cells = []
        for i, (col, row, role) in enumerate(CELLS):
            c = _cell(col, row, i)
            cells.append(c)
            x0, y1 = X0 + col * PX * 10, Y1 - row * PX * 10
            name = f"x{round(x0)}y{round(y1)}"
            for k in ("Embedding", "S1", "S2", "GEDI"):
                _write(p.raw(name, k), c[k], "float64", np.nan, x0, y1)
            _write(p.raw(name, "DEM"), c["DEM"], "int16", None, x0, y1)
            if role == "aoi":
                _write(p.raw(name, "ETH"), c["ETH"], "uint8", 0, x0, y1)
                _write(p.raw(name, "UMD"), c["UMD"], "uint8", 0, x0, y1)
                _write(p.raw(name, "GMTCH"), c["GMTCH"], "float32", np.nan, x0, y1)
        st = workflow.run(p, "stack", "report")
        assert all(st[s]["state"] == "done" for s in ("stack", "train", "predict", "report"))
        X = np.load(p.stack_files("x")[0], mmap_mode="r")
        Y = np.load(p.stack_files("GEDI")[0], mmap_mode="r")
        assert X.shape == (6, 76, PX, PX) and X.dtype == np.uint16 and Y.shape == (6, PX, PX) and Y.dtype == np.float32
        assert np.array_equal(X[0], _encode_x(cells[0])) and np.array_equal(Y[5], _enc999(cells[5]["GEDI"][0]))
        with rasterio.open(p.mosaic("annual")) as s:
            assert s.count == 76 and (s.width, s.height) == (2 * PX, 2 * PX) and s.crs.to_epsg() == EPSG
            assert (s.transform.c, s.transform.f) == (X0, Y1)
        with rasterio.open(p.mosaic("aoi_mask")) as s:
            assert (s.read(1) == 1).any()
        _check_maps_and_report(p)
    finally:
        _release(root)


def _base_model_available(p):
    try:
        p.base_model()
        return True
    except FileNotFoundError:
        return False
