"""Canopy-height tool: features of the stage orchestration (workflow.py) on small synthetic data, CPU only.

1. windowed prediction of large study areas (BLOCK / HALO made small): RF-SLS identical to rf.predict_rf_raster on
   the study-area pixels, UNet windows put back in place, blocks without study-area pixels skipped, the
   whole-raster path identical to predict.predict_array;
2. stale models: a rebuilt training stack retires every model to models/<model>.stale-* (nothing deleted) and
   trains it again; a retrained UNet-SLS retires KG-UNet2 (its GEDI teacher);
3. the input representation T (DEM + seasonal composites) from raw rasters to the report, and the refusal of a
   UNet-ALS checkpoint of another representation;
4. the run lock held by another live process;
5. the GEDI label guard of the train stage.

The synthetic cells and helpers are those of test_tool_workflow.py. The UNet-ALS checkpoints are untrained networks
saved in a temporary weights folder ($CHM_WEIGHTS), so no released weights are needed.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

import test_tool_workflow as tw  # noqa: E402  (synthetic project of test_tool_workflow.py)
from canopy_height.tool import MODEL_NAMES, MODELS, workflow  # noqa: E402
from canopy_height.tool.project import BENCHMARKS, INPUTS, SEASONAL_LAYERS, input_layers, lock_owner  # noqa: E402

# tiny models: one epoch, small batches, no DataLoader worker processes, teacher terms of KG-UNet2 from epoch 0
FAST = {"rf-sls": dict(n_estimators=5, max_depth=8, n_jobs=1),
        "unet-sls": dict(epochs=1, batch_size=2, val_workers=0),
        "kg-unet1": dict(epochs=1, batch_size=2, val_workers=0),
        "kg-unet2": dict(epochs=1, batch_size=2, val_workers=0, kg2_gate=-1)}
SEASONAL_BANDS = {"S1": ["VV", "VH"], "S2": ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12"]}
H1, W1 = 300, 400                                   # mosaic of the windowed-prediction tests


# ---------------------------------------------------------------------------------------------- helpers
@pytest.fixture
def weights(tmp_path, monkeypatch):
    """Temporary weights folder used as $CHM_WEIGHTS (project.default_base_model)."""
    w = tmp_path / "weights"
    monkeypatch.setenv("CHM_WEIGHTS", str(w))
    monkeypatch.delenv("CHM_UNET_ALS", raising=False)
    return w


def _fake_base(weights, inp, layout=None):
    """Untrained UNet saved as the UNet-ALS checkpoint of representation `inp` in weights/source (with the input
    layout `layout`, default inp)."""
    import torch
    from canopy_height import stats
    from canopy_height.models import build_model, model_config, save_checkpoint
    lay = layout or inp
    torch.manual_seed(7)
    cfg = model_config(lay)
    f = Path(weights) / "source" / INPUTS[inp]["checkpoint"]
    f.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(f, build_model(cfg, *stats.for_input(lay)), cfg, note="untrained test checkpoint")
    return f


def _heights(col, row):
    xs = tw.X0 + col * tw.PX * 10 + 5 + 10 * np.arange(tw.PX)
    ys = tw.Y1 - row * tw.PX * 10 - 5 - 10 * np.arange(tw.PX)
    return tw._height(*np.meshgrid(xs, ys))


def _seasonal_layer(layer, h, g):
    """Seasonal composite of season k as downloaded: S1_asc_k VV, VH in dB (around -15), S2_k B2 ... B12 as 0-1
    reflectance; float64."""
    k = int(layer[-1])
    if layer.startswith("S1"):
        return np.stack([-9 - h / 8, -15 - h / 6]) + 0.4 * k + 0.3 * g.standard_normal((2,) + h.shape)
    return np.stack([0.02 + 0.3 * np.exp(-h / (10.0 + b)) + 0.01 * k for b in range(9)]) + \
        0.005 * g.standard_normal((9,) + h.shape)


def _write_raw(p, gedi=None):
    """raw/<cell>_<layer>.tif of the cells of test_tool_workflow in the download formats: the layers of the
    project's input representation, GEDI (or gedi(i, heights) -> [256, 256]) and, for the study-area cells, the
    benchmark layers of project['benchmarks']."""
    layers = input_layers(p["input"])
    for i, (col, row, role) in enumerate(tw.CELLS):
        c, h, g = tw._cell(col, row, i), _heights(col, row), np.random.default_rng(1000 + i)
        x0, y1 = tw.X0 + col * tw.PX * 10, tw.Y1 - row * tw.PX * 10
        name = f"x{round(x0)}y{round(y1)}"
        for k in layers:
            f = p.raw(name, k)
            if k == "DEM":
                tw._write(f, c[k], "int16", None, x0, y1)
            elif k in SEASONAL_LAYERS:
                tw._write(f, _seasonal_layer(k, h, g), "float64", np.nan, x0, y1)
                with rasterio.open(f, "r+") as d:                    # band names as the paper's seasonal rasters
                    for j, b in enumerate(SEASONAL_BANDS[k[:2]], 1):
                        d.set_band_description(j, b)
            else:
                tw._write(f, c[k], "float64", np.nan, x0, y1)
        tw._write(p.raw(name, "GEDI"), c["GEDI"] if gedi is None else gedi(i, h)[None], "float64", np.nan, x0, y1)
        if role == "aoi":
            for product in p["benchmarks"]:
                k = BENCHMARKS[product]
                if k == "GMTCH":
                    tw._write(p.raw(name, k), c[k], "float32", np.nan, x0, y1)
                else:
                    tw._write(p.raw(name, k), c[k], "uint8", 0, x0, y1)


def _stale(p, m):
    return sorted(p.path("models").glob(f"{m}.stale-*"))


def _mtimes(p):
    return {m: p.model_file(m).stat().st_mtime_ns for m in MODELS if p.model_file(m).exists()}


# ---------------------------------------------------------------------------------------------- 1 windowed prediction
def _study_area():
    """Irregular study-area mask of the 300 x 400 mosaic: an ellipse with a hole and a small separate part; with
    128 px blocks, 5 of the 12 blocks hold no study-area pixel."""
    yy, xx = np.mgrid[:H1, :W1]
    inside = ((yy - 130) / 110.0) ** 2 + ((xx - 170) / 150.0) ** 2 < 1          # rows 20-240, cols 20-320
    inside &= (yy - 150) ** 2 + (xx - 250) ** 2 >= 15 ** 2                        # a hole
    inside |= (yy >= 270) & (yy < 290) & (xx >= 388) & (xx < 398)                 # a small separate part
    return inside


def _mosaic(d, inp="AE", seed=0):
    """mosaic annual.tif (76-band uint16; for T only the DEM band) and, for T, seasonal.tif (44-band uint16) with
    a patch without any input inside the study area and, for AE, a patch where only the embedding is missing."""
    g = np.random.default_rng(seed)
    if inp == "T":
        a = np.zeros((76, H1, W1), np.uint16)
        a[64] = g.integers(1, 3000, (H1, W1), dtype=np.uint16)
    else:
        a = g.integers(1, 40000, (76, H1, W1), dtype=np.uint16)
        a[:64, 150:170, 150:200] = 0                                             # embedding missing only
    a[:, 60:90, 100:160] = 0                                                     # no input at all -> NaN
    tw._write(d / "annual.tif", a, "uint16", 0, tw.X0, tw.Y1)
    s = None
    if inp == "T":
        s = g.integers(1, 40000, (44, H1, W1), dtype=np.uint16)
        s[:, 60:90, 100:160] = 0
        tw._write(d / "seasonal.tif", s, "uint16", 0, tw.X0, tw.Y1)
    return a, s


def _blocks(inside, block):
    """(r0, c0) of the core blocks with study-area pixels, in the order the workflow visits them."""
    return [(r, c) for r in range(0, H1, block) for c in range(0, W1, block) if inside[r:r + block, c:c + block].any()]


def _covered(inside, block):
    m = np.zeros_like(inside)
    for r, c in _blocks(inside, block):
        m[r:r + block, c:c + block] = True
    return m


@pytest.mark.parametrize("inp", ["AE", "T"])
def test_windowed_rf_map_equals_predict_rf_raster(tmp_path, monkeypatch, inp):
    """RF-SLS map block by block (BLOCK 128): identical to rf.predict_rf_raster on every study-area pixel, NaN
    outside and where every annual band is 0; blocks without study-area pixels are not read."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestRegressor
    from canopy_height import channels
    from canopy_height.rf import encode, predict_rf_raster
    a, s = _mosaic(tmp_path, inp)
    inside = _study_area()
    nodata = (a == 0).all(axis=0)
    spec = channels.INPUTS[inp]
    g = np.random.default_rng(3)
    X = g.integers(0, 40000, (500, spec["n_channels"]), dtype=np.uint16)
    rf = RandomForestRegressor(n_estimators=5, max_depth=6, random_state=0, n_jobs=1)
    rf.fit(encode(X, "paper", spec["n_embed"]), g.uniform(0, 45, 500))
    meta = {"input": inp, "encoding": "paper"}
    annual, seasonal = tmp_path / "annual.tif", (tmp_path / "seasonal.tif" if s is not None else None)
    ref = predict_rf_raster(rf, meta, a, s)

    reads, read = [], workflow._read
    monkeypatch.setattr(workflow, "_read", lambda path, window=None: reads.append((Path(path).name, window))
                        or read(path, window))
    monkeypatch.setattr(workflow, "BLOCK", 128)
    monkeypatch.setattr(workflow, "HALO", 64)
    got = workflow.predict_rf_map(rf, meta, annual, seasonal, inside)
    blocks = _blocks(inside, 128)
    assert 0 < len(blocks) < 12                                                   # the fixture skips blocks
    assert [(int(w.row_off), int(w.col_off)) for n, w in reads if n == "annual.tif"] == blocks
    assert all((int(w.height), int(w.width)) == (min(128, H1 - int(w.row_off)), min(128, W1 - int(w.col_off)))
               for _, w in reads)                                                 # core blocks only, no halo
    assert len(reads) == len(blocks) * (1 if s is None else 2)
    assert got.shape == (H1, W1) and got.dtype == np.float32
    assert np.array_equal(got[inside], ref[inside], equal_nan=True)
    assert np.isnan(got[~inside]).all()
    assert (inside & nodata).any() and np.isnan(got[inside & nodata]).all()
    assert np.isfinite(got[inside & ~nodata]).all()

    # prediction in chunks of rows and the one-block path (default BLOCK) give the same map
    assert np.array_equal(workflow.predict_rf_map(rf, meta, annual, seasonal, inside, chunk=1001), got,
                          equal_nan=True)
    monkeypatch.setattr(workflow, "BLOCK", 2048)
    reads.clear()
    assert np.array_equal(workflow.predict_rf_map(rf, meta, annual, seasonal, inside), got, equal_nan=True)
    assert [(int(w.row_off), int(w.col_off), int(w.height), int(w.width)) for n, w in reads
            if n == "annual.tif"] == [(0, 0, H1, W1)]


def _counting_predict_array(monkeypatch):
    """Wrap canopy_height.predict.predict_array (imported by predict_unet_map at call time); returns the list of
    the (H, W) of every call and the original function."""
    from canopy_height import predict as pr
    calls, real = [], pr.predict_array

    def counting(model, config, annual, seasonal=None, **kw):
        calls.append(tuple(annual.shape[1:]))
        return real(model, config, annual, seasonal, **kw)
    monkeypatch.setattr(pr, "predict_array", counting)
    return calls, real


def test_windowed_unet_map(tmp_path, monkeypatch):
    """UNet map (NormalizingUNet, CPU): the whole raster at once equals predict.predict_array exactly; block by
    block (BLOCK 128, HALO 64) finite on every study-area pixel with inputs, NaN where every annual band is 0 and on
    the skipped blocks, one predict_array call per block with study-area pixels."""
    torch = pytest.importorskip("torch")
    from canopy_height import stats
    from canopy_height.models import build_model, model_config
    a, _ = _mosaic(tmp_path, "AE")
    inside = _study_area()
    nodata = (a == 0).all(axis=0)
    torch.manual_seed(0)
    cfg = model_config("AE")
    model = build_model(cfg, *stats.for_input("AE")).eval()
    dev = torch.device("cpu")
    calls, real = _counting_predict_array(monkeypatch)

    whole = workflow.predict_unet_map(model, cfg, tmp_path / "annual.tif", None, inside, dev)
    assert calls == [(H1, W1)]                                  # 300 x 400 <= BLOCK + 2 HALO: predicted whole
    assert np.array_equal(whole, real(model, cfg, a, None, device=dev), equal_nan=True)

    monkeypatch.setattr(workflow, "BLOCK", 128)
    monkeypatch.setattr(workflow, "HALO", 64)
    calls.clear()
    out = workflow.predict_unet_map(model, cfg, tmp_path / "annual.tif", None, inside, dev)
    blocks = _blocks(inside, 128)
    assert len(calls) == len(blocks) < 12
    assert all(h <= 128 + 2 * 64 and w <= 128 + 2 * 64 for h, w in calls)
    assert out.shape == (H1, W1) and out.dtype == np.float32
    assert np.isfinite(out[inside & ~nodata]).all()
    assert (inside & nodata).any() and np.isnan(out[nodata]).all()
    assert np.isnan(out[~_covered(inside, 128)]).all()
    # halo context changes the values near block edges only a little
    both = inside & ~nodata
    assert np.corrcoef(out[both], whole[both])[0, 1] > 0.5


def _pick(ch):
    """Stand-in model that returns channel `ch` of its (raw, not normalised) input: a map that shows where every
    predicted pixel came from."""
    import torch

    class Pick(torch.nn.Module):
        def forward(self, x):
            return x[:, ch:ch + 1].clone()
    return Pick()


@pytest.mark.parametrize("inp,ch,block,halo", [("AE", 64, 128, 64),     # windows of one 256 px tile
                                               ("AE", 70, 200, 40),     # windows of 280 px: blended tiles
                                               ("T", 6, 128, 64)])      # seasonal windows (T channel 6 = S1 VV MAM)
def test_windowed_unet_map_puts_windows_in_place(tmp_path, monkeypatch, inp, ch, block, halo):
    torch = pytest.importorskip("torch")
    a, s = _mosaic(tmp_path, inp)
    inside = _study_area()
    nodata = (a == 0).all(axis=0)
    expect = (a[ch] if inp == "AE" else s[ch - 1]).astype(np.float32)
    calls, _ = _counting_predict_array(monkeypatch)
    monkeypatch.setattr(workflow, "BLOCK", block)
    monkeypatch.setattr(workflow, "HALO", halo)
    out = workflow.predict_unet_map(_pick(ch), {"input": inp}, tmp_path / "annual.tif",
                                    tmp_path / "seasonal.tif" if s is not None else None, inside,
                                    torch.device("cpu"))
    cov = _covered(inside, block)
    assert len(calls) == len(_blocks(inside, block))
    assert np.array_equal(out[cov & ~nodata], expect[cov & ~nodata])
    assert np.isnan(out[nodata | ~cov]).all()


# ---------------------------------------------------------------------------------------------- 2 stale models
def test_stale_models_are_retired_and_trained_again(tmp_path, weights):
    """stack -> train (all four) -> a re-downloaded input raster -> stack rebuilt -> every model retired and trained
    again -> kept on the next run; a removed UNet-SLS is trained again and retires KG-UNet2 (its GEDI teacher)."""
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    from canopy_height.tool import stack as st
    _fake_base(weights, "AE")
    root = tmp_path / "proj"
    p = tw._project(root, train_overrides=FAST, benchmarks=[])
    try:
        _write_raw(p)
        workflow.stack(p)
        sid = st.stack_id(p)
        assert sid
        assert workflow.train(p) == MODELS
        assert p.status()["train"]["message"] == "trained RF-SLS, UNet-SLS, KG-UNet1, KG-UNet2"
        for m in MODELS:
            assert (p.model_dir(m) / "stack_id.txt").read_text(encoding="utf-8").strip() == sid
        assert not list(p.path("models").glob("*.stale-*"))

        # an input raster of a used cell re-downloaded with other values: the stack is rebuilt, its id changes
        f = p.raw(f"x{round(tw.X0)}y{round(tw.Y1)}", "S2")
        before = f.stat().st_mtime_ns
        tw._write(f, tw._cell(0, 0, 0)["S2"] + 250.0, "float64", np.nan, tw.X0, tw.Y1)
        assert f.stat().st_mtime_ns > before
        workflow.stack(p)
        sid2 = st.stack_id(p)
        assert sid2 and sid2 != sid
        assert "stack: rebuilding the training stack (1 input rasters changed since the stack was built" in tw._log(p)

        # every model is retired (kept as models/<m>.stale-*) and trained on the new stack
        workflow.train(p)
        assert p.status()["train"]["message"] == "trained RF-SLS, UNet-SLS, KG-UNet1, KG-UNet2"
        log = tw._log(p)
        for m in MODELS:
            old = _stale(p, m)
            assert len(old) == 1
            d = old[0]
            assert (d / "stack_id.txt").read_text(encoding="utf-8").strip() == sid
            assert (d / "result.json").exists() and (d / p.model_file(m).name).exists()
            assert f"{MODEL_NAMES[m]}: trained on another training stack (stack_id.txt differs); previous run kept " \
                   f"as models/{d.name}, training again" in log
            assert (p.model_dir(m) / "stack_id.txt").read_text(encoding="utf-8").strip() == sid2
            assert (p.model_dir(m) / "result.json").exists() and p.model_file(m).exists()
        res = json.load(open(p.model_dir("kg-unet2") / "result.json"))
        assert Path(res["teacher_sls"]) == p.model_file("unet-sls")

        # nothing changed: every model is kept
        mt = _mtimes(p)
        workflow.train(p)
        assert p.status()["train"]["message"] == "kept RF-SLS, UNet-SLS, KG-UNet1, KG-UNet2"
        assert _mtimes(p) == mt and all(len(_stale(p, m)) == 1 for m in MODELS)

        # UNet-SLS removed: trained again; KG-UNet2 (GEDI teacher UNet-SLS) is retired and trained again,
        # RF-SLS and KG-UNet1 are kept
        shutil.rmtree(p.model_dir("unet-sls"))
        workflow.train(p)
        assert p.status()["train"]["message"] == "trained UNet-SLS, KG-UNet2, kept RF-SLS, KG-UNet1"
        assert "KG-UNet2: its GEDI teacher UNet-SLS was retrained; previous run kept as models/kg-unet2.stale-" \
            in tw._log(p)
        assert [len(_stale(p, m)) for m in MODELS] == [1, 1, 1, 2]
        now = _mtimes(p)
        assert all(now[m] == mt[m] for m in ("rf-sls", "kg-unet1"))
        assert all(now[m] > mt[m] for m in ("unet-sls", "kg-unet2"))
        for m in MODELS:
            assert (p.model_dir(m) / "stack_id.txt").read_text(encoding="utf-8").strip() == sid2
    finally:
        tw._release(root)


def _fake_trainers(monkeypatch, calls):
    """Instant stand-ins for rf.train_rf and train.train (files of a finished run, the input recorded) and a base
    checkpoint check that accepts anything: the orchestration of the train stage without training."""
    import canopy_height.rf as rf_mod
    import canopy_height.train as train_mod

    def fake_rf(out, annual, labels, seasonal=None, input_name="AE", log=print, **kw):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "rf_sls.joblib").write_bytes(b"fake")
        meta = dict(model="rf-sls", input=input_name, n_fit=1, holdout_rmse_m=1.0)
        json.dump(meta, open(out / "result.json", "w"))
        calls.append("rf-sls")
        return None, meta

    def fake_unet(recipe, out, annual, labels, seasonal=None, base=None, teacher_sls=None, device=None, log=print,
                  **over):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        # different bytes per run: a retrained model differs (a bit-identical retrain would keep its dependants)
        (out / "model.pth").write_bytes(f"fake {recipe} {len(calls)} {time.time_ns()}".encode())
        res = dict(model=dict(input=over.get("input", "AE")), epochs_run=1, final_val_loss=0.1,
                   teacher_sls=str(teacher_sls) if teacher_sls else None)
        json.dump(res, open(out / "result.json", "w"))
        calls.append(recipe)
        return out / "model.pth", res
    monkeypatch.setattr(rf_mod, "train_rf", fake_rf)
    monkeypatch.setattr(train_mod, "train", fake_unet)
    monkeypatch.setattr(workflow, "_base_for", lambda project: Path("UNet-ALS.pth"))


def test_teacher_retrained_in_an_earlier_run_retires_kg_unet2(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    p = tw._project(tmp_path / "proj")
    try:
        tw._write_stacks_and_mosaics(p)
        calls = []
        _fake_trainers(monkeypatch, calls)
        workflow.train(p)
        assert calls == MODELS
        shutil.rmtree(p.model_dir("unet-sls"))
        workflow.train(p, models=["unet-sls"])            # the teacher of KG-UNet2 is trained again ...
        calls.clear()
        workflow.train(p)                                  # ... so KG-UNet2 no longer fits it
        assert "kg-unet2" in calls, p.status()["train"]["message"]
    finally:
        tw._release(p.root)


def test_run_without_model_file_is_trained_again(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    p = tw._project(tmp_path / "proj")
    try:
        tw._write_stacks_and_mosaics(p)
        calls = []
        _fake_trainers(monkeypatch, calls)
        workflow.train(p)
        p.model_file("rf-sls").unlink()
        calls.clear()
        trained = workflow.train(p)
        assert "rf-sls" in calls and "rf-sls" in trained, p.status()["train"]["message"]
    finally:
        tw._release(p.root)


# ---------------------------------------------------------------------------------------------- 3 input T
def test_input_T_end_to_end(tmp_path, weights):
    """Input T (DEM + seasonal S1 / S2, no embedding): stack -> train (a UNet-ALS checkpoint of another layout is
    refused) -> predict -> report on the CPU."""
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    root = tmp_path / "proj"
    p = tw._project(root, input="T", train_overrides=FAST)
    try:
        assert p.seasonal
        _write_raw(p)
        assert not list(p.path("raw").glob("*_Embedding.tif")) and not list(p.path("raw").glob("*_S1.tif"))
        workflow.stack(p)
        assert p.status()["stack"]["state"] == "done"
        f = p.path("stacks", "train_seasonal_part001.npy")
        assert f.exists() and p.stack_files("seasonal") == [str(f)]
        S = np.load(f, mmap_mode="r")
        X = np.load(p.stack_files("x")[0], mmap_mode="r")
        assert S.shape == (6, 44, 256, 256) and S.dtype == np.uint16 and np.asarray(S).all()
        assert X.shape == (6, 76, 256, 256) and not np.asarray(X[:, :64]).any() and not np.asarray(X[:, 65:]).any()
        with rasterio.open(p.mosaic("seasonal")) as s:
            assert s.count == 44 and s.tags().get("input") == "T" and (s.width, s.height) == (2 * tw.PX, 2 * tw.PX)

        # the UNet-ALS checkpoint of T holds a network of another input layout: refused with a clear message
        _fake_base(weights, "T", layout="AE")
        with pytest.raises(ValueError, match="UNet-T-ALS.pth has input AE, the project uses T"):
            workflow.train(p, models=["kg-unet1"])
        assert p.status()["train"]["state"] == "failed" and not p.model_dir("kg-unet1").exists()

        _fake_base(weights, "T")
        st = workflow.run(p, "train", "report")
        assert all(st[s]["state"] == "done" for s in ("train", "predict", "report"))
        for m in MODELS:
            res = json.load(open(p.model_dir(m) / "result.json"))
            assert (res["input"] if m == "rf-sls" else res["model"]["input"]) == "T"
            if m != "rf-sls":
                assert [Path(x).name for x in res["seasonal"]] == ["train_seasonal_part001.npy"]
        rows, _ = tw._check_maps_and_report(p)               # maps on the mosaic grid, report with every section
        assert {("gedi_val", MODEL_NAMES[m]) for m in MODELS} <= {(r["section"], r["layer"]) for r in rows}
    finally:
        tw._release(root)


# ---------------------------------------------------------------------------------------------- 4 run lock
def test_run_lock_of_another_live_process(tmp_path):
    """logs/run.lock of a live process (pid and start time) refuses every stage; a reused pid (other start time)
    and a lock of a process that ended are ignored."""
    psutil = pytest.importorskip("psutil")
    p = tw._project(tmp_path / "proj")
    lock = p.path("logs", "run.lock", mkdir=True)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        rec = dict(pid=child.pid, create_time=psutil.Process(child.pid).create_time(), started="2026-09-28 10:00:00",
                   command="run plan..report")
        lock.write_text(json.dumps(rec), encoding="utf-8")
        assert lock_owner(p) == rec
        with pytest.raises(RuntimeError, match=f"is being run by process {child.pid} "):
            workflow.plan(p)
        with pytest.raises(RuntimeError, match=f"process {child.pid} "):
            workflow.run(p, "plan", "plan")
        assert p.status()["plan"]["state"] == "pending"                      # refused before the stage started
        assert json.loads(lock.read_text(encoding="utf-8")) == rec          # the owner's lock is left alone

        # same pid, another start time: the pid was reused, the lock is stale
        lock.write_text(json.dumps(dict(rec, create_time=rec["create_time"] - 100)), encoding="utf-8")
        assert lock_owner(p) is None
        workflow.plan(p)
        assert p.status()["plan"]["state"] == "done" and not lock.exists()
        lock.write_text(json.dumps(rec), encoding="utf-8")
        with pytest.raises(RuntimeError, match=str(child.pid)):
            workflow.plan(p)
    finally:
        child.terminate()                                                    # our own child, by its handle / pid
        child.wait(timeout=30)
        tw._release(p.root)

    # the owner has ended: the lock is ignored and the stage runs (and removes it)
    assert lock.exists() and lock_owner(p) is None
    try:
        rows = workflow.plan(p)
        assert len(rows) == 6 and p.status()["plan"]["state"] == "done" and not lock.exists()
    finally:
        tw._release(p.root)


# ---------------------------------------------------------------------------------------------- 5 GEDI labels
def test_too_few_gedi_labels(tmp_path, monkeypatch):
    """GEDI rasters with 150 labels per cell (6 cells: 900 < MIN_LABELS): the train stage refuses before any model;
    zeros and the -9999 export sentinel are not labels; at the threshold the stage trains."""
    pytest.importorskip("sklearn")
    assert workflow.MIN_LABELS > 900

    def sparse(i, h):
        g = np.random.default_rng(50 + i)
        idx = g.choice(h.size, 180, replace=False)
        lab = np.full(h.size, np.nan)
        lab[idx[:150]] = h.ravel()[idx[:150]]                 # rh95 > 0: labels
        lab[idx[150:170]] = 0.0                               # 0 m: not a label
        lab[idx[170:]] = -9999                                # export sentinel: no label
        return lab.reshape(h.shape)

    root = tmp_path / "proj"
    p = tw._project(root, input="A", models=["rf-sls"], train_overrides=FAST, benchmarks=[])
    try:
        _write_raw(p, gedi=sparse)
        workflow.stack(p)
        with pytest.raises(ValueError, match=rf"only 900 GEDI label pixels in the training region \(at least "
                                             rf"{workflow.MIN_LABELS} needed\)"):
            workflow.train(p)
        st = p.status()["train"]
        assert st["state"] == "failed" and "GEDI label pixels" in st["message"]
        assert not p.path("models").exists()

        monkeypatch.setattr(workflow, "MIN_LABELS", 900)
        assert workflow.train(p) == ["rf-sls"]
        assert "train: 6 chips, 900 GEDI label pixels, input A" in tw._log(p)
        assert json.load(open(p.model_dir("rf-sls") / "result.json"))["n_rows"] == 900
    finally:
        tw._release(root)
