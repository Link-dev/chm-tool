"""Canopy-height tool report: ALS chip rule, GEDI 80 m cap, benchmark notes, stale models and maps, windowed
statistics and quick-look clean-up, on small synthetic mosaics (no PyTorch needed except where marked)."""
import csv
import gc
import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from canopy_height.metrics import evaluate_chips  # noqa: E402
from canopy_height.tool import Project  # noqa: E402
from canopy_height.tool import report as rp  # noqa: E402
from canopy_height.tool import stack as stack_mod  # noqa: E402

EPSG, X0, Y1, PX = 32650, 584670.0, 550110.0, 256


def _proj(root, **cfg):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "project.yaml").write_text(yaml.safe_dump(dict(dict(name="rep", device="cpu"), **cfg)),
                                       encoding="utf-8")
    return Project(root)


def _write(path, a, dtype="float32", nodata=np.nan):
    a = np.asarray(a)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=a.shape[1], height=a.shape[0], count=1, dtype=dtype,
                       nodata=nodata, crs=f"EPSG:{EPSG}", transform=from_origin(X0, Y1, 10, 10), tiled=True,
                       blockxsize=256, blockysize=256) as d:
        d.write(a.astype(dtype), 1)


def _release(p):
    p.close_log()


def _csv(p):
    with open(p.path("report", "metrics.csv"), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture
def stale(monkeypatch):
    """stack.model_is_current stubbed: models listed in the returned dict are stale (value = reason)."""
    bad = {}
    monkeypatch.setattr(stack_mod, "model_is_current", lambda project, m: (m not in bad, bad.get(m, "")),
                        raising=False)
    return bad


# ---------------------------------------------------------------------------------------------- ALS chips
def _als_project(tmp_path):
    """2 x 4 chips: 3 kept (two full, one with 656 ALS px > 1 m), 4 dropped (3 px, 655 px, 700 px of which 49
    outside the study area, one 95 m cell), one with ALS <= 1 m only."""
    p = _proj(tmp_path / "als", benchmarks=["HRCH", "GFCH"])
    g = np.random.default_rng(0)
    H, W = 2 * PX, 4 * PX
    inside = np.ones((H, W), np.uint8)
    als = np.full((H, W), np.nan, np.float32)
    als[:PX, :2 * PX] = g.uniform(2, 40, (PX, 2 * PX))                   # chips (0, 0) and (0, 256): full
    als[5, 520:523] = 20.0                                               # chip (0, 512): 3 px
    als[PX:PX + 5, :131] = 15.0                                          # chip (256, 0): 655 px
    als[PX + 10:PX + 14, 256:420] = 12.0                                 # chip (256, 256): 656 px
    als[PX:PX + 7, 768:868] = 25.0                                       # chip (256, 768): 700 px ...
    inside[PX:PX + 7, 768:768 + 45 // 7 + 1] = 0                         # ... 49 of them outside
    als[PX + 100, 512:600] = 0.5                                         # chip (256, 512): ALS <= 1 m only
    als[0, 780] = 95.0                                                   # chip (0, 768): one cell > 80 m
    hrch = np.nan_to_num(als, nan=10.0) + g.normal(0, 2, (H, W)).astype(np.float32)
    hrch[:20, :20] = np.nan                                              # missing predictions count as 0
    gfch = 0.8 * np.nan_to_num(als, nan=5.0)
    _write(p.mosaic("aoi_mask"), inside, "uint8", None)
    _write(p.mosaic("ALS"), als)
    _write(p.mosaic("HRCH"), hrch)
    _write(p.mosaic("GFCH"), gfch)
    return p, inside == 1, als, dict(HRCH=hrch, GFCH=gfch)


def test_als_chip_rule_same_chips_every_layer(tmp_path, stale):
    p, inside, als, layers = _als_project(tmp_path)
    try:
        s = rp.report(p)
    finally:
        _release(p)
    al = s["als"]
    assert (al["chips_kept"], al["chips_dropped"]) == (3, 4)            # 3 + 655 + (700 - 49) px + the 95 m cell
    kept = [(0, 0), (0, 256), (256, 256)]
    a = np.where(inside, als, np.nan)
    ref = [np.where(np.isfinite(c), c, -999.0).astype(np.float32) for c in (a[r:r + PX, c:c + PX] for r, c in kept)]
    for name, lay in layers.items():
        m = np.where(inside, lay, np.nan)
        ev = evaluate_chips([m[r:r + PX, c:c + PX] for r, c in kept], ref)
        got = al["layers"][name]
        assert got["chip_median"]["n_chips"] == 3
        for k in ("rmse_median", "r2_median", "me_median"):
            assert got["chip_median"][k] == pytest.approx(ev["chip_median"][k], rel=1e-12)
        for k in ("n_px", "me", "rmse", "r2"):
            assert got["pooled"][k] == pytest.approx(ev["pooled"][k], rel=1e-12)
    rows = {(r["section"], r["layer"]): r for r in _csv(p)}
    assert "3 chips" in rows[("als", "HRCH")]["note"] and "4 with less left out" in rows[("als", "HRCH")]["note"]
    assert any("4 chips with ALS > 1 m in fewer than 1% of their cells" in n for n in s["notes"])


# ---------------------------------------------------------------------------------------------- GEDI, notes
def _gedi_project(tmp_path, name="gedi"):
    p = _proj(tmp_path / name, benchmarks=["HRCH", "GFCH", "GMTCH"])
    g = np.random.default_rng(1)
    H, W = 2 * PX, 4 * PX
    yy, xx = np.mgrid[:H, :W]
    inside = ((yy - H / 2) ** 2 / (H / 2) ** 2 + (xx - W / 2) ** 2 / (W / 2) ** 2 < 0.9).astype(np.uint8)
    h = (18 + 10 * np.sin(xx / 40.0) * np.cos(yy / 55.0)).astype(np.float32)
    gedi = np.full((H, W), np.nan, np.float32)
    m = g.random((H, W)) < 0.03
    gedi[m] = h[m] + g.normal(0, 3, m.sum())
    big = g.random((H, W)) < 0.002
    gedi[big] = g.uniform(81, 140, big.sum())                            # cloud-contaminated shots
    gedi[g.random((H, W)) < 0.001] = 0.0
    lay = {k: v.astype(np.float32) for k, v in dict(HRCH=h + g.normal(0, 2, h.shape), GFCH=0.8 * h,
                                                          GMTCH=h + 1.0).items()}
    lay["HRCH"][:, :30] = np.nan
    _write(p.mosaic("aoi_mask"), inside, "uint8", None)
    _write(p.mosaic("GEDI"), gedi)
    for k, v in lay.items():
        _write(p.mosaic(k), v)
    return p, inside == 1, gedi, lay


def test_gedi_cap_benchmark_notes_and_exact_statistics(tmp_path, stale):
    p, inside, gedi, lay = _gedi_project(tmp_path)
    try:
        s = rp.report(p)
    finally:
        _release(p)
    lab = inside & (gedi > 0) & (gedi <= 80)
    over = int((inside & (gedi > 80)).sum())
    assert over > 0
    sa = s["study_area"]
    assert sa["percentiles"] == "every pixel" and sa["px"] == int(inside.sum())
    assert sa["layers"]["GEDI"]["n_px"] == int(lab.sum()) and sa["layers"]["GEDI"]["n_labels_over_80"] == over
    assert sa["layers"]["GEDI"]["p90"] == pytest.approx(float(np.percentile(gedi[lab].astype(np.float64), 90)))
    for name, a in lay.items():
        got = sa["layers"][name]
        v = a[inside & np.isfinite(a)].astype(np.float64)
        assert got["n_px"] == v.size and got["mean"] == pytest.approx(v.mean(), rel=1e-9)
        assert [got[k] for k in ("p10", "p50", "p90")] == list(np.percentile(v, [10, 50, 90]))
        sel = lab & np.isfinite(a)
        pp, t = a[sel].astype(np.float64), gedi[sel].astype(np.float64)
        assert got["gedi"]["n"] == int(sel.sum())
        assert got["gedi"]["rmse"] == pytest.approx(np.sqrt(np.mean((pp - t) ** 2)), rel=1e-9)
        assert got["gedi"]["bias"] == pytest.approx(np.mean(pp - t), rel=1e-9, abs=1e-12)
        assert got["gedi"]["r2"] == pytest.approx(np.corrcoef(pp, t)[0, 1] ** 2, rel=1e-9)
    rows = {(r["section"], r["layer"]): r["note"] for r in _csv(p)}
    assert rp.NOTE_BENCH_GEDI in rows[("study_area", "HRCH")] and rp.NOTE_BENCH_GEDI in rows[("study_area", "GFCH")]
    assert rp.NOTE_BENCH_GEDI not in rows[("study_area", "GMTCH")]
    assert "0 < rh95 <= 80 m" in rows[("study_area", "GEDI")] and f"{over:,} labels > 80 m left out" in \
        rows[("study_area", "GEDI")]
    assert s["als"] is None and set(s["quicklooks"]) == {"HRCH.png", "GFCH.png", "GMTCH.png"}


def test_percentiles_from_lattice_subsample(tmp_path, stale, monkeypatch):
    monkeypatch.setattr(rp, "SAMPLE_PX", 10_000)
    p, inside, gedi, lay = _gedi_project(tmp_path, "lattice")
    try:
        s = rp.report(p)
    finally:
        _release(p)
    k = int(np.ceil(np.sqrt(inside.sum() / 10_000)))
    assert k > 1 and str(k) in s["study_area"]["percentiles"]
    for name, a in lay.items():
        got = s["study_area"]["layers"][name]
        v = a[inside & np.isfinite(a)].astype(np.float64)
        sub = a[::k, ::k][inside[::k, ::k] & np.isfinite(a[::k, ::k])].astype(np.float64)
        assert got["n_px"] == v.size and got["mean"] == pytest.approx(v.mean(), rel=1e-9)
        assert [got[q] for q in ("p10", "p50", "p90")] == list(np.percentile(sub, [10, 50, 90]))
        assert got["gedi"]["n"] == int((inside & (gedi > 0) & (gedi <= 80) & np.isfinite(a)).sum())


def test_old_quicklooks_removed(tmp_path, stale):
    p, *_ = _gedi_project(tmp_path, "ql")
    try:
        rp.report(p)
        assert all(p.path("report", f"{b}.png").exists() for b in ("HRCH", "GFCH", "GMTCH"))
        p.path("report", "my_notes.png").write_bytes(b"user file")
        p.path("report", "GFCH.png.tmp").write_bytes(b"partial")
        p.cfg["benchmarks"] = ["HRCH"]
        s = rp.report(p)
    finally:
        _release(p)
    assert s["quicklooks"] == ["HRCH.png"]
    left = sorted(f.name for f in p.path("report").iterdir() if ".png" in f.name)
    assert left == ["HRCH.png", "my_notes.png"]


def test_pairs_blocks_equal_one_pass():
    g = np.random.default_rng(2)
    p, t = g.normal(20, 5, 10_001), g.normal(20, 5, 10_001)
    p += 0.7 * t
    acc = rp.Pairs()
    for a, b in zip(np.array_split(p, 7), np.array_split(t, 7)):
        acc.add(a, b)
    r = acc.result()
    d = p - t
    assert r["n"] == p.size and r["rmse"] == pytest.approx(np.sqrt(np.mean(d * d)), rel=1e-12)
    assert r["bias"] == pytest.approx(d.mean(), rel=1e-10) and r["r2"] == pytest.approx(np.corrcoef(p, t)[0, 1] ** 2,
                                                                                        rel=1e-10)
    assert np.isnan(rp.agreement(np.full(50, 0.1), t[:50])["r2"]) and rp.agreement([], [])["n"] == 0


# ---------------------------------------------------------------------------------------------- models
def _model_project(tmp_path, name, n=10, splits=None, **cfg):
    """Training stack of n small label chips (with a seasonal part for the inputs T / TE), four 'trained' models,
    maps and a study-area mask."""
    p = _proj(tmp_path / name, **cfg)
    g = np.random.default_rng(3)
    Y = np.full((n, 16, 16), -999, np.float32)
    m = g.random(Y.shape) < 0.4
    Y[m] = g.uniform(1, 100, m.sum())                                   # some labels > 80 m
    p.path("stacks").mkdir(parents=True)
    np.save(p.stack_part("x", 1), np.zeros((n, 1, 16, 16), np.uint16))
    np.save(p.stack_part("GEDI", 1), Y)
    if p.seasonal:
        np.save(p.stack_part("seasonal", 1), np.ones((n, 2, 16, 16), np.uint16))
    for mdl in ("rf-sls", "unet-sls", "kg-unet1", "kg-unet2"):
        d = p.model_dir(mdl)
        d.mkdir(parents=True)
        (d / "result.json").write_text("{}")
        p.model_file(mdl).write_bytes(b"model")
    for mdl, (tr, va) in (splits or {}).items():
        np.savez(p.model_dir(mdl) / "split.npz", train=np.array(tr), val=np.array(va))
    inside = np.ones((PX, PX), np.uint8)
    _write(p.mosaic("aoi_mask"), inside, "uint8", None)
    for mdl in ("unet-sls", "kg-unet1", "kg-unet2"):
        _write(p.map_file(mdl), np.full((PX, PX), 20.0, np.float32))
    t = p.model_file("unet-sls").stat().st_mtime
    os.utime(p.model_file("kg-unet1"), (t + 100, t + 100))            # retrained after its map was made
    for mdl in ("unet-sls", "kg-unet2"):
        os.utime(p.map_file(mdl), (t + 200, t + 200))
    os.utime(p.map_file("kg-unet1"), (t + 50, t + 50))
    return p, Y


OFFSET = {"rf-sls": 1.0, "unet-sls": 2.0, "kg-unet1": -1.0, "kg-unet2": 5.0}


class _Calls(list):
    """Models predicted, in order; .seasonal: model -> the seasonal stack it was given."""

    def __init__(self):
        super().__init__()
        self.seasonal = {}


@pytest.fixture
def fake_predict(monkeypatch):
    calls = _Calls()

    def fake(project, model, stack, idx, chunk=16, seasonal=None):
        calls.append(model)
        calls.seasonal[model] = seasonal
        Y = np.load(project.stack_part("GEDI", 1))
        yield np.stack([Y[int(i)] + OFFSET[model] for i in idx])
    monkeypatch.setattr(rp, "_predict_chips", fake)
    return calls


def test_gedi_val_stale_models_in_sample_share_and_stale_maps(tmp_path, stale, fake_predict):
    others = [0, 2, 3, 4, 5, 6, 7, 9]
    p, Y = _model_project(tmp_path, "models", splits={"unet-sls": (others, [1, 8]),
                                                      "kg-unet1": ([2, 3, 4, 5, 6, 7, 8, 9], [0, 1]),
                                                      "kg-unet2": (others, [1, 8])})
    stale["kg-unet2"] = "stack_id.txt differs"
    try:
        s = rp.report(p)
    finally:
        _release(p)
    gv = s["gedi_validation"]
    assert gv["split"] == "models/unet-sls/split.npz" and gv["n_chips"] == 2
    lab = [Y[i][(Y[i] > 0) & (Y[i] <= 80)] for i in (1, 8)]
    assert gv["n_labels"] == sum(x.size for x in lab)
    assert gv["n_labels_over_80"] == int(sum((Y[i] > 80).sum() for i in (1, 8))) > 0
    assert "kg-unet2" not in fake_predict and gv["skipped"] == {"KG-UNet2": "stack_id.txt differs"}
    assert s["stale_models"] == {"KG-UNet2": "stack_id.txt differs"}
    ms = gv["models"]
    assert set(ms) == {"RF-SLS", "UNet-SLS", "KG-UNet1"}
    for name, mdl in (("RF-SLS", "rf-sls"), ("UNet-SLS", "unet-sls"), ("KG-UNet1", "kg-unet1")):
        assert ms[name]["bias"] == pytest.approx(OFFSET[mdl]) and ms[name]["rmse"] == pytest.approx(abs(OFFSET[mdl]))
    assert ms["UNet-SLS"]["note"].startswith("held-out chips") and ms["UNet-SLS"]["n_own_training_chips"] == 0
    assert ms["KG-UNet1"]["note"].startswith("partly in-sample: 1 of 2 chips")
    assert ms["RF-SLS"]["note"].startswith("in-sample")
    assert all("GEDI labels <= 80 m" in m["note"] for m in ms.values())
    assert fake_predict.seasonal == dict.fromkeys(["rf-sls", "unet-sls", "kg-unet1"])     # AE: no seasonal stack
    assert s["input"] == "AE"
    assert any("KG-UNet2 not scored" in n and "retrain" in n for n in s["notes"])
    # maps: UNet-SLS current; KG-UNet1 older than its model (left out); KG-UNet2 scored but flagged; no RF map
    layers = s["study_area"]["layers"]
    assert "UNet-SLS" in layers and "KG-UNet1" not in layers and "RF-SLS" not in layers
    assert "does not match the current training stack (stack_id.txt differs)" in layers["KG-UNet2"]["note"]
    assert any("KG-UNet1.tif is older than its model" in n for n in s["notes"])
    assert any("RF-SLS: no map" in n for n in s["notes"])
    assert sorted(s["quicklooks"]) == ["KG-UNet2.png", "UNet-SLS.png"]


def test_gedi_val_never_uses_a_split_that_does_not_fit(tmp_path, stale, fake_predict):
    """A UNet-SLS split.npz beyond the current stack marks the model stale; the other models are scored on the
    recipe's split of the current stack, with their in-sample share."""
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    from canopy_height.train import resolve, split_chips
    val = np.sort(split_chips(10, resolve("unet-sls"))[1])
    p, _ = _model_project(tmp_path, "nofit", splits={"unet-sls": (list(range(2, 20)), [0, 1, 25]),
                                                     "kg-unet1": ([int(val[0])], [int(val[1])])})
    try:
        s = rp.report(p)
    finally:
        _release(p)
    gv = s["gedi_validation"]
    assert gv["split"] == "split of the unet-sls recipe" and gv["n_chips"] == len(val)
    assert "UNet-SLS" in gv["skipped"] and "unet-sls" not in fake_predict
    assert gv["models"]["KG-UNet1"]["note"].startswith(f"partly in-sample: 1 of {len(val)} chips")
    assert gv["models"]["KG-UNet2"]["note"].startswith("held-out status unknown")


def test_report_without_stack_does_not_flag_models(tmp_path, monkeypatch):
    def boom(project, m):
        raise AssertionError("model_is_current must not be called without a training stack")
    monkeypatch.setattr(stack_mod, "model_is_current", boom, raising=False)
    p = _proj(tmp_path / "nostack")
    p.model_dir("unet-sls").mkdir(parents=True)
    (p.model_dir("unet-sls") / "result.json").write_text("{}")
    p.model_file("unet-sls").write_bytes(b"model")
    try:
        s = rp.report(p)
    finally:
        _release(p)
    assert s["models"] == ["UNet-SLS"] and s["stale_models"] == {} and s["gedi_validation"] is None
    assert s["n_rows"] == 0 and any("no training stack" in n for n in s["notes"])
    assert json.loads(p.path("report", "summary.json").read_text(encoding="utf-8"))["quicklooks"] == []


# ---------------------------------------------------------------------------------------------- inputs T / TE
def test_gedi_val_gives_the_seasonal_stack(tmp_path, stale, fake_predict):
    p, _ = _model_project(tmp_path, "tmodels", splits={"unet-sls": ([0, 2, 3, 4, 5, 6, 7, 9], [1, 8])}, input="T")
    try:
        s = rp.report(p)
    finally:
        _release(p)
    assert s["input"] == "T" and set(s["gedi_validation"]["models"]) == {"RF-SLS", "UNet-SLS", "KG-UNet1",
                                                                         "KG-UNet2"}
    for m, st in fake_predict.seasonal.items():
        assert st is not None and st.files == p.stack_files("seasonal") and len(st) == 10, m
    # a seasonal stack of another chip count is not used
    del st
    fake_predict.seasonal.clear()                           # releases the memory maps (Windows locks mapped files)
    gc.collect()
    np.save(p.stack_part("seasonal", 1), np.ones((9, 2, 16, 16), np.uint16))
    try:
        s = rp.report(p)
    finally:
        _release(p)
    assert all(v is None for v in fake_predict.seasonal.values()) and fake_predict.seasonal
    assert any("seasonal training stack has 9 chips, the training stack 10: not used" in n for n in s["notes"])


def test_seasonal_models_scored_with_the_seasonal_stack(tmp_path, stale):
    """Real UNet (T and AE) and RF-SLS (T) models: _predict_chips gives T models the seasonal chips (same result as
    predict_stack / predict_rf_chips on the annual and seasonal chips), AE models only the annual ones, and a T
    model without the seasonal stack is skipped with a note."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from canopy_height import channels, stats
    from canopy_height.data import Stack
    from canopy_height.models import build_model, load_model, model_config, save_checkpoint
    from canopy_height.predict import predict_stack
    from canopy_height.rf import predict_rf_chips
    from canopy_height.train import resolve, split_chips

    p = _proj(tmp_path / "real", input="T")
    g = np.random.default_rng(4)
    n = 6
    A = g.integers(0, 30000, (n, 76, 16, 16)).astype(np.uint16)
    Sa = g.integers(0, 20000, (n, 44, 16, 16)).astype(np.uint16)
    Y = np.where(g.random((n, 16, 16)) < 0.5, g.uniform(1, 60, (n, 16, 16)), -999).astype(np.float32)
    p.path("stacks").mkdir(parents=True)
    for k, a in (("x", A), ("seasonal", Sa), ("GEDI", Y)):
        np.save(p.stack_part(k, 1), a)
    As, Ss = Stack(p.stack_files("x")), Stack(p.stack_files("seasonal"))
    torch.manual_seed(0)
    for m, inp in (("unet-sls", "T"), ("kg-unet1", "AE")):
        cfg = model_config(inp)
        p.model_dir(m).mkdir(parents=True)
        save_checkpoint(p.model_file(m), build_model(cfg, *stats.for_input(inp)), cfg)
        (p.model_dir(m) / "result.json").write_text("{}")
    feats = np.transpose(channels.assemble("T", A, Sa), (0, 2, 3, 1)).reshape(-1, 45).astype(np.float32)
    rf = RandomForestRegressor(n_estimators=4, max_depth=6, random_state=0).fit(feats, np.tile([1.0, 30.0], len(feats) // 2))
    rf.chm_meta = dict(model="rf-sls", input="T", encoding="paper")
    p.model_dir("rf-sls").mkdir(parents=True)
    joblib.dump(rf, p.model_file("rf-sls"))
    (p.model_dir("rf-sls") / "result.json").write_text("{}")

    idx = np.array([1, 4, 5])
    cpu = torch.device("cpu")
    mt, ct = load_model(p.model_file("unet-sls"), device=cpu)
    exp_t = predict_stack(mt, ct, A[idx], Sa[idx], device=cpu)
    assert not np.array_equal(exp_t, predict_stack(mt, ct, A[idx], np.zeros_like(Sa[idx]), device=cpu))
    got = np.concatenate(list(rp._predict_chips(p, "unet-sls", As, idx, seasonal=Ss)))
    assert np.array_equal(got, exp_t)
    mi, ci = load_model(p.model_file("kg-unet1"), device=cpu)
    got = np.concatenate(list(rp._predict_chips(p, "kg-unet1", As, idx, seasonal=Ss)))
    assert np.array_equal(got, predict_stack(mi, ci, A[idx], device=cpu))
    got = np.concatenate(list(rp._predict_chips(p, "rf-sls", As, idx, seasonal=Ss)))
    assert np.array_equal(got, predict_rf_chips(rf, rf.chm_meta, A[idx], Sa[idx]))
    for m in ("unet-sls", "rf-sls"):
        with pytest.raises(rp.SeasonalStackMissing, match="input T needs the seasonal training stack"):
            list(rp._predict_chips(p, m, As, idx))

    # the report: the T models scored on the recipe's validation chips with their seasonal chips
    try:
        s = rp.report(p)
    finally:
        _release(p)
    gv = s["gedi_validation"]
    val = np.sort(split_chips(n, resolve("unet-sls"))[1])
    assert gv["split"] == "split of the unet-sls recipe" and gv["n_chips"] == len(val)
    assert set(gv["models"]) == {"RF-SLS", "UNet-SLS", "KG-UNet1"}
    yv = Y[val]
    mk = (yv > 0) & (yv <= 80)
    for name, pred in (("UNet-SLS", predict_stack(mt, ct, A[val], Sa[val], device=cpu)),
                       ("RF-SLS", predict_rf_chips(rf, rf.chm_meta, A[val], Sa[val])),
                       ("KG-UNet1", predict_stack(mi, ci, A[val], device=cpu))):
        a = rp.agreement(pred[mk], yv[mk])
        assert gv["models"][name]["n"] == a["n"] == int(mk.sum())
        assert gv["models"][name]["rmse"] == pytest.approx(a["rmse"], rel=1e-9)

    # without the seasonal stack: the T models are skipped with a note, the AE model is still scored
    del As, Ss
    gc.collect()                                            # releases the memory maps (Windows locks mapped files)
    for f in p.stack_files("seasonal"):
        os.remove(f)
    try:
        s = rp.report(p)
    finally:
        _release(p)
    assert set(s["gedi_validation"]["models"]) == {"KG-UNet1"}
    for name in ("RF-SLS", "UNet-SLS"):
        assert any(f"gedi_val: {name} skipped (input T needs the seasonal training stack" in x for x in s["notes"])
