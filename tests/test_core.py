import numpy as np
import pytest
import torch

from canopy_height import channels, labels, models, predict

def _rand_input(c, h=32, w=32, seed=0):
    g = np.random.default_rng(seed)
    x = g.integers(1, 20000, size=(2, c, h, w)).astype(np.float32)
    x[:, :, :3, :3] = 0                      # no-data corner
    x[0, 5, 10, 10] = 0                      # isolated zero
    return x


def test_input_layouts():
    assert [channels.INPUTS[k]["n_channels"] for k in ("AE", "A", "E", "T", "TE")] == [76, 12, 64, 45, 109]
    a = np.arange(2 * 76 * 4 * 4).reshape(2, 76, 4, 4)
    s = -np.arange(2 * 44 * 4 * 4).reshape(2, 44, 4, 4)
    te = channels.assemble("TE", a, s)
    assert te.shape == (2, 109, 4, 4) and (te[:, :65] == a[:, :65]).all() and (te[:, 65:] == s).all()
    t = channels.assemble("T", a, s)
    assert (t[:, 0] == a[:, 64]).all() and (t[:, 1:] == s).all()


def test_standardise_only_equals_external_zscore():
    mean, std = np.linspace(100, 900, 12).astype(np.float32), np.linspace(10, 90, 12).astype(np.float32)
    torch.manual_seed(0); m = models.NormalizingUNet(12, 1, mean, std, n_embed=0, zero_restore=False).eval()
    x = _rand_input(12)
    z = (x - mean[None, :, None, None]) / (std[None, :, None, None] + np.float32(1e-12))
    assert np.array_equal(m.normalize(torch.tensor(x)).numpy(), z)


def test_tiled_prediction():
    torch.manual_seed(0)
    cfg = models.model_config("AE")
    m = models.build_model(cfg, np.full(76, 500.0), np.full(76, 100.0)).eval()
    g = np.random.default_rng(1)
    img = g.integers(1, 20000, size=(76, 64, 64)).astype(np.uint16)
    one = predict.predict_batches(m, img[None].astype(np.float32), device="cpu")[0]
    assert np.array_equal(predict.predict_array(m, cfg, img, tile=64, overlap=0, device="cpu"), one)
    big = g.integers(1, 20000, size=(76, 100, 90)).astype(np.uint16)
    p = predict.predict_array(m, cfg, big, tile=64, overlap=16, device="cpu")
    assert p.shape == (100, 90) and np.isfinite(p).all()
    big[:, :5, :5] = 0
    assert np.isnan(predict.predict_array(m, cfg, big, tile=64, overlap=16, device="cpu")[:5, :5]).all()


def test_block_percentile():
    g = np.random.default_rng(2)
    a = g.random((30, 40)).astype(np.float32) * 50
    a[:7, :3] = np.nan; a[10:20, 10:20] = np.nan
    out = labels.block_percentile(a, 10, 90)
    for i in range(3):
        for j in range(4):
            blk = a[10 * i:10 * i + 10, 10 * j:10 * j + 10]
            ref = np.float32(np.nanpercentile(blk.astype(np.float64), 90)) if np.isfinite(blk).any() else np.nan
            assert (np.isnan(ref) and np.isnan(out[i, j])) or out[i, j] == ref


def test_tile_roundtrip(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    g = np.random.default_rng(3)
    ann = g.integers(1, 20000, size=(76, 300, 260)).astype(np.uint16)
    lab = g.random((300, 260)).astype(np.float32) * 30
    lab[250:, :] = np.nan
    prof = dict(driver="GTiff", width=260, height=300, crs="EPSG:32615", transform=from_origin(500000, 4000000, 10, 10))
    with rasterio.open(tmp_path / "a.tif", "w", count=76, dtype="uint16", **prof) as d:
        d.write(ann)
    with rasterio.open(tmp_path / "y.tif", "w", count=1, dtype="float32", nodata=np.nan, **prof) as d:
        d.write(lab, 1)
    n = labels.tile_rasters(tmp_path / "a.tif", tmp_path / "chips", tmp_path / "y.tif", tile=128)
    X, Y = np.load(tmp_path / "chips/annual.npy"), np.load(tmp_path / "chips/labels.npy")
    assert n == 6 and X.shape == (6, 76, 128, 128) and X.dtype == np.uint16
    assert np.array_equal(X[0], ann[:, :128, :128]) and np.array_equal(Y[0], lab[:128, :128])
    # last kept chip starts at (128, 256): columns >= 4 are padding, rows >= 122 are NaN in the source
    assert (Y[-1][:, 4:] == -999).all() and (Y[-1][122:, :] == -999).all() and (Y[-1][:122, :4] != -999).all()


def test_earlier_input_names(tmp_path):
    """The aliases IE / I are AE / A everywhere: lookups, new configs, loaded checkpoints, projects."""
    assert channels.canonical("IE") == "AE" and channels.canonical("I") == "A" and channels.canonical("E") == "E"
    assert channels.INPUTS["IE"] is channels.INPUTS["AE"] and "I" in channels.INPUTS
    assert list(channels.INPUTS) == ["AE", "A", "E", "T", "TE"]
    with pytest.raises(ValueError, match="unknown input representation"):
        channels.canonical("X")
    cfg = models.model_config("I")
    assert cfg["input"] == "A" and cfg["n_channels"] == 12
    old = dict(cfg, input="I")                                       # a checkpoint written before the renaming
    m = models.build_model(old)
    models.save_checkpoint(tmp_path / "old.pth", m, old)
    _, loaded = models.load_checkpoint(tmp_path / "old.pth")
    assert loaded["input"] == "A"
    from canopy_height.tool.project import Project
    from shapely.geometry import box
    p = Project.create(tmp_path / "proj", box(117.70, 4.90, 117.72, 4.92), input="IE")
    assert p["input"] == "AE" and Project(p.root)["input"] == "AE"
    (p.root / "project.yaml").write_text("input: I\n", encoding="utf-8")
    assert Project(p.root)["input"] == "A"
