"""tool/pretrained.py: regions, study-area checks and planning, and the mapping run with stubbed downloads."""
import os
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
shapely = pytest.importorskip("shapely")

from canopy_height.tool import download, pretrained, weights  # noqa: E402
from canopy_height.tool.gee import io as gee_io  # noqa: E402


def test_regions_models_and_default_areas():
    assert list(pretrained.REGIONS) == ["CONUS", "EBR", "MRF", "MUR", "SER", "SPC"]
    assert pretrained.model_files("CONUS") == {"UNet-ALS": "source/UNet-ALS.pth"}
    assert pretrained.model_files("SER") == {"UNet-ALS": "source/UNet-ALS.pth",
                                             **{m: f"SER/{m}.pth" for m in ("UNet-SLS", "KG-UNet1", "KG-UNet2")}}
    assert pretrained.label("MRF") == "MRF (New Zealand)" and pretrained.label("CONUS") == "CONUS (United States)"
    for r in pretrained.REGIONS:
        assert all(rel in weights.SHA256 for rel in pretrained.model_files(r).values())
        assert shapely.box(*pretrained.view_bounds(r)).contains(pretrained.default_aoi(r))
        if r != "CONUS":
            assert pretrained.region_geometry(r).contains(pretrained.default_aoi(r))
    w, s, e, n = pretrained.default_aoi("CONUS").bounds
    assert -125 < w < e < -66 and 24 < s < n < 50
    assert pretrained.REGIONS["SER"]["year"] == 2020 and pretrained.REGIONS["EBR"]["year"] == 2019


def test_check_study_areas():
    info = pretrained.check("MRF", pretrained.default_aoi("MRF"))
    assert info["inside"] == pytest.approx(1.0) and 4 <= info["cells"] <= 16 and info["year"] == 2020
    assert info["notes"] == []
    assert "trained with 2020 inputs, not 2022" in pretrained.check("MRF", pretrained.default_aoi("MRF"), 2022)["notes"][0]
    w, s, e, n = pretrained.region_geometry("MUR").bounds
    dw, ds, de, dn = pretrained.default_aoi("MUR").bounds
    part = pretrained.check("MUR", shapely.box(dw, ds, e + 0.2, dn))        # reaches beyond the region
    assert 0 < part["inside"] < 1 and "outside the MUR region" in part["notes"][0]
    with pytest.raises(ValueError, match="outside the MUR region"):
        pretrained.check("MUR", shapely.box(10, 10, 10.05, 10.05))
    with pytest.raises(ValueError, match="more than 200"):
        pretrained.check("MUR", shapely.box(w, s, e, n))


def test_plan_and_open_or_create(tmp_path):
    aoi = pretrained.default_aoi("SPC")
    p = pretrained.create(tmp_path / "p", "SPC", aoi, gee_project="stub-project")
    assert pretrained.region_of(p) == "SPC" and p["input"] == "AE" and p["year"] == 2019 and p["benchmarks"] == []
    cells = pretrained.plan(p)
    assert (cells.role == "aoi").all() and cells.use.all() and len(cells) == pretrained.check("SPC", aoi)["cells"]
    assert pretrained.open_or_create(tmp_path / "p", "SPC", aoi, workers=1)["workers"] == 1
    with pytest.raises(ValueError, match="another study area, region or year"):
        pretrained.open_or_create(tmp_path / "p", "SPC", aoi, year=2020)
    with pytest.raises(ValueError, match="another study area, region or year"):
        pretrained.open_or_create(tmp_path / "p", "MUR", pretrained.default_aoi("MUR"))
    p.close_log()


def _unet_als():
    d = os.environ.get("CHM_WEIGHTS")
    f = Path(d) / "source" / "UNet-ALS.pth" if d else None
    if f is None or not f.exists():
        pytest.skip("UNet-ALS checkpoint not found ($CHM_WEIGHTS/source/UNet-ALS.pth)")
    return Path(d)


def test_run_maps_inside_the_region_only(tmp_path, monkeypatch):
    """CONUS run with stubbed Earth Engine; the 'region' covers the western half of the study area only."""
    pytest.importorskip("torch")
    src = _unet_als()
    aoi = pretrained.default_aoi("CONUS")
    w, s, e, n = aoi.bounds
    west_half = shapely.box(w - 1, s - 1, (w + e) / 2, n + 1)
    monkeypatch.setattr(pretrained, "region_geometry", lambda region, cache_dir=None: west_half)
    calls = []

    def stub(layers, out, epsg, x0, y1, year, *a, **k):                # inputs of one cell, constant values
        calls.append(tuple(layers))
        for layer in layers:
            nb = {"Embedding": 64, "S1": 2, "S2": 9}.get(layer, 1)
            v = {"Embedding": 0.1, "S1": -10.0, "S2": 500.0, "DEM": 300.0}.get(layer, 1.0)
            dtype = download.INT_LAYERS.get(layer, ("float64", None))[0]
            download._write(out[layer], np.full((nb, 256, 256), v).astype(dtype), dtype,
                            [f"b{i}" for i in range(nb)], epsg, x0, y1, 10.0, year=str(year))
        return "ok"
    monkeypatch.setattr(download, "export", stub)
    monkeypatch.setattr(download, "s1_regions", lambda *a, **k: {})
    monkeypatch.setattr(gee_io, "ee_init", lambda *a, **k: None)
    p = pretrained.create(tmp_path / "p", "CONUS", aoi, gee_project="stub-project", workers=2)
    maps = pretrained.run(p, weights_source=src, weights_dir=tmp_path / "w", log=lambda *a: None)
    assert list(maps) == ["UNet-ALS"] and calls and not any("GEDI" in c for c in calls)
    with rasterio.open(maps["UNet-ALS"]) as m:
        a = m.read(1)
        assert m.tags()["region"] == "CONUS" and m.tags()["model"] == "UNet-ALS"
    cols = np.where(np.isfinite(a).any(axis=0))[0]
    assert cols.size and cols.max() < a.shape[1] * 0.6                 # nothing east of the region's edge
    assert (tmp_path / "w" / "source" / "UNet-ALS.pth").exists()
    n = len(calls)
    pretrained.run(p, weights_source=src, weights_dir=tmp_path / "w", log=lambda *a: None)
    assert len(calls) == n                                              # inputs kept: nothing downloaded again


def test_weights_fetch_from_a_folder(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "KG-UNet2.pth").write_bytes(b"weights")
    monkeypatch.setitem(weights.SHA256, "SER/KG-UNet2.pth", weights.sha256(tmp_path / "src" / "KG-UNet2.pth"))
    out = weights.fetch("SER/KG-UNet2.pth", tmp_path / "src", tmp_path / "dest", log=lambda *a: None)
    assert out == tmp_path / "dest" / "SER" / "KG-UNet2.pth" and out.read_bytes() == b"weights"
    monkeypatch.setitem(weights.SHA256, "SER/KG-UNet2.pth", "0" * 64)
    out.unlink()
    with pytest.raises(ValueError, match="differs from the published"):
        weights.fetch("SER/KG-UNet2.pth", tmp_path / "src", tmp_path / "dest", log=lambda *a: None)
