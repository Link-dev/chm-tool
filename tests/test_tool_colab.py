"""canopy_height.tool.colab: session settings, weights check, the study-area map, the local stack copy."""
import sys
import types

import numpy as np
import pytest

from canopy_height.tool import colab
from canopy_height.tool.project import Project

shapely = pytest.importorskip("shapely")


def _project(tmp_path):
    p = Project.create(tmp_path / "proj", shapely.box(117.70, 4.90, 117.72, 4.92), year=2020)
    d = p.path("stacks")
    d.mkdir()
    np.save(d / "train_part001.npy", np.zeros((2, 76, 4, 4), np.uint16))
    np.save(d / "train_GEDI_part001.npy", np.zeros((2, 1, 4, 4), np.float32))
    (d / "hashes.json").write_text('{"files": {"a": "1"}}')
    return p


def test_map_view():
    (lat, lon), z = colab.map_view((173.0, -41.6, 173.9, -41.0))          # a site region, about 75 x 65 km
    assert (lat, lon) == pytest.approx((-41.3, 173.45)) and z == 9
    assert colab.map_view((-72.22, 42.50, -72.13, 42.57))[1] == 12          # 7 x 8 km
    assert colab.map_view((-125, 24, -66, 50))[1] == 4                      # the contiguous US


class _Widget:
    def __init__(self, *a, **k):
        self.kw = k


class _Map(_Widget):
    def __init__(self, **k):
        super().__init__(**k)
        self.layers = ()

    def add(self, layer):
        self.layers += (layer,)

    def remove(self, layer):
        self.layers = tuple(x for x in self.layers if x is not layer)


class _DrawControl(_Widget):
    def on_draw(self, fn):
        self.fn = fn


def _fake_ipyleaflet(monkeypatch):
    L = types.SimpleNamespace(Map=_Map, TileLayer=_Widget, GeoJSON=_Widget, DrawControl=_DrawControl,
                              SearchControl=_Widget)
    monkeypatch.setitem(sys.modules, "ipyleaflet", L)


def test_drawer_uses_the_default_area_until_a_shape_is_drawn(monkeypatch):
    _fake_ipyleaflet(monkeypatch)
    default, region = shapely.box(173.44, -41.29, 173.53, -41.22), shapely.box(173.0, -41.6, 173.9, -41.0)
    d = colab.AoiDrawer(default, outline=region, bounds=region.bounds)
    assert d.map.kw["zoom"] == 9 and d.aoi is default and d._default_layer in d.map.layers
    dc = next(x for x in d.map.layers if isinstance(x, _DrawControl))
    drawn = shapely.box(173.5, -41.4, 173.6, -41.3)
    dc.fn(dc, action="created", geo_json={"geometry": shapely.geometry.mapping(drawn)})
    assert d.aoi.equals(drawn) and d._default_layer not in d.map.layers          # the default area is hidden
    dc.fn(dc, action="deleted", geo_json={"geometry": shapely.geometry.mapping(shapely.box(0, 0, 1, 1))})
    assert d.aoi.equals(drawn)                                                   # another shape was deleted
    dc.fn(dc, action="deleted", geo_json={"geometry": shapely.geometry.mapping(drawn)})
    assert d.aoi is default and d._default_layer in d.map.layers


def test_draw_map_falls_back_to_the_default_area(monkeypatch):
    monkeypatch.setitem(sys.modules, "ipyleaflet", None)                         # import fails
    notes = []
    assert colab.draw_map(shapely.box(0, 0, 1, 1), log=notes.append) is None
    assert "default study area" in notes[0]


def test_settings_low_ram_and_gpu():
    s, w = colab.colab_settings(dict(ram_gb=12.7, gpu="Tesla T4", gpu_gb=14.7))
    assert s == {"rf_max_rows": colab.COLAB_RF_MAX_ROWS} and w == []
    s, w = colab.colab_settings(dict(ram_gb=64, gpu=None, gpu_gb=0))
    assert s == {} and len(w) == 1
    assert len(colab.colab_settings(dict(ram_gb=64, gpu="small", gpu_gb=8))[1]) == 1


def test_local_stack_copy_used_only_while_current(tmp_path):
    p = _project(tmp_path)
    lp = colab.LocalStacks(p.root, tmp_path / "cache")
    assert lp.stack_files("x") == p.stack_files("x")                 # no copy yet
    lp.cache.mkdir(parents=True)
    for f in p.path("stacks").iterdir():
        (lp.cache / f.name).write_bytes(f.read_bytes())
    assert lp.stack_files("x") == [str(lp.cache / "train_part001.npy")]
    assert lp.stack_files("GEDI") == [str(lp.cache / "train_GEDI_part001.npy")]
    p.path("stacks", "hashes.json").write_text('{"files": {"a": "2"}}')   # stack rebuilt: the copy is stale
    assert lp.stack_files("x") == p.stack_files("x")


def _enotconn():
    import errno
    return OSError(errno.ENOTCONN, "Transport endpoint is not connected")


def test_is_disconnect_follows_the_exception_chain():
    try:
        try:
            raise _enotconn()
        except OSError:
            raise RuntimeError("stage failed")
    except RuntimeError as e:
        assert colab.is_disconnect(e)
    assert not colab.is_disconnect(OSError(2, "No such file"))
    assert not colab.is_disconnect(ValueError("x"))


def test_run_remounts_drive_and_continues(tmp_path, monkeypatch):
    p = _project(tmp_path)
    calls, mounts = [], []

    def fake_run(*a):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("download failed") from _enotconn()
        return "status"
    monkeypatch.setattr(colab, "_run", fake_run)
    monkeypatch.setattr(colab, "remount_drive", lambda log=print: mounts.append(1) or True)
    assert colab.run(p, log=lambda *a: None) == "status"
    assert len(calls) == 2 and len(mounts) == 1


def test_run_gives_up_after_repeated_disconnects(tmp_path, monkeypatch):
    p = _project(tmp_path)

    def fake_run(*a):
        raise _enotconn()
    monkeypatch.setattr(colab, "_run", fake_run)
    monkeypatch.setattr(colab, "remount_drive", lambda log=print: True)
    with pytest.raises(colab.DriveDisconnected, match="force_remount"):
        colab.run(p, log=lambda *a: None, remounts=2)


def test_run_does_not_hide_other_errors(tmp_path, monkeypatch):
    p = _project(tmp_path)

    def fake_run(*a):
        raise ValueError("a real bug")
    monkeypatch.setattr(colab, "_run", fake_run)
    with pytest.raises(ValueError, match="a real bug"):
        colab.run(p, log=lambda *a: None)


def test_loader_workers_capped_to_cores_in_memory_only(tmp_path, monkeypatch):
    p = _project(tmp_path)
    p.cfg["train_overrides"] = {"kg-unet2": {"epochs": 30, "num_workers": 1}}
    monkeypatch.setattr(colab.os, "cpu_count", lambda: 2)
    colab.cap_loader_workers(p, log=lambda *a: None)
    o = p.cfg["train_overrides"]
    assert o["unet-sls"] == {"num_workers": 2} and o["kg-unet1"] == {"num_workers": 2}
    assert o["kg-unet2"] == {"epochs": 30, "num_workers": 1}          # the user's own setting is kept
    assert "rf-sls" not in o
    assert Project(p.root).cfg["train_overrides"] == {}                 # project.yaml unchanged
    q = _project(tmp_path / "b")
    monkeypatch.setattr(colab.os, "cpu_count", lambda: 20)
    colab.cap_loader_workers(q, log=lambda *a: None)
    assert q.cfg["train_overrides"] == {}


def test_fetch_weights_checks_sha256(tmp_path, monkeypatch):
    src = tmp_path / "w" / "source"
    src.mkdir(parents=True)
    (src / "UNet-E-ALS.pth").write_bytes(b"not the checkpoint")
    monkeypatch.delenv("CHM_WEIGHTS", raising=False)
    with pytest.raises(ValueError, match="sha256"):
        colab.fetch_weights("E", tmp_path / "w", dest=tmp_path / "dest", log=lambda *a: None)
    assert not (tmp_path / "dest" / "source" / "UNet-E-ALS.pth").exists()
    monkeypatch.setitem(colab.weights.SHA256, "source/UNet-E-ALS.pth", colab.weights.sha256(src / "UNet-E-ALS.pth"))
    out = colab.fetch_weights("E", tmp_path / "w", dest=tmp_path / "dest", log=lambda *a: None)
    assert out.exists()
    import os
    assert os.environ["CHM_WEIGHTS"] == str(tmp_path / "dest")
