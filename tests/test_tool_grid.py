"""Study area reading and cell planning of the canopy-height tool (no Earth Engine: land cover is stubbed)."""
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

gpd = pytest.importorskip("geopandas")
import shapely  # noqa: E402

from canopy_height.tool import grid  # noqa: E402
from canopy_height.tool.project import Project  # noqa: E402

REF_UTM = shapely.box(584670.0, 545560.0, 589220.0, 550110.0)    # a forest footprint, EPSG:32650
REF_GRID = {"epsg": 32650, "x0": 584670.0, "y1": 550110.0, "res": 10.0, "cell_px": 256}
AOI_CELLS = ["x584670y550110", "x587230y550110", "x584670y547550", "x587230y547550"]
RING_1KM = ["x582110y552670", "x584670y552670", "x587230y552670", "x582110y550110", "x582110y547550",   # 0 m
            "x589790y552670", "x589790y550110", "x589790y547550",                                        # 570 m
            "x582110y544990", "x584670y544990", "x587230y544990",
            "x589790y544990"]                                                                            # 806 m
RING_1KM_DIST = [0] * 5 + [570] * 6 + [806]
RING_3KM_MORE = ["x582110y555230", "x584670y555230", "x587230y555230",                                   # 2560 m
                 "x579550y552670", "x579550y550110", "x579550y547550",
                 "x589790y555230", "x579550y544990"]                                                     # 2623 m


def ref4326():
    return grid.to_crs(REF_UTM, 4326, src_epsg=32650)


class Stub:
    """Land-cover lookup without Earth Engine: records the cells asked for; water share by cell name."""

    def __init__(self, water=None):
        self.water, self.asked = water or {}, []

    def __call__(self, cells, epsg):
        assert epsg == 32650
        names = [grid.cell_name(x, y) for x, y in cells]
        self.asked.append(names)
        return [(self.water.get(n, 0.0), 0.01, 0.9) for n in names]


def no_gee(cells, epsg):
    raise AssertionError(f"land cover looked up again for {len(cells)} cells")


def project(tmp_path, **settings):
    return Project.create(tmp_path / "p", ref4326(), **settings)


def assert_ref_xy(aoi4326):
    xy = grid.to_crs(aoi4326, 32650)
    assert np.allclose(xy.bounds, REF_UTM.bounds, rtol=0, atol=1e-6)
    assert abs(xy.area - REF_UTM.area) < 1e-3
    assert grid.make_grid(xy, 32650) == REF_GRID


# ------------------------------------------------------------------------------------------------ grid
def test_reference_grid(tmp_path):
    f = tmp_path / "aoi.geojson"
    grid.write_aoi(ref4326(), f)
    fc = json.load(open(f, encoding="utf-8"))
    assert fc["type"] == "FeatureCollection" and len(fc["features"]) == 1
    aoi = grid.read_aoi(f)
    assert aoi.geom_type == "Polygon" and grid.utm_epsg(aoi) == 32650
    assert_ref_xy(aoi)
    assert grid.cell_box(REF_GRID, 0, 0).bounds == (584670.0, 547550.0, 587230.0, 550110.0)
    assert grid.cell_box(REF_GRID, -1, -1).bounds == (582110.0, 550110.0, 584670.0, 552670.0)
    assert grid.utm_epsg(shapely.box(-47.1, -23.6, -47.0, -23.5)) == 32723


def test_plan_ring(tmp_path):
    p = project(tmp_path, train_cells=16)
    stub = Stub()
    df = grid.plan_cells(p, stub)
    assert list(df.columns) == grid.CELL_COLS
    assert list(df.cell) == AOI_CELLS + RING_1KM
    assert list(df.role) == ["aoi"] * 4 + ["ring"] * 12
    assert list(df.dist_m) == [0] * 4 + RING_1KM_DIST
    assert list(zip(df.col[:4], df.row[:4])) == [(0, 0), (1, 0), (0, 1), (1, 1)]
    assert df.use.all() and df.attrs["ring_m"] == 1000
    assert json.load(open(p.grid_file)) == REF_GRID
    # land cover: study-area cells first, then only the cells within the current D, each once
    assert stub.asked[0] == AOI_CELLS and [len(a) for a in stub.asked] == [4, 5, 7]
    assert sorted(sum(stub.asked, [])) == sorted(df.cell)
    back = pd.read_csv(p.cells_file)
    assert list(back.cell) == list(df.cell) and back.use.dtype == bool and back.dist_m.tolist() == df.dist_m.tolist()
    assert len(pd.read_csv(p.path("plan", "landcover.csv"))) == 16
    again = grid.plan_cells(p, no_gee)                            # cached land cover, same plan
    pd.testing.assert_frame_equal(again, df)


def test_plan_water(tmp_path):
    wet = {"x589790y544990": 0.99}
    stub = Stub(wet)
    df = grid.plan_cells(project(tmp_path, train_cells=16), stub)
    # 15 usable cells within 1 km -> D grows; nothing new at 2 km; the 2.56 km cells come in at 3 km
    assert df.attrs["ring_m"] == 3000
    assert list(df.cell) == AOI_CELLS + RING_1KM + RING_3KM_MORE
    assert list(df.cell[~df.use]) == ["x589790y544990"] and df.water[~df.use].iloc[0] == 0.99
    assert df.use.sum() == 23 and [len(a) for a in stub.asked] == [4, 5, 7, 8]
    df = grid.plan_cells(project(tmp_path / "b", train_cells=15), Stub(wet))
    assert df.attrs["ring_m"] == 1000 and len(df) == 16 and df.use.sum() == 15
    df = grid.plan_cells(project(tmp_path / "c", train_cells=15, water_max=0.999), Stub(wet))
    assert df.use.all()


def test_plan_no_ring_and_cap(tmp_path):
    for n in (4, 3):
        stub = Stub()
        df = grid.plan_cells(project(tmp_path / str(n), train_cells=n), stub)
        assert list(df.cell) == AOI_CELLS and (df.role == "aoi").all() and df.attrs["ring_m"] is None
        assert stub.asked == [AOI_CELLS]
    df = grid.plan_cells(project(tmp_path / "cap", train_cells=400, ring_max_km=0.5), Stub())
    assert df.attrs["ring_m"] == 500 and list(df.cell) == AOI_CELLS + RING_1KM[:5]
    df = grid.plan_cells(project(tmp_path / "epsg", train_cells=16, epsg=32650), Stub())
    assert list(df.cell) == AOI_CELLS + RING_1KM


def _log(p):
    return p.log_file.read_text(encoding="utf-8") if p.log_file.exists() else ""


def test_plan_large_study_area_warns(tmp_path):
    p = project(tmp_path, train_cells=3)
    df = grid.plan_cells(p, Stub())
    assert list(df.cell) == AOI_CELLS and df.use.all()                  # no study-area cell is dropped
    c = grid.training_cost(4)
    assert "WARNING the study area alone has 4 cells, more than train_cells 3: all 4 are training cells" in _log(p)
    assert abs(c["stack_gb"] - 4 * 10.22e6 / 1e9) < 1e-3 and c["unet_x"] == 0.01
    assert abs(grid.training_cost(3481)["stack_gb"] - 35.6) < 0.1                # the 150 km study area
    q = project(tmp_path / "q", train_cells=4)
    grid.plan_cells(q, Stub())
    assert "WARNING" not in _log(q)


def test_plan_checks_before_earth_engine(tmp_path):
    p = project(tmp_path)
    for k, v, msg in (("epsg", 2289, "not a projected CRS in metres"),      # US survey feet
                      ("epsg", 4326, "not a projected CRS in metres"),
                      ("epsg", 32640, "distorts distances by"),             # UTM zone 60 degrees away
                      ("year", 2015, "choose 2019-2024")):
        p.cfg[k] = v
        with pytest.raises(ValueError, match=msg):
            grid.plan_cells(p, no_gee)
        p.cfg[k] = Project(p.root).cfg[k]
    p.cfg.update(epsg=32652, train_cells=4)                                  # 11 degrees away: ~1.9 %
    grid.plan_cells(p, lambda cells, epsg: [(0.0, 0.0, 0.9)] * len(cells))
    assert "WARNING EPSG:32652 distorts distances by up to 1.9%" in _log(p)
    assert 0.018 < grid.scale_error(ref4326(), 32652) < 0.02 and grid.scale_error(ref4326(), 32650) < 1e-3
    assert grid.scale_error(shapely.box(10, 44.9, 10.1, 45), 3857) > 0.4

    x, y = REF_UTM.bounds[0], REF_UTM.bounds[3]
    tiny = grid.to_crs(shapely.box(x + 1, y - 4, x + 4, y - 1), 4326, src_epsg=32650)  # 3 m, no pixel centre
    with pytest.raises(ValueError, match="no 10 m pixel centre"):
        grid.plan_cells(Project.create(tmp_path / "tiny", tiny), no_gee)
    small = grid.to_crs(shapely.box(x + 1, y - 29, x + 29, y - 1), 4326, src_epsg=32650)   # 3 x 3 pixels
    q = Project.create(tmp_path / "small", small, train_cells=1)
    grid.plan_cells(q, Stub())
    assert "WARNING the study area covers only 9 pixels of 10 m" in _log(q)
    assert grid.aoi_pixels(REF_UTM, pd.DataFrame({"x0": [584670.0], "y1": [550110.0]})) == 256 * 256


# ------------------------------------------------------------------------------------------------ read_aoi
def test_read_shapefile_and_zip(tmp_path):
    d = tmp_path / "shp"
    d.mkdir()
    gpd.GeoDataFrame({"id": [1]}, geometry=[REF_UTM], crs=32650).to_file(d / "area.shp")
    assert_ref_xy(grid.read_aoi(d / "area.shp"))
    z = tmp_path / "area.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in d.iterdir():
            zf.write(f, f"sub/{f.name}")
    assert_ref_xy(grid.read_aoi(z))
    assert_ref_xy(grid.read_aoi(str(z)))


def test_read_featurecollection(tmp_path):
    a, b, c = shapely.box(117.0, 4.0, 117.1, 4.1), shapely.box(117.05, 4.05, 117.15, 4.15), shapely.box(118, 5, 118.1, 5.1)
    feats = [{"type": "Feature", "properties": {"k": i}, "geometry": shapely.geometry.mapping(g)}
             for i, g in enumerate([a, b, c, shapely.Point(117.5, 4.5)])]
    fc = {"type": "FeatureCollection", "features": feats}
    f = tmp_path / "several.geojson"
    f.write_text(json.dumps(fc), encoding="utf-8")
    aoi = grid.read_aoi(f)
    assert aoi.geom_type == "MultiPolygon" and len(aoi.geoms) == 2
    assert abs(aoi.area - shapely.union_all([a, b, c]).area) < 1e-12
    assert grid.read_aoi(fc).equals(aoi)
    assert grid.read_aoi(feats[0]).equals(a) and grid.read_aoi(feats[0]["geometry"]).equals(a)
    assert grid.read_aoi(a).equals(a)


def test_read_projected_geopackage(tmp_path):
    f = tmp_path / "area.gpkg"
    west, east = shapely.box(584670, 545560, 587000, 550110), shapely.box(587000, 545560, 589220, 550110)
    gpd.GeoDataFrame(geometry=[west], crs=32650).to_file(f, layer="west", driver="GPKG")
    gpd.GeoDataFrame(geometry=[east], crs=32650).to_file(f, layer="east", driver="GPKG")
    aoi = grid.read_aoi(f)
    assert aoi.geom_type == "Polygon"
    assert_ref_xy(aoi)
    legacy = {"type": "Feature", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32650"}},
              "geometry": shapely.geometry.mapping(REF_UTM)}
    assert_ref_xy(grid.read_aoi(legacy))


def test_read_kml(tmp_path):
    pyogrio = pytest.importorskip("pyogrio")
    drv = pyogrio.list_drivers(write=True)
    if "KML" not in drv and "LIBKML" not in drv:
        pytest.skip("no KML driver")
    f = tmp_path / "area.kml"
    gpd.GeoDataFrame({"Name": ["area"]}, geometry=[ref4326()], crs=4326).to_file(
        f, driver="KML" if "KML" in drv else "LIBKML")
    assert_ref_xy(grid.read_aoi(f))


@pytest.mark.filterwarnings("ignore:'crs' was not provided")
def test_read_errors(tmp_path):
    gpd.GeoDataFrame(geometry=[REF_UTM]).to_file(tmp_path / "nocrs_utm.shp")
    with pytest.raises(ValueError, match="not longitude / latitude"):
        grid.read_aoi(tmp_path / "nocrs_utm.shp")
    gpd.GeoDataFrame(geometry=[ref4326()]).to_file(tmp_path / "nocrs_ll.shp")
    assert_ref_xy(grid.read_aoi(tmp_path / "nocrs_ll.shp"))
    with pytest.raises(ValueError, match="no polygon"):
        grid.read_aoi({"type": "MultiPoint", "coordinates": [[117.8, 4.9], [117.9, 5.0]]})
    with pytest.raises(FileNotFoundError):
        grid.read_aoi(tmp_path / "missing.geojson")


def test_antimeridian(tmp_path):
    fiji = shapely.MultiPolygon([shapely.box(179.6, -16.9, 180, -16.5), shapely.box(-180, -16.9, -179.8, -16.5)])
    with pytest.raises(ValueError, match="crosses the 180th meridian"):
        grid.read_aoi(fiji)
    with pytest.raises(ValueError, match="crosses the 180th meridian"):
        grid.utm_epsg(fiji)
    with pytest.raises(ValueError, match="crosses the 180th meridian"):
        Project.create(tmp_path / "p", fiji)
    wrapped = {"type": "Polygon", "coordinates": [[[179.6, -16.9], [180.2, -16.9], [180.2, -16.5], [179.6, -16.9]]]}
    with pytest.raises(ValueError, match="crosses the 180th meridian"):
        grid.read_aoi(wrapped)
    assert grid.utm_epsg(shapely.box(179.6, -16.9, 179.99, -16.5)) == 32760       # east of it: zone 60 S
    assert grid.utm_epsg(shapely.box(-179.99, -16.9, -179.8, -16.5)) == 32701     # west of it: zone 1 S


def test_read_non_utf8_attributes(tmp_path, monkeypatch):
    """Attributes are never needed: a GeoJSON in GBK (e.g. json.dump on a Chinese Windows) reads fine."""
    fc = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"name": "保护区"},
                                                     "geometry": shapely.geometry.mapping(ref4326())}]}
    f = tmp_path / "gbk.geojson"
    f.write_bytes(json.dumps(fc, ensure_ascii=False).encode("gbk"))
    with pytest.raises(UnicodeDecodeError):
        f.read_bytes().decode("utf-8")
    assert_ref_xy(grid.read_aoi(f))
    monkeypatch.setattr(grid.locale, "getpreferredencoding", lambda do_setlocale=True: "utf-8")
    assert_ref_xy(grid.read_aoi(f))                                  # GDAL, geometry only
    d = tmp_path / "shp"
    d.mkdir()
    gpd.GeoDataFrame({"name": ["研究区"]}, geometry=[REF_UTM], crs=32650).to_file(d / "a.shp", encoding="gbk")
    (d / "a.cpg").write_text("UTF-8")                                  # wrong code page
    assert_ref_xy(grid.read_aoi(d / "a.shp"))


# ------------------------------------------------------------------------------------------------ import_cells
def rows_table(n=3, **cols):
    t = pd.DataFrame({"part": 1, "row_in_part": range(n), "piece": [f"P{i}" for i in range(n)], "tile": 1,
                      "kind": ["test", "near", "far"][:n], "cluster": "X", "min_dist_m": [0.0, 10.0, 1500.0][:n],
                      "epsg": 32650, "tile_x0": [584670.0, 587230.0, 582110.0][:n],
                      "tile_y1": [550110.0, 547550.0, 552670.0][:n], "padded": False,
                      "x_src": r"D:\x", "lab_src": r"D:\lab"})
    for k, v in cols.items():
        t[k] = v
    return t


def test_import_cells_table(tmp_path):
    p = project(tmp_path)
    df = grid.import_cells(p, rows_table())
    assert list(df.columns) == grid.CELL_COLS + grid.IMPORT_COLS
    assert list(df.cell) == ["P0", "P1", "P2"] and list(df.role) == ["aoi", "ring", "ring"]
    assert list(df.dist_m) == [0, 10, 1500] and df.use.all() and df.water.isna().all() and df.col.isna().all()
    g = json.load(open(p.grid_file))
    assert g == {"epsg": 32650, "x0": 582110.0, "y1": 552670.0, "res": 10.0, "cell_px": 256,
                 "cells_on_common_grid": True}
    back = pd.read_csv(p.cells_file).iloc[1]
    assert p.cell_raster(back, "S2") == Path(r"D:\x") / "P1_S2.tif"
    assert p.cell_raster(back, "GEDI") == Path(r"D:\lab") / "P1_GEDI.tif"
    grid.import_cells(p, rows_table(tile_x0=[584670.0, 587230.0, 582120.0]))
    assert json.load(open(p.grid_file))["cells_on_common_grid"] is False
    grid.import_cells(p, rows_table(epsg=[32650, 32650, 32651]))
    assert json.load(open(p.grid_file))["epsg"] is None
    with pytest.raises(ValueError, match="not whole"):
        grid.import_cells(p, rows_table(padded=[False, True, False]))
    with pytest.raises(ValueError, match="not whole"):
        grid.import_cells(p, rows_table(tile=[1, 2, 1]))

    # the study area must cover a pixel of the imported study-area cells (their maps)
    far = Project.create(tmp_path / "far", shapely.box(10.0, 50.0, 10.01, 50.01))
    with pytest.raises(ValueError, match="covers no pixel of the 1 imported study-area cells"):
        grid.import_cells(far, rows_table())
    assert not far.cells_file.exists() and not far.grid_file.exists()
    assert len(grid.import_cells(far, rows_table(kind=["near", "near", "far"]))) == 3     # no study-area cells
    assert len(grid.import_cells(far, rows_table(tile_x0=[584670.0, 587230.0, 582120.0]))) == 3   # no common grid
