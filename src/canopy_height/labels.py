"""Label preparation and tiling of rasters into training chips."""
import csv
from pathlib import Path

import numpy as np

STRIP = 64  # 10 m rows per processing strip


def block_percentile(a, f, q):
    """NaN-aware q-th percentile of every f x f block of a 2-D array (rows, cols divisible by f).

    Same operator as the paper's references (np.nanpercentile, linear interpolation, float64).
    Blocks without any valid cell are NaN.
    """
    h, w = a.shape[0] // f, a.shape[1] // f
    blk = a.reshape(h, f, w, f).transpose(0, 2, 1, 3).reshape(h * w, f * f).astype("float64")
    nv = np.count_nonzero(~np.isnan(blk), axis=1)
    out = np.full(h * w, np.nan, dtype="float64")
    full = nv == f * f
    if full.any():
        out[full] = np.percentile(blk[full], q, axis=1)
    for i in np.nonzero((nv > 0) & ~full)[0]:
        out[i] = np.nanpercentile(blk[i], q)
    return out.reshape(h, w).astype("float32")


def prepare_als(chm_paths, grid_path, out_path, percentiles=(90,), factor=10):
    """Aggregate 1 m ALS canopy-height rasters to the 10 m grid of `grid_path` (one band per percentile).

    Each 10 m cell is the percentile of the factor x factor 1 m cells inside it. The 1 m rasters are
    sampled (nearest neighbour) onto the 1 m sub-grid of the target grid; where several rasters overlap
    the first one wins. Cells without ALS are NaN.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import Window

    with rasterio.open(grid_path) as g:
        H, W, T, crs = g.height, g.width, g.transform, g.crs
    fine_T = T * Affine.scale(1.0 / factor)
    srcs = [rasterio.open(p) for p in ([chm_paths] if isinstance(chm_paths, (str, Path)) else chm_paths)]
    vrts = [WarpedVRT(s, crs=crs, transform=fine_T, width=W * factor, height=H * factor,
                      resampling=Resampling.nearest, src_nodata=s.nodata, nodata=np.nan, dtype="float32")
            for s in srcs]
    prof = dict(driver="GTiff", dtype="float32", count=len(percentiles), width=W, height=H, crs=crs, transform=T,
                nodata=np.nan, tiled=True, blockxsize=256, blockysize=256, compress="deflate", predictor=3)
    try:
        with rasterio.open(out_path, "w", **prof) as dst:
            for r0 in range(0, H, STRIP):
                h = min(STRIP, H - r0)
                win = Window(0, r0 * factor, W * factor, h * factor)
                a = np.full((h * factor, W * factor), np.nan, dtype="float32")
                for v in vrts:
                    b = v.read(1, window=win)
                    fill = np.isnan(a) & ~np.isnan(b)
                    a[fill] = b[fill]
                for k, q in enumerate(percentiles):
                    dst.write(block_percentile(a, factor, q), k + 1, window=Window(0, r0, W, h))
            for k, q in enumerate(percentiles):
                dst.set_band_description(k + 1, f"ALS canopy height p{q:g} (m)")
    finally:
        for v in vrts:
            v.close()
        for s in srcs:
            s.close()


def tile_rasters(annual_path, out_dir, labels_path=None, seasonal_path=None, tile=256, label_band=1,
                 min_labelled=1):
    """Cut rasters into [n, bands, tile, tile] chip stacks for training.

    Writes annual.npy (uint16), seasonal.npy (if given), labels.npy (float32, NaN / nodata -> -999) and
    index.csv (chip, pixel row / column of the upper-left corner, map coordinates). Edges are
    reflect-padded (inputs) or -999-padded (labels). With labels, chips with fewer than `min_labelled`
    labelled cells are skipped.
    """
    import rasterio
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    with rasterio.open(annual_path) as src:
        annual, T, crs = src.read(), src.transform, src.crs
    grids = {}
    seasonal = labels = None
    if seasonal_path:
        with rasterio.open(seasonal_path) as s:
            seasonal, grids["seasonal"] = s.read(), (s.transform, s.shape)
    if labels_path:
        with rasterio.open(labels_path) as s:
            labels = s.read(label_band).astype("float32")
            if s.nodata is not None and not np.isnan(s.nodata):
                labels[labels == s.nodata] = np.nan
            grids["labels"] = (s.transform, s.shape)
    for k, (t, shp) in grids.items():
        if t != T or shp != annual.shape[1:]:
            raise ValueError(f"{k} raster is not on the annual raster's grid")
    H, W = annual.shape[1:]
    ph, pw = (-H) % tile, (-W) % tile

    def pad(a, mode):
        if ph == 0 and pw == 0:
            return a
        widths = ((0, 0),) * (a.ndim - 2) + ((0, ph), (0, pw))
        return np.pad(a, widths, mode=mode) if mode != "constant" else np.pad(a, widths, constant_values=-999)

    annual = pad(annual, "reflect" if ph < H and pw < W else "symmetric")
    if seasonal is not None:
        seasonal = pad(seasonal, "reflect" if ph < H and pw < W else "symmetric")
    if labels is not None:
        labels = np.where(np.isfinite(labels), labels, -999).astype("float32")
        labels = pad(labels, "constant")
    keep = []
    for r in range(0, H + ph, tile):
        for c in range(0, W + pw, tile):
            if labels is not None and int((labels[r:r + tile, c:c + tile] != -999).sum()) < min_labelled:
                continue
            keep.append((r, c))
    if not keep:
        raise ValueError("no chip passes the label filter")
    np.save(out / "annual.npy", np.stack([annual[:, r:r + tile, c:c + tile] for r, c in keep]))
    if seasonal is not None:
        np.save(out / "seasonal.npy", np.stack([seasonal[:, r:r + tile, c:c + tile] for r, c in keep]))
    if labels is not None:
        np.save(out / "labels.npy", np.stack([labels[r:r + tile, c:c + tile] for r, c in keep]))
    with open(out / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chip", "row", "col", "x_ul", "y_ul", "crs"])
        for i, (r, c) in enumerate(keep):
            x, y = T * (c, r)
            w.writerow([i, r, c, x, y, crs.to_string() if crs else ""])
    return len(keep)
