"""Prediction on chip stacks and on GeoTIFF rasters of any size."""
import contextlib

import numpy as np
import torch

from . import channels
from .data import model_input


@contextlib.contextmanager
def tf32(enabled):
    """TF32 on A100/H100 GPUs changes predictions by up to ~1e-2 m; off by default for inference."""
    flags = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = enabled
    torch.backends.cuda.matmul.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = flags


def default_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def predict_batches(model, x, batch=4, device=None, allow_tf32=False):
    """x: array-like [n, C, H, W] of raw values -> float32 [n, H, W]."""
    device = device or default_device()
    model.eval()
    out = np.empty((x.shape[0],) + tuple(x.shape[2:]), dtype=np.float32)
    with tf32(allow_tf32), torch.inference_mode():
        for i in range(0, x.shape[0], batch):
            j = min(i + batch, x.shape[0])
            xb = torch.tensor(np.asarray(x[i:j], dtype=np.float32)).to(device)
            out[i:j] = model(xb).squeeze(1).cpu().numpy()
    return out


def predict_stack(model, config, annual, seasonal=None, batch=4, device=None, allow_tf32=False):
    """Predict every chip of memory-mapped stacks (data.Stack)."""
    out = []
    for i in range(0, len(annual), 64):
        j = min(i + 64, len(annual))
        out.append(predict_batches(model, model_input(config["input"], annual, seasonal, slice(i, j)),
                                   batch, device, allow_tf32))
    return np.concatenate(out)


# ---------------------------------------------------------------- rasters
def blend_window(tile, overlap):
    """Per-pixel weight of a tile: 1 in the centre, linear ramp over `overlap` pixels at the edges."""
    if overlap <= 0:
        return np.ones((tile, tile))
    i = np.arange(tile)
    r = np.minimum(1.0, np.minimum(i + 1, tile - i) / (overlap + 1.0))
    return np.outer(r, r)


def tile_origins(size, tile, overlap):
    if size <= tile:
        return [0]
    stride = tile - overlap
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] != size - tile:
        starts.append(size - tile)
    return starts


def _pad_to(a, tile):
    """Reflect-pad [C, h, w] to at least tile x tile (bottom / right)."""
    ph, pw = max(0, tile - a.shape[1]), max(0, tile - a.shape[2])
    if ph == 0 and pw == 0:
        return a
    mode = "reflect" if ph < a.shape[1] and pw < a.shape[2] else "symmetric"
    return np.pad(a, ((0, 0), (0, ph), (0, pw)), mode=mode)


def predict_array(model, config, annual, seasonal=None, tile=256, overlap=32, batch=4, device=None,
                  allow_tf32=False):
    """Predict a raster held in memory: annual [76, H, W] (+ seasonal [44, H, W]) -> float32 [H, W].

    Overlapping tiles are blended with `blend_window`. Accumulation is in float64, so a pixel covered
    by a single tile gets exactly that tile's prediction. Pixels where every annual band is 0 are NaN.
    """
    x = channels.assemble(config["input"], annual, seasonal)
    H, W = x.shape[1:]
    w = blend_window(tile, overlap)
    acc = np.zeros((H, W)); wsum = np.zeros((H, W))
    jobs = [(r, c) for r in tile_origins(H, tile, overlap) for c in tile_origins(W, tile, overlap)]
    for k in range(0, len(jobs), batch):
        chunk = jobs[k:k + batch]
        xb = np.stack([_pad_to(x[:, r:r + tile, c:c + tile], tile) for r, c in chunk])
        pb = predict_batches(model, xb, batch, device, allow_tf32)
        for (r, c), p in zip(chunk, pb):
            h, ww = min(tile, H - r), min(tile, W - c)
            acc[r:r + h, c:c + ww] += p[:h, :ww].astype(np.float64) * w[:h, :ww]
            wsum[r:r + h, c:c + ww] += w[:h, :ww]
    out = (acc / wsum).astype(np.float32)
    out[(np.asarray(annual) == 0).all(axis=0)] = np.nan
    return out


def predict_geotiff(model, config, annual_path, out_path, seasonal_path=None, tile=256, overlap=32,
                    batch=4, device=None, allow_tf32=False):
    """GeoTIFF in (76-band annual raster, optional 44-band seasonal raster) -> 1-band float32 GeoTIFF.

    The raster is read whole; for very large areas split the input first (memory is roughly
    input bytes + 16 bytes per pixel).
    """
    import rasterio
    with rasterio.open(annual_path) as src:
        annual = src.read()
        profile = src.profile
    if annual.shape[0] != 76:
        raise ValueError(f"{annual_path}: expected 76 bands, found {annual.shape[0]}")
    seasonal = None
    if channels.needs_seasonal(config["input"]):
        if seasonal_path is None:
            raise ValueError(f"model input {config['input']} needs the seasonal raster")
        with rasterio.open(seasonal_path) as s:
            if (s.width, s.height, s.transform, s.crs) != (profile["width"], profile["height"],
                                                          profile["transform"], profile["crs"]):
                raise ValueError("annual and seasonal rasters must share the same grid")
            seasonal = s.read()
    pred = predict_array(model, config, annual, seasonal, tile, overlap, batch, device, allow_tf32)
    profile.update(count=1, dtype="float32", nodata=np.nan, compress="deflate", predictor=3,
                   tiled=True, blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(pred, 1)
        dst.set_band_description(1, "canopy height (m)")
    return pred
