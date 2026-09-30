"""Command-line interface: `chm <command> --help`."""
import argparse
import json
import sys

import numpy as np

from . import channels


def _device(name):
    import torch
    if name:
        return torch.device(name)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def cmd_predict(a):
    from .data import Stack, open_stack
    if a.model.endswith((".joblib", ".pkl")):
        from .rf import load_rf, predict_rf_chips, predict_rf_raster
        rf, meta = load_rf(a.model)
        if a.annual[0].endswith(".npy"):
            p = predict_rf_chips(rf, meta, Stack(a.annual), open_stack(a.seasonal))
            np.save(a.out, p)
        else:
            p = _rf_geotiff(rf, meta, a)
        print(f"{a.out}: shape {p.shape}, mean {np.nanmean(p):.2f} m")
        return
    from .models import load_model
    from .predict import predict_geotiff, predict_stack
    dev = _device(a.device)
    zr = None if a.zero_restore is None else a.zero_restore == "on"
    model, cfg = load_model(a.model, device=dev, zero_restore=zr)
    if a.annual[0].endswith(".npy"):
        p = predict_stack(model, cfg, Stack(a.annual), open_stack(a.seasonal), a.batch, dev, a.tf32)
        np.save(a.out, p)
    else:
        p = predict_geotiff(model, cfg, a.annual[0], a.out, a.seasonal[0] if a.seasonal else None, a.tile,
                            a.overlap, a.batch, dev, a.tf32)
    print(f"{a.out}: shape {p.shape}, mean {np.nanmean(p):.2f} m")


def _rf_geotiff(rf, meta, a):
    import rasterio
    from .rf import predict_rf_raster
    with rasterio.open(a.annual[0]) as src:
        annual, prof = src.read(), src.profile
    seasonal = None
    if a.seasonal:
        with rasterio.open(a.seasonal[0]) as s:
            seasonal = s.read()
    p = predict_rf_raster(rf, meta, annual, seasonal)
    prof.update(count=1, dtype="float32", nodata=np.nan, compress="deflate", predictor=3, tiled=True,
                blockxsize=256, blockysize=256)
    with rasterio.open(a.out, "w", **prof) as dst:
        dst.write(p, 1)
    return p


def cmd_train(a):
    if a.recipe == "rf-sls":
        from .rf import train_rf
        _, meta = train_rf(a.out, a.annual, a.labels, a.seasonal, a.input or "AE", a.encoding,
                           not a.no_save_model)
        print(f"RF-SLS: {meta['n_fit']:,} pixels, hold-out RMSE {meta['holdout_rmse_m']:.3f} m -> {a.out}")
        return
    from .train import train
    over = dict(epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, patience=a.patience, select=a.select,
                seed=a.seed, split_seed=a.split_seed, num_workers=a.num_workers, gpus=a.gpus,
                zero_to_nodata=a.zero_to_nodata, input=a.input,
                trainable=a.trainable.split(",") if a.trainable else None,
                zero_restore=None if a.zero_restore is None else a.zero_restore == "on",
                milestones=[int(m) for m in a.milestones.split(",")] if a.milestones else None,
                kg2_mode=a.kg2_mode, kg2_pool=a.kg2_pool, kg2_gate=a.kg2_gate)
    if a.config:
        import yaml
        over.update({k: v for k, v in yaml.safe_load(open(a.config)).items() if over.get(k) is None})
    path, res = train(a.recipe, a.out, a.annual, a.labels, a.seasonal, a.base, a.teacher_sls,
                      _device(a.device), **over)
    print(f"model: {path} (epochs run {res['epochs_run']}, best val {res['best_val_loss']:.4f})")


def cmd_prepare_als(a):
    from .labels import prepare_als
    prepare_als(a.chm, a.grid, a.out, tuple(a.percentile), a.factor)
    print(a.out)


def cmd_tile(a):
    from .labels import tile_rasters
    n = tile_rasters(a.annual, a.out, a.labels, a.seasonal, a.tile, a.label_band, a.min_labelled)
    print(f"{n} chips -> {a.out}")


def cmd_evaluate(a):
    from .metrics import evaluate_chips
    from .data import Stack

    def load(p, band=1):
        if p.endswith(".npy"):
            return Stack(p)
        import rasterio
        with rasterio.open(p) as s:
            x = s.read(band).astype("float64")
            if s.nodata is not None and not np.isnan(s.nodata):
                x[x == s.nodata] = np.nan
        x = np.where(np.isfinite(x), x, -999.0)
        t = a.tile
        return [x[r:r + t, c:c + t] for r in range(0, x.shape[0], t) for c in range(0, x.shape[1], t)]
    mask = load(a.mask_ref, a.mask_ref_band) if a.mask_ref else None
    res = evaluate_chips(load(a.pred), load(a.ref, a.ref_band), mask, not a.no_cap80)
    res.pop("per_chip")
    print(json.dumps(res, indent=1))


def cmd_convert(a):
    from .models import convert_legacy
    cfg = convert_legacy(a.src, a.out, a.input, None if a.zero_restore is None else a.zero_restore == "on",
                         a.package_stats, name=a.name)
    print(json.dumps(cfg))


def cmd_info(a):
    from .models import describe
    print(describe(a.model))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="chm", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    onoff = ["on", "off"]

    p = sub.add_parser("predict", help="predict canopy height for a GeoTIFF or a .npy chip stack")
    p.add_argument("--model", required=True, help="UNet checkpoint (.pth) or RF-SLS model (.joblib)")
    p.add_argument("--annual", nargs="+", required=True, help="76-band annual GeoTIFF, or .npy chip stack part(s)")
    p.add_argument("--seasonal", nargs="+", help="44-band seasonal GeoTIFF / stack part(s) (T and TE models)")
    p.add_argument("--out", required=True)
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--batch", type=int, default=4, help="chips per forward pass (4 reproduces the paper arrays exactly)")
    p.add_argument("--zero-restore", choices=onoff, help="override the checkpoint setting")
    p.add_argument("--tf32", action="store_true", help="allow TF32 (faster on A100/H100, ~1e-2 m differences)")
    p.add_argument("--device")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("train", help="train or fine-tune a model on chip stacks")
    p.add_argument("--recipe", required=True,
                   choices=["unet-sls", "kg-unet1", "kg-unet2", "rf-sls", "finetune", "unet-als"])
    p.add_argument("--encoding", choices=["paper", "plain"], default="paper",
                   help="rf-sls embedding encoding (paper: uint16 wrap-around as in the paper; plain: raw values)")
    p.add_argument("--no-save-model", action="store_true", help="rf-sls: do not write the fitted forest")
    p.add_argument("--annual", nargs="+", required=True, help="annual input stack(s) .npy")
    p.add_argument("--labels", nargs="+", required=True, help="label stack(s) .npy (-999 = missing)")
    p.add_argument("--seasonal", nargs="+", help="seasonal input stack(s) .npy (T / TE inputs)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--base", help="checkpoint to start from / ALS teacher (kg-unet1, kg-unet2, finetune)")
    p.add_argument("--teacher-sls", help="GEDI teacher checkpoint (kg-unet2)")
    p.add_argument("--config", help="YAML file with settings (command-line values take precedence)")
    p.add_argument("--input", choices=channels.NAMES, help="input layout for new models (default AE): "
                   + "; ".join(f"{k} = {v}" for k, v in channels.DESCRIPTIONS.items()))
    for k, t in (("epochs", int), ("batch-size", int), ("lr", float), ("patience", int), ("seed", int),
                 ("split-seed", int), ("num-workers", int), ("gpus", int), ("kg2-pool", int), ("kg2-gate", int)):
        p.add_argument(f"--{k}", type=t)
    p.add_argument("--milestones", help="comma-separated epochs")
    p.add_argument("--select", choices=["last", "best"])
    p.add_argument("--trainable", help="comma-separated blocks to train (default: recipe)")
    p.add_argument("--zero-to-nodata", action="store_true", default=None, help="treat zero labels as missing")
    p.add_argument("--zero-restore", choices=onoff)
    p.add_argument("--kg2-mode", choices=["add", "replace"])
    p.add_argument("--device")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("prepare-als", help="aggregate 1 m ALS canopy height to the 10 m input grid")
    p.add_argument("--chm", nargs="+", required=True, help="1 m canopy-height GeoTIFF(s)")
    p.add_argument("--grid", required=True, help="GeoTIFF on the target 10 m grid (e.g. the annual input)")
    p.add_argument("--out", required=True)
    p.add_argument("--percentile", type=float, nargs="+", default=[90])
    p.add_argument("--factor", type=int, default=10)
    p.set_defaults(func=cmd_prepare_als)

    p = sub.add_parser("tile", help="cut input and label rasters into training chips")
    p.add_argument("--annual", required=True)
    p.add_argument("--labels")
    p.add_argument("--label-band", type=int, default=1)
    p.add_argument("--seasonal")
    p.add_argument("--out", required=True)
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--min-labelled", type=int, default=1)
    p.set_defaults(func=cmd_tile)

    p = sub.add_parser("evaluate", help="accuracy of predictions against a reference (GeoTIFF or .npy)")
    p.add_argument("--pred", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--ref-band", type=int, default=1)
    p.add_argument("--mask-ref", help="reference that selects the evaluated cells (e.g. p90 when --ref is p95)")
    p.add_argument("--mask-ref-band", type=int, default=1)
    p.add_argument("--tile", type=int, default=256, help="chip size for per-chip medians (GeoTIFF inputs)")
    p.add_argument("--no-cap80", action="store_true")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("convert", help="wrap an original state dict into a package checkpoint")
    p.add_argument("src"); p.add_argument("out")
    p.add_argument("--input", required=True, choices=channels.NAMES)
    p.add_argument("--zero-restore", choices=onoff)
    p.add_argument("--package-stats", action="store_true")
    p.add_argument("--name")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("info", help="show the configuration stored in a checkpoint")
    p.add_argument("model")
    p.set_defaults(func=cmd_info)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    sys.exit(main())
