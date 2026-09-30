"""Training recipes.

    unet-als   UNet from random initialisation on ALS canopy height (the source-domain model)
    unet-sls   UNet from random initialisation on GEDI labels
    kg-unet1   start from a UNet-ALS checkpoint; fine-tune only the last two blocks (up4, outc)
    kg-unet2   as kg-unet1, plus teacher terms from the UNet-ALS checkpoint (image-gradient agreement)
               and a UNet-SLS checkpoint (agreement after k x k average pooling)
    finetune   start from any checkpoint; train all blocks or the listed ones

`zero_restore` (reset no-data inputs to 0 after normalisation) defaults to the input layout's setting
for new models and to the checkpoint's setting when starting from one.

Defaults are the settings used in the paper (unet-sls / kg-unet1 / kg-unet2: international sites;
unet-als: 3DEP pre-training). Every setting can be overridden. Labels of -999 are missing and ignored;
batches without any labelled cell are skipped. With a fixed seed, training is repeatable on the same
hardware and software.
"""
import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import channels, stats
from .data import Stack, labelled_pixels, open_stack
from .losses import avgpool_loss, charbonnier, edge_aware_gradient_loss
from .models import build_model, load_model, model_config, save_checkpoint
from .torch_data import ChipDataset

_COMMON = dict(input="AE", init=None, trainable=None, epochs=100, batch_size=25, lr=5e-4, milestones=[50],
               gamma=0.5, split="train_test_split", val_frac=0.2, split_seed=42, seed=42, select="last",
               patience=None, zero_to_nodata=False, zero_restore=None, kg2=None, num_workers=None,
               gpus=1, val_workers=1)
RECIPES = {
    "unet-sls": dict(_COMMON),
    "kg-unet1": dict(_COMMON, init="base", trainable=["up4", "outc"], epochs=50),
    "kg-unet2": dict(_COMMON, init="base", trainable=["up4", "outc"], epochs=50, zero_to_nodata=True,
                     kg2=dict(mode="add", pool=4, gate=10, w_label=0.5, w_als=0.25, w_sls=0.25)),
    "unet-als": dict(_COMMON, epochs=100, batch_size=150, lr=1e-4, milestones=[150], gamma=0.8,
                     split="shuffle", split_seed=43, seed=43, select="best", patience=10, num_workers=24,
                     gpus=-1),
    "finetune": dict(_COMMON, init="base", epochs=50),
}


def resolve(recipe, **overrides):
    cfg = copy.deepcopy(RECIPES[recipe])
    for k, v in overrides.items():
        if v is None:
            continue
        if k.startswith("kg2_"):
            if cfg["kg2"] is None:
                raise ValueError(f"{k} applies to kg-unet2 only")
            cfg["kg2"][k[4:]] = v
        elif k in cfg:
            cfg[k] = v
        else:
            raise ValueError(f"unknown setting {k}")
    cfg["recipe"] = recipe
    cfg["input"] = channels.canonical(cfg["input"])
    return cfg


def split_chips(n, cfg):
    """Chip-level train / validation split."""
    if cfg["split"] == "train_test_split":
        from sklearn.model_selection import train_test_split
        return train_test_split(np.arange(n), test_size=cfg["val_frac"], random_state=cfg["split_seed"])
    if cfg["split"] == "shuffle":          # np.random.seed(s); shuffle; first floor(frac * n) validate
        np.random.seed(cfg["split_seed"])
        idx = np.arange(n)
        np.random.shuffle(idx)
        k = int(np.floor(cfg["val_frac"] * n))
        return idx[k:], idx[:k]
    raise ValueError(cfg["split"])


def train(recipe, out_dir, annual, labels, seasonal=None, base=None, teacher_sls=None, device=None,
          log=print, **overrides):
    """Train a model and write out_dir/{model.pth, history.csv, split.npz, result.json}.

    annual / seasonal / labels: lists of .npy chip-stack files (seasonal only for T / TE inputs);
    base: checkpoint to start from (kg-unet1, kg-unet2, finetune) and ALS teacher of kg-unet2;
    teacher_sls: GEDI teacher checkpoint of kg-unet2.
    """
    cfg = resolve(recipe, **overrides)
    out = Path(out_dir)
    if (out / "result.json").exists():
        raise FileExistsError(f"refusing to overwrite finished run {out}")
    out.mkdir(parents=True, exist_ok=True)

    random.seed(cfg["seed"]); np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    A, Y, S = Stack(annual), Stack(labels), open_stack(seasonal)
    assert len(A) == len(Y) and (S is None or len(S) == len(A)), "stacks differ in chip count"
    t0 = time.time()
    n = len(A)
    idx_tr, idx_va = split_chips(n, cfg)
    bpe = len(idx_tr) // cfg["batch_size"]
    assert bpe >= 1, f"fewer than {cfg['batch_size']} training chips"
    nw = cfg["num_workers"] if cfg["num_workers"] is not None else (6 if bpe >= 4 else 0)

    if cfg["init"] == "base":
        if base is None:
            raise ValueError(f"recipe {recipe} starts from a checkpoint: pass base=")
        model, mcfg = load_model(base, device=device, zero_restore=cfg["zero_restore"])
        cfg["input"] = mcfg["input"]                  # the checkpoint defines the input layout
    else:
        zr = channels.INPUTS[cfg["input"]]["zero_restore"] if cfg["zero_restore"] is None else cfg["zero_restore"]
        mcfg = model_config(cfg["input"], zero_restore=zr)
        mean, std = stats.for_input(cfg["input"])
        model = build_model(mcfg, mean, std).to(device)
    if channels.needs_seasonal(cfg["input"]) and S is None:
        raise ValueError(f"input {cfg['input']} needs seasonal stacks")
    if cfg["trainable"]:
        for p in model.parameters():
            p.requires_grad = False
        for blk in cfg["trainable"]:
            for p in getattr(model, blk).parameters():
                p.requires_grad = True

    log(f"[{time.time()-t0:6.1f}s] recipe={recipe} input={cfg['input']} chips={n} train={len(idx_tr)} "
        f"val={len(idx_va)} batches/epoch={bpe} epochs={cfg['epochs']} select={cfg['select']}")

    g = torch.Generator(); g.manual_seed(cfg["seed"])
    zt = cfg["zero_to_nodata"]
    train_loader = DataLoader(ChipDataset(cfg["input"], A, S, Y, idx_tr, zt), batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=nw, drop_last=True, pin_memory=True, generator=g)
    val_loader = DataLoader(ChipDataset(cfg["input"], A, S, Y, idx_va, zt), batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=cfg["val_workers"], drop_last=False, pin_memory=True)

    teachers = None
    if cfg["kg2"]:
        if base is None or teacher_sls is None:
            raise ValueError("kg-unet2 needs base= (ALS teacher) and teacher_sls=")
        teachers = (load_model(base, device=device)[0], load_model(teacher_sls, device=device)[0])
        for t in teachers:
            t.eval()

    net = model
    ngpu = torch.cuda.device_count() if cfg["gpus"] == -1 else cfg["gpus"]
    if ngpu > 1:
        net = torch.nn.DataParallel(model)
    params = model.parameters() if not cfg["trainable"] else filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg["milestones"], gamma=cfg["gamma"])
    kg = cfg["kg2"]
    skipped = {"train": 0, "val": 0}

    def step(images, labels, epoch):
        if not bool((labels != -999).any()):
            skipped["train"] += 1
            return None
        # the forward pass normalises its input in place, so every model gets its own copy
        pred = net(images.to(device, copy=True))
        lab = labels.to(device)
        if kg:
            teach = epoch > kg["gate"]
            if not (teach and kg["mode"] == "replace"):
                loss = kg["w_label"] * charbonnier(pred, lab)
            if teach:
                # The teachers are fixed targets: without autograd their activations are not kept for backward.
                # Same weights bit for bit as with autograd on (checked on an RTX 3090 Ti), ~6x faster on
                # 24 GB GPUs, where the kept activations spilled into shared memory.
                with torch.no_grad():
                    t_als = teachers[0](images.to(device, copy=True))
                    t_sls = teachers[1](images.to(device, copy=True))
                e = kg["w_als"] * edge_aware_gradient_loss(pred, t_als)
                p = kg["w_sls"] * avgpool_loss(pred, t_sls, kg["pool"])
                # add: label term + both teacher terms; replace: teacher terms only after the gate
                loss = e + p if kg["mode"] == "replace" else loss + e + p
        else:
            loss = charbonnier(pred, lab)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        return loss.item()

    def validate():
        net.eval()
        va, nv = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                if not bool((labels != -999).any()):
                    skipped["val"] += 1
                    continue
                va += charbonnier(net(images.to(device, copy=True)), labels.to(device)).item(); nv += 1
        return va / max(nv, 1)

    hist, best, best_epoch, since = [], np.inf, None, 0
    for epoch in range(cfg["epochs"]):
        net.train()
        tr, nb = 0.0, 0
        for images, labels in train_loader:
            loss = step(images, labels, epoch)
            if loss is not None:
                tr += loss; nb += 1
        tr /= max(nb, 1)
        scheduler.step()
        va = validate()
        hist.append((epoch, tr, va))
        log(f"epoch {epoch}\ttrain {tr:.4f}\tval {va:.4f}\t[{time.time()-t0:.0f}s]")
        if not np.isfinite(tr):
            raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
        if va < best:
            best, best_epoch, since = va, epoch, 0
            if cfg["select"] == "best":
                save_checkpoint(out / "model.pth", model, mcfg, epoch=epoch, val_loss=va, recipe=cfg)
        else:
            since += 1
            if cfg["patience"] and since >= cfg["patience"]:
                log(f"early stop at epoch {epoch} (best {best_epoch})")
                break
    if cfg["select"] == "last":
        save_checkpoint(out / "model.pth", model, mcfg, epoch=hist[-1][0], val_loss=hist[-1][2], recipe=cfg)

    with open(out / "history.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["epoch", "train_loss", "val_loss"]); w.writerows(hist)
    np.savez(out / "split.npz", train=idx_tr, val=idx_va)
    res = dict(recipe=cfg, model=mcfg, n_chips=n, n_train=len(idx_tr), n_val=len(idx_va), batches_per_epoch=bpe,
               num_workers=nw, gpus=max(ngpu, 1), epochs_run=len(hist), best_epoch=best_epoch,
               best_val_loss=float(best), final_train_loss=hist[-1][1], final_val_loss=hist[-1][2],
               skipped_empty_batches=skipped, seconds=round(time.time() - t0, 1), torch=torch.__version__,
               device=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
               base=str(base) if base else None, teacher_sls=str(teacher_sls) if teacher_sls else None,
               annual=[str(p) for p in A.files], labels=[str(p) for p in Y.files],
               seasonal=[str(p) for p in S.files] if S else None)
    if n <= 5000:
        res["labelled_px"] = labelled_pixels(Y, zt)
    json.dump(res, open(out / "result.json", "w"), indent=2)
    return out / "model.pth", res
