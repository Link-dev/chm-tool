"""Report of a project: accuracy against GEDI and ALS, study-area statistics and quick-look maps.

Three sections; each is skipped with a note when its inputs are missing:

  gedi_val    GEDI labels of the validation chips of the UNet recipe (models/unet-sls/split.npz; without a
              UNet-SLS run on the current stack, the recipe's split of the training stack) against every trained
              model that matches the current training stack (stack.model_is_current; the others are not scored,
              with a note). A UNet's row says whether any of these chips was one of its own training chips (its
              split.npz); with the paper's recipes none is. RF-SLS is fitted on a random 80 % of the labelled
              pixels of all chips, so for RF-SLS they are in-sample. Models whose input needs the seasonal
              composites (T / TE) are given the chips of the seasonal training stack as well.
  study_area  mean / p10 / p50 / p90 of every map, benchmark (and GEDI / ALS) inside the study area, and the
              agreement of each layer with the GEDI labels inside it. The local models were trained on these
              labels and HRCH / GFCH were calibrated with GEDI data of the same period, so their agreement is not
              an independent accuracy.
  als         with mosaic/ALS.tif: accuracy against ALS in the paper's convention for the international sites,
              inside the study area: 256 px chips of the mosaic grid, kept when >= 1 % of their cells have
              ALS > 1 m (ALS NaN -> -999, as for the paper's test chips), the same chips for every layer;
              canopy_height.metrics.evaluate_chips (ALS cells > 1 m, 80 m cap, missing predictions count as 0):
              chip-median RMSE and r2, pooled ME.

GEDI labels count when 0 < rh95 <= 80 m (the 80 m cap of the ALS convention); labels above 80 m are left out and
counted. A map older than its model file or the study-area mosaic is left out (run the predict stage).
bias / ME = mean(prediction - reference), r2 = squared Pearson correlation.

Rasters are read in strips of 256 rows. Counts, means and GEDI agreements use every pixel; p10 / p50 / p90 use
the pixels in every k-th row and column of the mosaic, k = ceil(sqrt(study-area pixels / SAMPLE_PX)) (every pixel
up to SAMPLE_PX, else about SAMPLE_PX of them); the ALS chips are evaluated exactly.

Output: report/metrics.csv (COLUMNS), report/summary.json and one quick-look PNG per map, benchmark and ALS (those
of earlier reports are removed first), all on one colour scale (0 .. max(40 m, p99 of all layers), viridis, NaN
transparent).
"""
import csv
import json
import math
import textwrap
import time
import warnings
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .project import BENCHMARK_SOURCES, BENCHMARKS, MODEL_NAMES, replace_retry

COLUMNS = ["section", "layer", "n", "rmse", "bias", "r2", "mean", "p10", "p50", "p90", "note"]
SECTIONS = ["gedi_val", "study_area", "als"]
CHIP = 256                     # chip size of the ALS evaluation (paper convention) and strip height
MIN_ALS_SHARE = 0.01           # ALS chips kept when >= 1 % of their cells have ALS > 1 m (paper test tiles)
GEDI_MAX = 80.0                # GEDI labels above are left out (80 m cap of the ALS convention)
SAMPLE_PX = 5_000_000          # percentiles from at most about this many study-area pixels
COLOUR_PX = 200_000            # values per layer for the colour scale
VMAX_MIN = 40.0                # colour scale reaches at least 40 m
QUICKLOOK_PX = 2000            # quick-looks are subsampled to at most this many pixels per side
NOTE_MODEL_GEDI = "in-sample: the model was trained on these GEDI labels"
GEDI_CALIBRATED = ("HRCH", "GFCH")      # Lang et al. 2023 (GEDI 2019-2020), Potapov et al. 2021 (GEDI 2019)
NOTE_BENCH_GEDI = "calibrated with GEDI data of the same period, so its agreement with GEDI is not independent"


# ---------------------------------------------------------------------------------------------- metrics
class Pairs:
    """n, RMSE, bias (pred - ref) and r2 (squared Pearson) of paired values added block by block; exact (centred
    sums merged per block)."""

    def __init__(self):
        self.n, self.sd, self.sdd = 0, 0.0, 0.0
        self.mp = self.mt = self.m2p = self.m2t = self.cpt = 0.0

    def add(self, pred, ref):
        p = np.asarray(pred, dtype=np.float64).ravel()
        t = np.asarray(ref, dtype=np.float64).ravel()
        k = p.size
        if k == 0:
            return self
        d = p - t
        self.sd += float(d.sum())
        self.sdd += float(np.dot(d, d))
        mp, mt = float(p.mean()), float(t.mean())
        dp, dt = p - mp, t - mt
        n = self.n + k
        ep, et, f = mp - self.mp, mt - self.mt, self.n * k / n
        self.m2p += float(np.dot(dp, dp)) + ep * ep * f
        self.m2t += float(np.dot(dt, dt)) + et * et * f
        self.cpt += float(np.dot(dp, dt)) + ep * et * f
        self.mp += ep * k / n
        self.mt += et * k / n
        self.n = n
        return self

    def result(self):
        nan = float("nan")
        if self.n == 0:
            return dict(n=0, rmse=nan, bias=nan, r2=nan)
        # a variance at rounding level (constant values) gives no r2
        ok = all(m2 > 1e-12 * self.n * (1.0 + m * m) for m2, m in ((self.m2p, self.mp), (self.m2t, self.mt)))
        return dict(n=int(self.n), rmse=math.sqrt(self.sdd / self.n), bias=self.sd / self.n,
                    r2=min(1.0, self.cpt ** 2 / (self.m2p * self.m2t)) if ok and self.n > 1 else nan)


def agreement(pred, ref):
    """n, RMSE, bias (pred - ref) and r2 (squared Pearson) of paired 1-D arrays."""
    return Pairs().add(pred, ref).result()


class Values:
    """Count and mean of every value added, and percentiles of the subsample added with them."""

    def __init__(self):
        self.n, self.s, self.sub = 0, 0.0, []

    def add(self, v, sub):
        self.n += int(v.size)
        self.s += float(v.sum(dtype=np.float64))
        self.sub.append(np.asarray(sub, dtype=np.float32))

    def result(self):
        """(dict n_px, mean, p10, p50, p90; the subsample as float32)."""
        v = np.concatenate(self.sub) if self.sub else np.empty(0, np.float32)
        self.sub = []
        nan = float("nan")
        d = dict(n_px=self.n, mean=self.s / self.n if self.n else nan, p10=nan, p50=nan, p90=nan)
        if v.size:
            d.update(zip(("p10", "p50", "p90"), map(float, np.percentile(v.astype(np.float64), [10, 50, 90]))))
        return d, v


def _row(section, layer, note="", **values):
    r = {k: values.get(k) for k in COLUMNS}
    r.update(section=section, layer=layer, note=note)
    return r


def _clean(x):
    """NaN -> None (strict JSON), numpy scalars -> Python."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "" if not math.isfinite(float(v)) else f"{float(v):.4f}"
    return str(v)


# ---------------------------------------------------------------------------------------------- inputs
def _grid(src):
    return (src.width, src.height, src.transform, src.crs)


def _read(src, window):
    """Band 1 of a window as float32 (nodata and values <= -999 -> NaN)."""
    a = src.read(1, window=window).astype(np.float32)
    nd = src.nodata
    if nd is not None and not np.isnan(nd):
        a[a == nd] = np.nan
    a[a <= -999] = np.nan
    return a


class _Chips:
    """CHIP px chips of a raster at (row, col) offsets, read on access: NaN outside the study area, missing values
    -> `fill` if given."""

    def __init__(self, src, mask, offsets, fill=None):
        self.src, self.mask, self.offsets, self.fill = src, mask, offsets, fill

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, i):
        from rasterio.windows import Window
        r, c = self.offsets[i]
        w = Window(c, r, min(CHIP, self.src.width - c), min(CHIP, self.src.height - r))
        a = _read(self.src, w)
        a[self.mask.read(1, window=w) != 1] = np.nan
        return a if self.fill is None else np.where(np.isfinite(a), a, self.fill).astype(np.float32)


def _trained(project):
    from .workflow import trained_models
    return trained_models(project)


def _newer(path, *others):
    """path is at least as new as every existing file in others (the rule of the predict stage)."""
    t = Path(path).stat().st_mtime
    return all(t >= Path(o).stat().st_mtime for o in others if Path(o).exists())


def _stale_note(current, m):
    ok, why = current.get(m, (True, ""))
    return "" if ok else f"the model does not match the current training stack ({why})"


def _layers(project, trained, notes, current):
    """(name, file, kind, extra note) of every up-to-date map and every benchmark mosaic present."""
    out = []
    mosaics = [project.mosaic("annual"), project.mosaic("aoi_mask")]
    for m in trained:
        name, f = MODEL_NAMES[m], project.map_file(m)
        if not f.exists():
            notes.append(f"{name}: no map ({f.name}): run the predict stage")
        elif not _newer(f, project.model_file(m), *mosaics):
            notes.append(f"{name}: {f.name} is older than its model or the study-area mosaic, left out: run the "
                         f"predict stage")
        else:
            out.append((name, f, "model", _stale_note(current, m)))
    for b in project["benchmarks"]:
        f = project.mosaic(b)
        if f.exists():
            out.append((b, f, "benchmark", ""))
        else:
            notes.append(f"{b}: mosaic/{b}.tif missing, left out")
    return out


def _join(*parts):
    return "; ".join(p for p in parts if p)


# ---------------------------------------------------------------------------------------------- section 1
def _unet_recipe(project):
    from ..train import resolve
    over = (project["train_overrides"] or {}).get("unet-sls") or {}
    try:
        return resolve("unet-sls", **over)
    except ValueError:
        return resolve("unet-sls")


class SeasonalStackMissing(FileNotFoundError):
    """A model whose input needs the seasonal composites, without a seasonal training stack."""


def _predict_chips(project, model, stack, idx, chunk=16, seasonal=None):
    """Predictions of chips `idx` of the training stack, chunk by chunk ([k, H, W] float32). `seasonal`: the
    seasonal training stack (same chips), passed on when the model's input needs it (channels.needs_seasonal of
    its config / metadata 'input'; SeasonalStackMissing without it)."""
    from .. import channels
    if model == "rf-sls":
        from ..rf import load_rf, predict_rf_chips
        rf, meta = load_rf(project.model_file(model))
        inp = meta["input"]
        f = lambda x, s: predict_rf_chips(rf, meta, x, s)                            # noqa: E731
    else:
        from ..models import load_model
        from ..predict import predict_stack
        from .workflow import torch_device
        dev = torch_device(project)
        net, cfg = load_model(project.model_file(model), device=dev)
        inp = cfg["input"]
        f = lambda x, s: predict_stack(net, cfg, x, s, device=dev)                  # noqa: E731
    need = channels.needs_seasonal(inp)
    if need and seasonal is None:
        raise SeasonalStackMissing(f"input {inp} needs the seasonal training stack (stacks/train_seasonal_part*.npy): "
                                   f"run the stack stage")
    for k in range(0, len(idx), chunk):
        ii = [int(i) for i in idx[k:k + chunk]]
        yield f(np.stack([stack[i] for i in ii]), np.stack([seasonal[i] for i in ii]) if need else None)


def _split(f):
    """(train, val) chip indices of a split.npz, None if it is missing or unreadable."""
    try:
        with np.load(f) as z:
            return np.asarray(z["train"], np.int64), np.asarray(z["val"], np.int64)
    except (OSError, KeyError, ValueError):
        return None


def _held_out(split, val):
    """Row note of a UNet and the number of scored chips that were its own training chips."""
    if split is None:
        return "held-out status unknown (no split.npz)", None
    k = int(np.isin(val, split[0]).sum())
    if k == 0:
        return "held-out chips", 0
    if k == len(val):
        return "in-sample: every chip was a training chip of this model", k
    return f"partly in-sample: {k} of {len(val)} chips were training chips of this model", k


def gedi_validation(project, trained, rows, notes, current=None):
    """Section 1: every trained model that matches the current stack against the GEDI labels (0 < rh95 <= 80 m)
    of the UNet validation chips."""
    annual, labels = project.stack_files("x"), project.stack_files("GEDI")
    if not annual or len(labels) != len(annual):
        notes.append("gedi_val: no training stack, skipped")
        return None
    if not trained:
        notes.append("gedi_val: no trained model, skipped")
        return None
    current = current or {}
    from ..data import Stack
    A, Y = Stack(annual), Stack(labels)
    n = len(A)
    seasonal = project.stack_files("seasonal")
    S = Stack(seasonal) if seasonal else None             # inputs T / TE: passed to the models that need it
    if S is not None and len(S) != n:
        notes.append(f"gedi_val: the seasonal training stack has {len(S)} chips, the training stack {n}: not used")
        S = None
    models, splits, skipped = [], {}, {}
    for m in trained:
        name = MODEL_NAMES[m]
        ok, why = current.get(m, (True, ""))
        sp = None if m == "rf-sls" else _split(project.model_dir(m) / "split.npz")
        if ok and sp is not None and max(int(sp[0].max(initial=-1)), int(sp[1].max(initial=-1))) >= n:
            ok, why = False, f"its split.npz refers to chips beyond the {n} of the current stack"
        if not ok:
            skipped[name] = why
            notes.append(f"gedi_val: {name} not scored: the model does not match the current training stack ({why}); "
                         f"retrain it (train stage)")
            continue
        models.append(m)
        splits[m] = sp
    res = dict(split=None, n_chips=0, n_labels=0, labels=f"0 < rh95 <= {GEDI_MAX:g} m", models={}, skipped=skipped)
    if not models:
        notes.append("gedi_val: no trained model matches the current training stack, skipped")
        return res
    if splits.get("unet-sls") is not None:
        val, source = splits["unet-sls"][1], "models/unet-sls/split.npz"
    else:
        try:
            from ..train import split_chips
            val, source = split_chips(n, _unet_recipe(project))[1], "split of the unet-sls recipe"
        except ImportError as e:
            notes.append(f"gedi_val: skipped ({e})")
            return res
    val = np.sort(np.asarray(val, dtype=np.int64))
    masks, ref, n_over = [], [], 0
    for i in val:
        y = np.asarray(Y[int(i)])
        n_over += int((y > GEDI_MAX).sum())
        mk = (y > 0) & (y <= GEDI_MAX)
        masks.append(mk)
        ref.append(y[mk])
    ref = np.concatenate(ref) if ref else np.empty(0, np.float32)
    res.update(split=source, n_chips=int(len(val)), n_labels=int(ref.size), n_labels_over_80=n_over)
    if n_over:
        notes.append(f"gedi_val: {n_over:,} GEDI labels > {GEDI_MAX:g} m of the validation chips left out (the "
                     f"80 m cap of the ALS convention)")
    if ref.size == 0:
        notes.append("gedi_val: the validation chips hold no GEDI label, skipped")
        return res
    for m in models:
        name = MODEL_NAMES[m]
        try:
            preds = [p[k] for p, k in zip(_iter_chips(_predict_chips(project, m, A, val, seasonal=S)), masks)]
        except (ImportError, SeasonalStackMissing) as e:
            notes.append(f"gedi_val: {name} skipped ({e})")
            continue
        p = np.concatenate(preds)
        ok = np.isfinite(p)
        a = agreement(p[ok], ref[ok])
        if m == "rf-sls":
            note, k = "in-sample: RF-SLS is fitted on 80 % of the labelled pixels of all chips", None
        else:
            note, k = _held_out(splits[m], val)
        note = _join(note, f"GEDI labels <= {GEDI_MAX:g} m")
        rows.append(_row("gedi_val", name, note, **a))
        res["models"][name] = dict(a, note=note, n_own_training_chips=k)
        project.log(f"report: GEDI validation {name}: n {a['n']:,}, RMSE {a['rmse']:.2f} m, bias {a['bias']:+.2f} m, "
                    f"r2 {a['r2']:.3f}")
    return res


def _iter_chips(batches):
    for b in batches:
        yield from b


# ---------------------------------------------------------------------------------------------- sections 2, 3
def study_area_and_als(project, trained, rows, notes, current=None):
    """Sections 2 and 3, reading every raster in strips of CHIP rows; returns (study_area, als, quick-look layers,
    colour-scale samples)."""
    import rasterio
    from rasterio.windows import Window
    from ..metrics import evaluate_chips
    mask_f = project.mosaic("aoi_mask")
    if not mask_f.exists():
        notes.append("study_area / als: mosaic/aoi_mask.tif missing, skipped")
        return None, None, [], []
    current = current or {}
    with ExitStack() as es:
        mask = es.enter_context(rasterio.open(mask_f))
        grid, H, W = _grid(mask), mask.height, mask.width
        strips = [Window(0, r0, W, min(CHIP, H - r0)) for r0 in range(0, H, CHIP)]
        n_in = sum(int((mask.read(1, window=w) == 1).sum()) for w in strips)
        k = max(1, math.ceil(math.sqrt(n_in / SAMPLE_PX)))
        step = max(1, math.ceil(max(H, W) / QUICKLOOK_PX))
        sa = dict(area_km2=float(n_in * abs(mask.res[0] * mask.res[1]) / 1e6), px=n_in,
                  percentiles="every pixel" if k == 1 else f"rows and columns of the mosaic whose index is a multiple "
                                                            f"of {k}", layers={})

        def open_on_grid(f, name, what):
            s = es.enter_context(rasterio.open(f))
            if _grid(s) == grid:
                return s
            notes.append(f"{name}: {f.name} is not on the mosaic grid, {what}")
            return None

        gedi = None
        if project.mosaic("GEDI").exists():
            gedi = open_on_grid(project.mosaic("GEDI"), "study_area", "no GEDI agreement")
        else:
            notes.append("study_area: mosaic/GEDI.tif missing, no GEDI agreement")
        als = None
        if project.mosaic("ALS").exists():
            als = open_on_grid(project.mosaic("ALS"), "als", "skipped")
        elif project["als"]:
            notes.append("als: ALS rasters are set but mosaic/ALS.tif is missing (stack stage)")
        layers = []
        for name, f, kind, extra in _layers(project, trained, notes, current):
            s = open_on_grid(f, name, "left out")
            if s is not None:
                layers.append((name, s, kind, extra))
        if als is not None:
            layers.append(("ALS", als, "reference", ""))

        # one pass over the strips: counts, sums, subsamples, GEDI pairs, quick-looks, ALS chip selection
        acc = {name: (Values(), Pairs(), []) for name, _, _, _ in layers}
        gv, n_over, kept, dropped = Values(), 0, [], 0
        with np.errstate(invalid="ignore"):
            for w in strips:
                r0 = int(w.row_off)
                inside = mask.read(1, window=w) == 1
                rs, rq = (-r0) % k, (-r0) % step
                lab = g = None
                if gedi is not None:
                    g = _read(gedi, w)
                    g[~inside] = np.nan
                    n_over += int((g > GEDI_MAX).sum())
                    g[~((g > 0) & (g <= GEDI_MAX))] = np.nan
                    lab = np.isfinite(g)
                    gs = g[rs::k, ::k]
                    gv.add(g[lab], gs[np.isfinite(gs)])
                for name, s, kind, _ in layers:
                    vals, pairs, ql = acc[name]
                    a = _read(s, w)
                    a[~inside] = np.nan
                    fin = np.isfinite(a)
                    sub = a[rs::k, ::k]
                    vals.add(a[fin], sub[np.isfinite(sub)])
                    if lab is not None:
                        sel = lab & fin
                        pairs.add(a[sel], g[sel])
                    ql.append(a[rq::step, ::step].copy())
                    if kind == "reference":
                        for c0 in range(0, W, CHIP):
                            n1 = int((a[:, c0:c0 + CHIP] > 1).sum())        # = (NaN -> -999) > 1
                            if n1 / CHIP ** 2 >= MIN_ALS_SHARE:
                                kept.append((r0, c0))
                            elif n1:
                                dropped += 1

        if gedi is not None:
            d, _ = gv.result()
            note = f"GEDI rh95 labels, 0 < rh95 <= {GEDI_MAX:g} m ({n_over:,} labels > {GEDI_MAX:g} m left out)"
            rows.append(_row("study_area", "GEDI", note, n=d["n_px"], **d))
            sa["layers"]["GEDI"] = dict(d, kind="labels", n_labels_over_80=n_over)
            if n_over:
                notes.append(f"study_area: {n_over:,} GEDI labels > {GEDI_MAX:g} m left out (the 80 m cap of the ALS "
                             f"convention)")
        quick, samples = [], []
        for name, s, kind, extra in layers:
            vals, pairs, ql = acc.pop(name)
            d, v = vals.result()
            ag = pairs.result()
            if kind == "model":
                note = _join(NOTE_MODEL_GEDI, extra)
            elif kind == "benchmark":
                note = _join(BENCHMARK_SOURCES.get(name, "benchmark"), NOTE_BENCH_GEDI if name in GEDI_CALIBRATED
                             else "")
            else:
                note = "ALS reference"
            rows.append(_row("study_area", name, note, **d, **ag))
            sa["layers"][name] = dict(d, gedi=ag, kind=kind, note=note)
            samples.append(v[::max(1, v.size // COLOUR_PX)].copy())
            quick.append((name, np.concatenate(ql), kind))

        al = None
        if als is not None:
            al = dict(convention=f"evaluate_chips on the {CHIP} px chips with ALS > 1 m in >= {MIN_ALS_SHARE:.0%} of "
                                 f"their cells (ALS > 1 m, 80 m cap, inside the study area)",
                      chips_kept=len(kept), chips_dropped=dropped, layers={})
            if dropped:
                notes.append(f"als: {dropped} chips with ALS > 1 m in fewer than {MIN_ALS_SHARE:.0%} of their cells "
                             f"left out (the paper's test-tile rule), {len(kept)} kept")
            if not kept:
                notes.append(f"als: no chip with ALS > 1 m in at least {MIN_ALS_SHARE:.0%} of its cells inside the "
                             f"study area, skipped")
            ref = _Chips(als, mask, kept, fill=-999.0)
            for name, s, kind, extra in layers if kept else []:
                if kind == "reference":
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    ev = evaluate_chips(_Chips(s, mask, kept), ref)
                ev.pop("per_chip")
                cm, pooled = ev["chip_median"], ev["pooled"]
                if pooled is None:
                    notes.append(f"als: no ALS cell > 1 m inside the study area for {name}")
                else:
                    note = _join(f"{cm['n_chips']} chips of {CHIP} px with ALS > 1 m in >= {MIN_ALS_SHARE:.0%} of "
                                 f"their cells ({dropped} with less left out); chip-median RMSE and r2, pooled ME",
                                 extra)
                    rows.append(_row("als", name, note, n=pooled["n_px"], rmse=cm["rmse_median"], bias=pooled["me"],
                                     r2=cm["r2_median"]))
                    project.log(f"report: ALS {name}: chip-median RMSE {cm['rmse_median']:.2f} m, r2 "
                                f"{cm['r2_median']:.3f}, ME {pooled['me']:+.2f} m ({cm['n_chips']} chips)")
                al["layers"][name] = dict(ev, kind=kind)
    return sa, al, quick, samples


# ---------------------------------------------------------------------------------------------- quick-looks
def remove_quicklooks(out_dir, notes=None):
    """Delete the quick-looks of earlier reports: the PNGs listed in summary.json and those named after a layer."""
    names = {f"{n}.png" for n in [*MODEL_NAMES.values(), *BENCHMARKS, "ALS", "GEDI"]}
    try:
        with open(out_dir / "summary.json", encoding="utf-8") as fh:
            names.update(Path(str(x)).name for x in json.load(fh).get("quicklooks") or [])
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    for p in [out_dir / n for n in names if n.lower().endswith(".png")] + list(out_dir.glob("*.png.tmp")):
        try:
            p.unlink(missing_ok=True)
        except OSError as e:
            if notes is not None:
                notes.append(f"report: old quick-look {p.name} not removed ({e})")


def quicklooks(out_dir, quick, samples):
    """One PNG per layer on a common colour scale; returns (colour scale, file names)."""
    if not quick:
        return None, []
    from matplotlib import colormaps
    from matplotlib.figure import Figure
    v = np.concatenate(samples) if samples else np.empty(0)
    vmax = float(max(VMAX_MIN, np.percentile(v, 99) if v.size else 0.0))
    cmap = colormaps["viridis"].with_extremes(bad=(0, 0, 0, 0))
    files = []
    for name, a, kind in quick:
        h, w = a.shape
        fig = Figure(figsize=(7.5, max(3.0, min(9.0, 6.0 * h / w + 1.0))), layout="constrained")
        ax = fig.add_subplot()
        im = ax.imshow(np.ma.masked_invalid(a), cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        sub = {"model": "local model (this project)", "benchmark": BENCHMARK_SOURCES.get(name, ""),
               "reference": "ALS reference"}[kind]
        ax.set_title(f"{name}\n" + textwrap.fill(sub, 64), fontsize=10)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, shrink=0.8, label="canopy height (m)")
        fn = f"{name}.png"
        tmp = out_dir / f"{fn}.tmp"
        fig.savefig(tmp, format="png", dpi=110, transparent=True)
        replace_retry(tmp, out_dir / fn)
        files.append(fn)
    return dict(vmin=0.0, vmax=vmax, cmap="viridis", units="m"), files


# ---------------------------------------------------------------------------------------------- report
def report(project):
    """Write report/metrics.csv, report/summary.json and the quick-looks; returns the summary."""
    from . import stack
    t0 = time.time()
    out = project.path("report")
    out.mkdir(parents=True, exist_ok=True)
    rows, notes = [], []
    trained = _trained(project)
    if not trained:
        notes.append("no trained model: only benchmarks are reported")
    if project.stack_files("x"):
        current = {m: tuple(stack.model_is_current(project, m)) for m in trained}
    else:                                   # no stack to compare with (e.g. deleted after training)
        current = {m: (True, "") for m in trained}
    stale ={MODEL_NAMES[m]: why for m, (ok, why) in current.items() if not ok}
    summary = dict(project=project["name"] or project.root.name, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                   year=project["year"], input=project["input"], models=[MODEL_NAMES[m] for m in trained],
                   benchmarks=list(project["benchmarks"]), stale_models=stale, notes=notes)
    summary["gedi_validation"] = gedi_validation(project, trained, rows, notes, current)
    sa, al, quick, samples = study_area_and_als(project, trained, rows, notes, current)
    summary["study_area"], summary["als"] = sa, al
    remove_quicklooks(out, notes)
    summary["colour_scale"], summary["quicklooks"] = quicklooks(out, quick, samples)
    summary["n_rows"] = len(rows)
    summary["seconds"] = round(time.time() - t0, 1)

    tmp = out / "metrics.csv.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for r in sorted(rows, key=lambda r: SECTIONS.index(r["section"])):
            w.writerow([_fmt(r[k]) for k in COLUMNS])
    replace_retry(tmp, out / "metrics.csv")
    tmp = out / "summary.json.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_clean(summary), fh, indent=1)
    replace_retry(tmp, out / "summary.json")
    for n in notes:
        project.log(f"report note: {n}")
    return summary
