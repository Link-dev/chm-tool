"""Accuracy metrics. r^2 is the squared Pearson correlation; ME / bias = mean(prediction - reference)."""
import numpy as np


class Stack:
    """Consecutive multi-part chip arrays viewed as one, read chip by chip as float64."""

    def __init__(self, files):
        self.a = [np.load(f, mmap_mode="r") for f in files]
        self.n = sum(x.shape[0] for x in self.a)

    def __getitem__(self, i):
        for x in self.a:
            if i < x.shape[0]:
                return np.asarray(x[i], dtype=np.float64)
            i -= x.shape[0]
        raise IndexError(i)


def new_acc():
    return dict(per_chip=[], n=0, se=0.0, ae=0.0, sx=0.0, sy=0.0, sxx=0.0, syy=0.0, sxy=0.0)


def add(a, x, y):
    """Accumulate reference x and prediction y for pooled metrics."""
    d = y - x
    a["n"] += x.size; a["se"] += float((d * d).sum()); a["ae"] += float(d.sum())
    a["sx"] += float(x.sum()); a["sy"] += float(y.sum())
    a["sxx"] += float((x * x).sum()); a["syy"] += float((y * y).sum()); a["sxy"] += float((x * y).sum())


def pooled(a):
    n = a["n"]
    if n == 0:
        return None
    cov = a["sxy"] / n - (a["sx"] / n) * (a["sy"] / n)
    vx = a["sxx"] / n - (a["sx"] / n) ** 2
    vy = a["syy"] / n - (a["sy"] / n) ** 2
    return {"n_px": n, "me": a["ae"] / n, "rmse": float(np.sqrt(a["se"] / n)),
            "r2": float(cov * cov / (vx * vy)) if vx > 0 and vy > 0 else float("nan")}


def summarize(rows):
    """Medians over chips (chips without evaluation pixels are None and skipped)."""
    ok = [r for r in rows if r]
    g = lambda k: np.array([r[k] for r in ok], dtype=float)
    return {"n_chips": len(ok),
            "rmse_median": float(np.median(g("rmse"))), "rmse_mean": float(np.mean(g("rmse"))),
            "mae_median": float(np.median(g("mae"))), "r2_median": float(np.nanmedian(g("r2"))),
            "me_median": float(np.median(g("me")))}


def metrics(p, t):
    """Pooled metrics of prediction p against reference t (1-D arrays)."""
    d = p - t
    r = np.corrcoef(p, t)[0, 1] if p.std() > 0 else np.nan
    return dict(n=int(t.size), r2=float(r * r), rmse=float(np.sqrt(np.mean(d * d))),
                bias=float(d.mean()), mae=float(np.abs(d).mean()))


def evaluate_chips(pred, ref, mask_ref=None, cap80=True):
    """Accuracy of predicted chips against reference chips (paper convention for the international sites).

    Per chip, reference values of -999 / < 0 (and > 80 m with cap80) are set to 0 and cells with
    reference > 1 m are evaluated; `mask_ref` (e.g. the p90 reference when scoring against p95) selects
    the cells instead, and cells where `ref` is missing are dropped. Returns medians over chips of RMSE,
    MAE, r^2 (squared Pearson) and ME, plus pooled metrics over all evaluated cells.
    """
    acc, rows = new_acc(), []
    for i in range(len(ref)):
        lab = np.asarray(mask_ref[i] if mask_ref is not None else ref[i], dtype=np.float64).copy()
        lab[lab == -999] = 0; lab[lab < 0] = 0
        if cap80:
            lab[lab > 80] = 0
        lab[lab < 1] = 0
        msk = lab > 1
        if mask_ref is not None:
            val = np.asarray(ref[i], dtype=np.float64)
            msk &= np.isfinite(val) & (val != -999)
            lab = val
        if not msk.any():
            rows.append(None)
            continue
        x = lab[msk]
        y = np.nan_to_num(np.asarray(pred[i], dtype=np.float64))[msk]
        d = y - x
        r = np.corrcoef(x, y)[0, 1] if (x.std() > 0 and y.std() > 0) else np.nan
        rows.append(dict(chip=i, n=int(msk.sum()), rmse=float(np.sqrt(np.mean(d * d))), mae=float(np.mean(np.abs(d))),
                         me=float(np.mean(d)), r2=float(r * r) if np.isfinite(r) else float("nan")))
        add(acc, x, y)
    return dict(chip_median=summarize(rows), pooled=pooled(acc), per_chip=rows)
