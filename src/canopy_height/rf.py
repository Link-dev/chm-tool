"""RF-SLS: random forest trained on pixels with a local GEDI label.

Training rows are the cells with a label > 0 in the label stacks. A random 80 % of the rows fit
RandomForestRegressor(600 trees, max_depth 50, min_samples_leaf 4, max_features sqrt, random_state 42);
the other 20 % give a pixel-level hold-out score (for reference only). The forest is deterministic
for a fixed random_state.

Feature encoding of the embedding channels 0-63:
  paper  raw value minus 10000 computed in uint16, so values below 10000 wrap modulo 65536. This is how
         the paper's RF-SLS models were trained and evaluated; use it to reproduce the paper.
  plain  raw values (no subtraction, no wrap-around). Recommended for new models.
Only NumPy and scikit-learn are needed (no PyTorch).
"""
import json
import time
from pathlib import Path

import numpy as np

from . import channels
from .data import Stack

RF_PARAMS = dict(n_estimators=600, max_depth=50, min_samples_leaf=4, max_features="sqrt", n_jobs=-1,
                 random_state=42)


def encode(X, encoding, n_embed):
    """[k, bands] uint16 rows -> float32 features."""
    if X.dtype != np.uint16:
        raise TypeError(f"uint16 input required, got {X.dtype}")
    if encoding == "paper":
        X = X.copy()
        X[:, :n_embed] -= np.uint16(10000)
        return X.astype(np.float32)
    if encoding == "plain":
        return X.astype(np.float32)
    raise ValueError(encoding)


def _rows(annual, labels, input_name, seasonal=None, block=25):
    """Labelled cells (> 0) of chip stacks as [k, bands] uint16 and [k] float32, in C order."""
    xs, ys = [], []
    for i in range(0, len(annual), block):
        j = min(i + block, len(annual))
        yb = labels[i:j]
        m = yb > 0
        if not m.any():
            continue
        xb = channels.assemble(input_name, annual[i:j], None if seasonal is None else seasonal[i:j])
        xs.append(np.transpose(xb, (0, 2, 3, 1))[m, :])
        ys.append(yb[m])
    return np.concatenate(xs), np.concatenate(ys).ravel()


def train_rf(out_dir, annual, labels, seasonal=None, input_name="AE", encoding="paper", save_model=True,
             log=print, max_rows=None, **params):
    """Fit RF-SLS on chip stacks; writes out_dir/{rf_sls.joblib, result.json}.

    max_rows: if more labelled pixels are available, a fixed random subset of this size (seed 42, original order
    kept) is used, which bounds the memory of the forest for very large training regions. None (default) = all,
    as in the paper.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    out = Path(out_dir)
    if (out / "result.json").exists():
        raise FileExistsError(f"refusing to overwrite finished run {out}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    A, Y = Stack(annual), Stack(labels)
    S = Stack(seasonal) if seasonal else None
    input_name = channels.canonical(input_name)
    parts = [_rows(Stack(a), Stack(y), input_name, None if S is None else Stack(s))
             for a, y, s in zip(A.files, Y.files, (S.files if S else [None] * len(A.files)))]
    X = np.concatenate([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
    del parts
    n_all = int(X.shape[0])
    if max_rows and n_all > max_rows:
        keep = np.sort(np.random.default_rng(42).choice(n_all, int(max_rows), replace=False))
        X, y = X[keep], y[keep]
        log(f"[{time.time()-t0:6.1f}s] {n_all:,} labelled pixels; random subset of {int(max_rows):,} used (max_rows)")
    X = encode(X, encoding, channels.INPUTS[input_name]["n_embed"])
    ok = np.isfinite(y)
    for j in range(X.shape[1]):
        ok &= np.isfinite(X[:, j])
    X, y = X[ok], y[ok]
    X_fit, X_hold, y_fit, y_hold = train_test_split(X, y, test_size=0.2, random_state=42)
    log(f"[{time.time()-t0:6.1f}s] {X.shape[0]:,} labelled pixels, {X_fit.shape[0]:,} used for fitting")
    p = dict(RF_PARAMS, **params)
    rf = RandomForestRegressor(**p).fit(X_fit, y_fit)
    y_hat = rf.predict(X_hold)
    meta = dict(model="rf-sls", input=input_name, encoding=encoding, params=p, n_rows=int(X.shape[0]),
                n_rows_available=n_all, max_rows=max_rows,
                n_fit=int(X_fit.shape[0]), holdout_r2=float(r2_score(y_hold, y_hat)),
                holdout_rmse_m=float(np.sqrt(np.mean((y_hold - y_hat) ** 2))),
                holdout_mae_m=float(mean_absolute_error(y_hold, y_hat)),
                sklearn=__import__("sklearn").__version__, fit_seconds=round(time.time() - t0, 1))
    log(f"[{time.time()-t0:6.1f}s] fitted; hold-out RMSE {meta['holdout_rmse_m']:.3f} m")
    if save_model:
        import joblib
        rf.chm_meta = meta
        joblib.dump(rf, out / "rf_sls.joblib")
    json.dump(meta, open(out / "result.json", "w"), indent=2)
    return rf, meta


def load_rf(path):
    """Fitted forest and its metadata (input layout, encoding).

    Metadata is read from the model (attribute chm_meta) or from a JSON file next to it
    (<model path>.json), e.g. {"model": "rf-sls", "input": "AE", "encoding": "paper"}.

    Loaded with mmap_mode='r': the trees copy their node arrays out of the memory-mapped file instead of out of a
    second in-memory copy, which halves the peak memory of loading (a 3.9 GB forest: 8.5 -> 4.6 GB) with the same
    predictions; joblib ignores it for compressed files and plain pickles.
    """
    import joblib
    rf = joblib.load(path, mmap_mode="r")
    meta = getattr(rf, "chm_meta", None)
    side = Path(str(path) + ".json")
    if meta is None and side.exists():
        meta = json.load(open(side))
    if meta is None:
        raise ValueError(f"{path} has no metadata; write {side} with input and encoding")
    meta = dict(meta, input=channels.canonical(meta["input"]))       # earlier names IE, I -> AE, A
    n = channels.INPUTS[meta["input"]]["n_channels"]
    if getattr(rf, "n_features_in_", n) != n:
        raise ValueError(f"forest has {rf.n_features_in_} features, input {meta['input']} has {n}")
    return rf, meta


def predict_rf_chips(rf, meta, annual, seasonal=None):
    """Predict chip stacks chip by chip -> float32 [n, H, W]."""
    n_embed = channels.INPUTS[meta["input"]]["n_embed"]
    out = np.empty((len(annual),) + tuple(annual.shape[2:]), dtype=np.float32)
    for i in range(len(annual)):
        x = channels.assemble(meta["input"], annual[i:i + 1], None if seasonal is None else seasonal[i:i + 1])
        f = encode(np.transpose(x, (0, 2, 3, 1)).reshape(-1, x.shape[1]), meta["encoding"], n_embed)
        out[i] = rf.predict(f).astype(np.float32).reshape(out.shape[1:])
    return out


def predict_rf_raster(rf, meta, annual, seasonal=None, rows=256):
    """Predict an in-memory raster [76, H, W] strip by strip -> float32 [H, W] (NaN where all bands are 0)."""
    n_embed = channels.INPUTS[meta["input"]]["n_embed"]
    H, W = annual.shape[1:]
    out = np.empty((H, W), dtype=np.float32)
    for r in range(0, H, rows):
        x = channels.assemble(meta["input"], annual[None, :, r:r + rows], None if seasonal is None else
                              seasonal[None, :, r:r + rows])[0]
        f = encode(x.reshape(x.shape[0], -1).T, meta["encoding"], n_embed)
        out[r:r + rows] = rf.predict(f).astype(np.float32).reshape(-1, W)
    out[(np.asarray(annual) == 0).all(axis=0)] = np.nan
    return out
