"""Shared helpers: paths, data access and ISLES metrics (Dice, HD95, volume diff, lesion F1, count diff)
in the common 1 mm space used by training/ and results/."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

HERE = Path(__file__).resolve().parent
DATA = HERE / "data_fast"
RES = HERE / "outputs" / "fast_results"
CONN = np.ones((3, 3, 3), bool)


def load_index():
    return pd.read_csv(DATA / "index.csv")


def load_case(case, mmap=False):
    m = "r" if mmap else None
    return np.load(DATA / f"{case}_img.npy", mmap_mode=m), np.load(DATA / f"{case}_seg.npy", mmap_mode=m)


def hd95(a, b):
    idx = np.argwhere(a | b)
    lo, hi = np.maximum(idx.min(0) - 2, 0), idx.max(0) + 3
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    a, b = a[sl], b[sl]
    sa = a & ~ndimage.binary_erosion(a, CONN)
    sb = b & ~ndimage.binary_erosion(b, CONN)
    da = ndimage.distance_transform_edt(~sa)
    db = ndimage.distance_transform_edt(~sb)
    return float(np.percentile(np.concatenate([db[sa], da[sb]]), 95))


def metrics(pred, gt):
    """Dice, HD95 (mm), |volume diff| (mL), lesion-wise F1, |lesion count diff|; voxels are 1 mm^3."""
    g, p = gt.astype(bool), pred.astype(bool)
    lg, ng = ndimage.label(g, CONN)
    lp, npred = ndimage.label(p, CONN)
    tp = len(np.setdiff1d(np.unique(lg[p]), [0]))
    fp = npred - len(np.setdiff1d(np.unique(lp[g]), [0]))
    fn = ng - tp
    r = {"gt_ml": g.sum() / 1000, "pred_ml": p.sum() / 1000,
         "abs_vol_diff_ml": abs(int(p.sum()) - int(g.sum())) / 1000,
         "lesion_f1": 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 1.0,
         "lesion_count_diff": abs(npred - ng), "dice": np.nan, "hd95_mm": np.nan}
    if g.any():
        r["dice"] = 2 * (g & p).sum() / (g.sum() + p.sum())
        r["hd95_mm"] = hd95(g, p) if p.any() else np.nan
    return r


def case_metrics(case, split, pred_packed, shape):
    pred = np.unpackbits(pred_packed)[: int(np.prod(shape))].reshape(shape).astype(bool)
    _, gt = load_case(case)
    return {"case": case, "split": split, **metrics(pred, gt > 0)}


def save_pred(model, case, pred):
    d = RES / "preds" / model
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{case}.npy", np.packbits(pred.astype(bool)))
