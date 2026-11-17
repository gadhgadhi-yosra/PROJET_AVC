"""Classical machine-learning baselines: voxel-wise lesion classification from hand-crafted features (CPU).

Features per brain voxel (1 mm, z-scored T1): intensity, Gaussian-smoothed intensity (sigma 1/2/4 mm),
local std (sigma 2), left-right asymmetry (smoothed T1 minus its mirror - lesions are unilateral),
gradient magnitude, distance to the brain edge, normalised position (z, y, |x - centre|).
Models: logistic regression, random forest, histogram gradient boosting, trained on a balanced voxel
sample from 250 training cases. Decision threshold tuned on 15 other training cases (mean Dice).
Evaluated on a fixed random subset (150 ID + 150 OOD) of the CNN's evaluation cases. Writes outputs/fast_results/ml_*_cases.csv.
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RES, load_case, load_index, metrics, save_pred

N_FIT, N_TUNE = 250, 15
EVAL_PER_SET = 150  # random ID / OOD cases evaluated (full set took > 1 h on the laptop CPU)
N_POS, N_NEG = 2000, 4000
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]
FEATS = ["t1", "g1", "g2", "g4", "std2", "asym", "grad", "dist_edge", "z", "y", "x_lat"]
MODELS = None  # set in workers


def features(case):
    img, seg = load_case(case)
    x = img.astype(np.float32)
    brain = x != 0
    g2 = ndimage.gaussian_filter(x, 2)
    std2 = np.sqrt(np.maximum(ndimage.gaussian_filter(x * x, 2) - g2 ** 2, 0))
    f = [x, ndimage.gaussian_filter(x, 1), g2, ndimage.gaussian_filter(x, 4), std2,
         g2 - g2[:, :, ::-1], ndimage.gaussian_gradient_magnitude(x, 1),
         ndimage.distance_transform_edt(brain).astype(np.float32)]
    pos = np.nonzero(brain)
    X = np.column_stack([a[pos] for a in f] + [pos[0] / x.shape[0], pos[1] / x.shape[1],
                                                np.abs(pos[2] / x.shape[2] - 0.5)]).astype(np.float32)
    return X, seg[pos] > 0, pos, x.shape, seg > 0


def sample(case):
    X, y, *_ = features(case)
    rng = np.random.default_rng(abs(hash(case)) % 2 ** 32)
    pi, ni = np.flatnonzero(y), np.flatnonzero(~y)
    pick = np.concatenate([rng.choice(pi, min(N_POS, len(pi)), replace=False),
                           rng.choice(ni, min(N_NEG, len(ni)), replace=False)])
    return X[pick], y[pick]


def init(models):
    global MODELS
    MODELS = models


def probs(case):
    X, _, pos, shape, gt = features(case)
    out = {}
    for name, m in MODELS.items():
        pr = np.zeros(shape, np.float32)
        pr[pos] = m.predict_proba(X)[:, 1]
        out[name] = pr
    return out, gt


def tune_case(case):
    pr, gt = probs(case)
    return {(n, t): metrics(p > t, gt)["dice"] for n, p in pr.items() for t in THRESHOLDS}


def eval_case(args):
    case, split, thr = args
    pr, gt = probs(case)
    rows = []
    for n, p in pr.items():
        pred = p > thr[n]
        save_pred(n, case, pred)
        rows.append({"case": case, "split": split, "model": n, **metrics(pred, gt)})
    return rows


def save(rows, thr):
    RES.mkdir(parents=True, exist_ok=True)
    for n, d in pd.DataFrame(rows).groupby("model"):
        d.assign(threshold=thr[n]).to_csv(RES / f"{n}_cases.csv", index=False)


def main():
    t0 = time.time()
    idx = load_index()
    rng = np.random.default_rng(0)
    train = idx[(idx["set"] == "train") & (idx["lesion_vox"] > 0)]["case"].to_numpy()
    rng.shuffle(train)
    fit_cases, tune_cases = train[:N_FIT], train[N_FIT:N_FIT + N_TUNE]
    with ProcessPoolExecutor(4) as ex:
        parts = list(ex.map(sample, fit_cases, chunksize=4))
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    print(f"training voxels: {len(y)} ({y.mean():.1%} lesion)  [{(time.time() - t0) / 60:.1f} min]", flush=True)

    models = {
        "ml_logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
        "ml_random_forest": RandomForestClassifier(30, max_depth=14, min_samples_leaf=20, max_samples=0.5,
                                                   n_jobs=4, random_state=0),
        "ml_gradient_boosting": HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=63, random_state=0),
    }
    for n, m in models.items():
        m.fit(X, y)
        print(f"  fitted {n}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
    models["ml_random_forest"].set_params(n_jobs=1)
    rf = models["ml_random_forest"]
    imp = pd.Series(rf.feature_importances_, FEATS).sort_values(ascending=False).round(3)
    print("random forest feature importance:\n" + imp.to_string(), flush=True)

    with ProcessPoolExecutor(4, initializer=init, initargs=(models,)) as ex:
        tuned = list(ex.map(tune_case, tune_cases))
    thr = {}
    for n in models:
        scores = {t: np.nanmean([d[(n, t)] for d in tuned]) for t in THRESHOLDS}
        thr[n] = max(scores, key=scores.get)
        print(f"  {n}: threshold {thr[n]} (tuning Dice {scores[thr[n]]:.3f})", flush=True)

    ev = pd.concat([d.sample(min(EVAL_PER_SET, len(d)), random_state=0)
                    for _, d in idx[idx["set"].isin(["id_val", "ood_test"])].groupby("set")])
    jobs = [(c, "ID" if s == "id_val" else "OOD", thr) for c, s in zip(ev["case"], ev["set"])]
    rows = []
    with ProcessPoolExecutor(4, initializer=init, initargs=(models,)) as ex:
        for i, r in enumerate(ex.map(eval_case, jobs, chunksize=2), 1):
            rows += r
            if i % 50 == 0 or i == len(jobs):
                print(f"  eval {i}/{len(jobs)}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
                save(rows, thr)
    df = pd.DataFrame(rows)
    s = df[df["gt_ml"] > 0].groupby(["model", "split"])[["dice", "lesion_f1", "abs_vol_diff_ml"]].mean().round(3)
    print(s.to_string(), f"\ndone in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
