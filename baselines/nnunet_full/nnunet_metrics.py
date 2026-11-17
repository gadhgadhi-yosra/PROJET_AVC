"""Validation metrics of the nnU-Net baseline, ID vs OOD (PDF sections 3.3 and 3.6).

  ID  = fold-0 validation predictions written by nnU-Net at the end of training (training centres)
  OOD = predictions on the held-out centres (run_nnunet.py predict)

Per case: Dice, HD95 (mm), absolute volume difference (mL), lesion-wise F1, lesion count difference
(lesions = 26-connected components). Summaries: overall, by centre, by lesion size, by stage
(days post stroke), plus the central measure gap = Dice(ID) - Dice(OOD) with a bootstrap 95 % CI.
Cases with an empty reference mask have no Dice/HD95 and are reported separately.

Writes outputs/nnunet_validation_metrics_cases.csv and outputs/nnunet_validation_metrics.txt.
Usage: nnunet/.venv/Scripts/python nnunet/nnunet_full/nnunet_metrics.py [TRAINER]
"""
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parents[1]  # nnunet/ root
DATASET = "Dataset601_ATLAS3"
TRAINER = sys.argv[1] if len(sys.argv) > 1 else "nnUNetTrainer_100epochs"
MODEL = f"{TRAINER}__nnUNetPlans_4GB__3d_fullres"
FOLD_DIR = HERE / "nnUNet_results" / DATASET / MODEL / "fold_0"
PRED = {"ID": FOLD_DIR / "validation", "OOD": HERE / "predictions_ood" / f"{MODEL}_fold0"}
GT = {"ID": HERE / "nnUNet_raw" / DATASET / "labelsTr", "OOD": HERE / "nnUNet_raw" / DATASET / "labelsTs"}
OUT = HERE / "outputs"
CONN = np.ones((3, 3, 3), bool)
SIZE_BINS = [0, 1, 10, 50, np.inf]
SIZE_LABELS = ["< 1 mL", "1-10 mL", "10-50 mL", "> 50 mL"]


def hd95(a, b, spacing):
    """95th percentile of symmetric surface distances, computed on the joint bounding box."""
    idx = np.argwhere(a | b)
    lo, hi = np.maximum(idx.min(0) - 2, 0), idx.max(0) + 3
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    a, b = a[sl], b[sl]
    sa = a & ~ndimage.binary_erosion(a, CONN)
    sb = b & ~ndimage.binary_erosion(b, CONN)
    da = ndimage.distance_transform_edt(~sa, sampling=spacing)
    db = ndimage.distance_transform_edt(~sb, sampling=spacing)
    return float(np.percentile(np.concatenate([db[sa], da[sb]]), 95))


def case_metrics(args):
    case, split, pred_path, gt_path = args
    g_img = nib.load(gt_path)
    g = np.asarray(g_img.dataobj) > 0
    p = np.asarray(nib.load(pred_path).dataobj) > 0
    spacing = g_img.header.get_zooms()[:3]
    ml = float(np.prod(spacing)) / 1000.0
    lg, ng = ndimage.label(g, CONN)
    lp, npred = ndimage.label(p, CONN)
    tp = len(np.setdiff1d(np.unique(lg[p]), [0]))         # reference lesions touched by the prediction
    fp = npred - len(np.setdiff1d(np.unique(lp[g]), [0]))  # predicted lesions touching no reference lesion
    fn = ng - tp
    r = {"case": case, "split": split, "gt_ml": g.sum() * ml, "pred_ml": p.sum() * ml,
         "abs_vol_diff_ml": abs(int(p.sum()) - int(g.sum())) * ml,
         "n_lesions_gt": ng, "n_lesions_pred": npred, "lesion_count_diff": abs(npred - ng),
         "lesion_f1": 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 1.0,
         "dice": np.nan, "hd95_mm": np.nan}
    if g.any():
        r["dice"] = 2 * (g & p).sum() / (g.sum() + p.sum())
        r["hd95_mm"] = hd95(g, p, spacing) if p.any() else np.nan
    return r


def boot_gap_ci(a, b, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    gaps = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return np.percentile(gaps, [2.5, 97.5])


def stage(days):
    if pd.isna(days):
        return "unknown"
    return "acute (< 7 d)" if days < 7 else "subacute (7-180 d)" if days <= 180 else "chronic (> 180 d)"


def table(df, by, cols=("dice", "hd95_mm", "abs_vol_diff_ml", "lesion_f1", "lesion_count_diff")):
    g = df.groupby(by, observed=True)
    t = g[list(cols)].mean().round(3)
    t.insert(0, "n", g.size())
    return t


def training_summary():
    logs = sorted(FOLD_DIR.glob("training_log_*.txt"))
    if not logs:
        return []
    txt = "\n".join(l.read_text(encoding="utf-8", errors="ignore") for l in logs)
    epochs = re.findall(r"Epoch (\d+)\s*$", txt, re.M)
    pseudo = re.findall(r"Pseudo dice \[([^\]]*)\]", txt)
    best = re.findall(r"New best EMA pseudo Dice: ([\d.]+)", txt)
    times = [float(x) for x in re.findall(r"Epoch time: ([\d.]+) s", txt)]
    lines = [f"epochs run: {int(epochs[-1]) + 1 if epochs else '?'}"
             + (f"   mean epoch time: {np.mean(times) / 60:.1f} min" if times else "")]
    if pseudo:
        last = re.findall(r"\d+\.\d+", pseudo[-1])
        lines.append(f"last pseudo Dice (val patches): {last[-1] if last else '?'}   "
                     f"best EMA pseudo Dice: {best[-1] if best else '?'}")
    return lines


def main():
    split = pd.read_csv(OUT / "split.csv")
    jobs = []
    for s, d in PRED.items():
        if not d.exists():
            print(f"[{s}] no predictions in {d} - skipped")
            continue
        for f in sorted(d.glob("*.nii.gz")):
            case = f.name[:-len(".nii.gz")]
            jobs.append((case, s, f, GT[s] / f.name))
    with ProcessPoolExecutor(6) as ex:
        rows = list(ex.map(case_metrics, jobs, chunksize=4))
    df = pd.DataFrame(rows).merge(split[["case", "site", "days_post_stroke"]], on="case", how="left")
    df["lesion_size"] = pd.cut(df["gt_ml"], SIZE_BINS, labels=SIZE_LABELS, right=False)
    df["stage"] = df["days_post_stroke"].map(stage)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "nnunet_validation_metrics_cases.csv", index=False)

    nonempty = df[df["gt_ml"] > 0]
    L = [f"nnU-Net v2 baseline - {MODEL}, fold 0", "=" * 78, *training_summary(), ""]
    nn_sum = FOLD_DIR / "validation" / "summary.json"
    if nn_sum.exists():
        fg = json.loads(nn_sum.read_text())["foreground_mean"]
        L.append(f"nnU-Net summary.json (ID val, foreground_mean): Dice {fg['Dice']:.4f}  IoU {fg['IoU']:.4f}")
        L.append("")

    L += ["Overall (cases with a non-empty reference mask; HD95 also excludes empty predictions)", "-" * 78]
    for s, d in nonempty.groupby("split"):
        L.append(f"{s:<4} n={len(d):<4} centres={d['site'].nunique():<3}"
                 f" Dice {d['dice'].mean():.3f} ± {d['dice'].std():.3f} (median {d['dice'].median():.3f})"
                 f" | HD95 {d['hd95_mm'].median():.1f} mm (median)"
                 f" | AVD {d['abs_vol_diff_ml'].mean():.1f} mL"
                 f" | lesion F1 {d['lesion_f1'].mean():.3f}"
                 f" | count diff {d['lesion_count_diff'].mean():.1f}")
    if {"ID", "OOD"} <= set(nonempty["split"]):
        a = nonempty.loc[nonempty["split"] == "ID", "dice"].to_numpy()
        b = nonempty.loc[nonempty["split"] == "OOD", "dice"].to_numpy()
        lo, hi = boot_gap_ci(a, b)
        p = mannwhitneyu(a, b).pvalue
        L += ["", f"Generalisation gap  Dice(ID) - Dice(OOD) = {a.mean() - b.mean():+.3f}"
                  f"  [95% bootstrap CI {lo:+.3f}, {hi:+.3f}]  Mann-Whitney p = {p:.3g}"]
    empty = df[df["gt_ml"] == 0]
    if len(empty):
        L += ["", f"Empty reference masks: {len(empty)} cases, "
                  f"{int((empty['pred_ml'] > 0).sum())} with a false-positive prediction "
                  f"(mean {empty['pred_ml'].mean():.2f} mL)"]

    pd.set_option("display.width", 200)
    for title, by in [("By lesion size", ["split", "lesion_size"]), ("By stage (days post stroke)", ["split", "stage"]),
                      ("By centre", ["split", "site"])]:
        L += ["", title, "-" * 78, table(nonempty, by).to_string()]
    txt = "\n".join(L)
    print(txt)
    (OUT / "nnunet_validation_metrics.txt").write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
