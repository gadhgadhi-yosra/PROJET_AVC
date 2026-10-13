"""Results of all models (training/train_cnn.py, training/train_ml.py): one table + 3 figures for the slides.

Reads outputs/fast_results/*_cases.csv (metrics per case in the common 1 mm space).
Fair comparison: the table and fig 1 use only the cases every model was evaluated on (ML = random
150 ID + 150 OOD subset); the CNNs are also reported on all cases. Dice/HD95 exclude empty reference masks.

Writes to outputs/:
  results.txt / results.csv          metrics per model: Dice, HD95, volume diff, lesion F1, count diff,
                                     gap Dice(ID) - Dice(OOD) with 95 % bootstrap CI, size-adjusted gap
  fig1_dice_id_vs_ood.png            Dice ID vs OOD for every model
  fig2_dice_by_lesion_size.png       best model: Dice by lesion size, ID vs OOD (lesion size drives Dice)
  fig3_examples.png                  best model: reference vs prediction, ID and OOD cases
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import HERE, RES, load_case, load_index

OUT = HERE / "outputs"
ID_C, OOD_C, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
SIZE_BINS, SIZE_LABELS = [0, 1, 10, 50, np.inf], ["< 1 mL", "1-10 mL", "10-50 mL", "> 50 mL"]
LABELS = {
    "cnn_pretrained_freeze-none": "CNN pretrained, full fine-tune",
    "cnn_pretrained_freeze-partial": "CNN pretrained, early encoder frozen",
    "cnn_pretrained_freeze-encoder": "CNN pretrained, whole encoder frozen",
    "cnn_random_init": "CNN random init (no pretraining)",
    "ml_gradient_boosting": "ML gradient boosting",
    "ml_random_forest": "ML random forest",
    "ml_logistic_regression": "ML logistic regression",
}


# ---------------------------------------------------------------- table
def boot_gap(a, b, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    g = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return np.percentile(g, [2.5, 97.5])


def size_adjusted_gap(i, o):
    """Gap with both sets re-weighted to the same (pooled) lesion-size mix: Dice rises with lesion size and
    the OOD centres happen to have larger lesions, which hides the centre effect in the raw gap."""
    bi = pd.cut(i["gt_ml"], SIZE_BINS, labels=SIZE_LABELS, right=False)
    bo = pd.cut(o["gt_ml"], SIZE_BINS, labels=SIZE_LABELS, right=False)
    w = pd.concat([bi, bo]).value_counts(normalize=True)
    mi = i.groupby(bi, observed=True)["dice"].mean()
    mo = o.groupby(bo, observed=True)["dice"].mean()
    return float(sum(w[k] * (mi[k] - mo[k]) for k in w.index if k in mi and k in mo))


def summarise(df):
    rows = []
    for m, d in df.groupby("model"):
        ne = d[d["gt_ml"] > 0]
        i, o = ne[ne["split"] == "ID"], ne[ne["split"] == "OOD"]
        lo, hi = boot_gap(i["dice"].to_numpy(), o["dice"].to_numpy())
        rows.append({
            "model": m, "description": LABELS.get(m, m),
            "dice_ID": i["dice"].mean(), "dice_OOD": o["dice"].mean(),
            "gap": i["dice"].mean() - o["dice"].mean(), "gap_ci_low": lo, "gap_ci_high": hi,
            "gap_size_adj": size_adjusted_gap(i, o),
            "hd95_ID": i["hd95_mm"].median(), "hd95_OOD": o["hd95_mm"].median(),
            "vol_diff_ml_ID": i["abs_vol_diff_ml"].mean(), "vol_diff_ml_OOD": o["abs_vol_diff_ml"].mean(),
            "lesion_f1_ID": i["lesion_f1"].mean(), "lesion_f1_OOD": o["lesion_f1"].mean(),
            "count_diff_ID": i["lesion_count_diff"].mean(), "count_diff_OOD": o["lesion_count_diff"].mean(),
            "n_ID": len(i), "n_OOD": len(o),
        })
    return pd.DataFrame(rows).sort_values("dice_ID", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- figures
def style(ax):
    ax.set_facecolor(SURF)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0, colors=INK2, labelsize=8)
    ax.set_axisbelow(True)


def fig1(s):
    s = s.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(s) + 1.7), dpi=200)
    fig.patch.set_facecolor(SURF)
    style(ax)
    y = np.arange(len(s))
    ax.hlines(y, s["dice_OOD"], s["dice_ID"], color="#c9c8c2", lw=2, zorder=1)
    ax.scatter(s["dice_ID"], y, s=70, color=ID_C, edgecolor=SURF, lw=2, zorder=3,
               label="ID: new patients, training centres")
    ax.scatter(s["dice_OOD"], y, s=70, color=OOD_C, edgecolor=SURF, lw=2, zorder=3,
               label="OOD: centres never seen in training")
    for k, r in s.iterrows():
        ax.text(max(r["dice_ID"], r["dice_OOD"]) + 0.012, k, f"{r['dice_ID']:.2f} / {r['dice_OOD']:.2f}",
                va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, s["description"], fontsize=9, color=INK)
    ax.set_xlim(0, min(1, s[["dice_ID", "dice_OOD"]].to_numpy().max() + 0.15))
    ax.set_xlabel(f"Mean Dice (same {s['n_ID'].iloc[0]} ID + {s['n_OOD'].iloc[0]} OOD cases for every model)",
                  fontsize=9, color=INK2)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9)
    ax.set_title("Stroke lesion segmentation: Dice on known vs unseen centres  (labels: ID / OOD)",
                 fontsize=11, color=INK, pad=26, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_dice_id_vs_ood.png", facecolor=SURF)
    plt.close(fig)


def fig2(d, name):
    d = d.assign(size=pd.cut(d["gt_ml"], SIZE_BINS, labels=SIZE_LABELS, right=False))
    t = d.groupby(["size", "split"], observed=True)["dice"].agg(["mean", "size"]).unstack("split")
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    fig.patch.set_facecolor(SURF)
    style(ax)
    x, w = np.arange(len(t)), 0.36
    for off, sp, c in [(-w / 2 - 0.01, "ID", ID_C), (w / 2 + 0.01, "OOD", OOD_C)]:
        bars = ax.bar(x + off, t[("mean", sp)], w, color=c, label=sp, zorder=2)
        for b, n, v in zip(bars, t[("size", sp)], t[("mean", sp)]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}\nn={n}", ha="center", va="bottom",
                    fontsize=7.5, color=INK2)
    ax.set_xticks(x, t.index.astype(str), fontsize=9, color=INK)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean Dice", fontsize=9, color=INK2)
    ax.set_xlabel("Reference lesion volume", fontsize=9, color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title(f"{name}: Dice depends mostly on lesion size, not on the centre", fontsize=10.5,
                 color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_dice_by_lesion_size.png", facecolor=SURF)
    plt.close(fig)


def fig3(df, model, name):
    df = df[(df["model"] == model) & (df["gt_ml"] >= 1)]
    picks = []
    for sp in ["ID", "OOD"]:
        d = df[df["split"] == sp].sort_values("dice").reset_index(drop=True)
        picks += [(sp, d.iloc[int(q * (len(d) - 1))]) for q in (0.25, 0.5, 0.75)]
    fig, axes = plt.subplots(2, 3, figsize=(10, 7.4), dpi=200)
    fig.patch.set_facecolor(SURF)
    for ax, (sp, r) in zip(axes.ravel(), picks):
        img, seg = load_case(r["case"])
        pred = np.unpackbits(np.load(RES / "preds" / model / f"{r['case']}.npy"))[: seg.size].reshape(seg.shape)
        z = int(seg.sum(axis=(1, 2)).argmax())  # slice with the largest lesion area
        sl = np.asarray(img[z], np.float32)
        ax.imshow(np.where(sl == 0, -2.5, sl)[::-1], cmap="gray", vmin=-2.5, vmax=2.5)
        ax.contour(seg[z][::-1], levels=[0.5], colors=ID_C, linewidths=1.6)
        ax.contour(pred[z][::-1], levels=[0.5], colors=OOD_C, linewidths=1.6)
        ax.set_title(f"{sp} - {r['case'].upper()}  Dice {r['dice']:.2f}\n"
                     f"lesion {r['gt_ml']:.1f} mL, predicted {r['pred_ml']:.1f} mL", fontsize=8.5, color=INK)
        ax.axis("off")
    fig.legend(handles=[plt.Line2D([], [], color=ID_C, lw=2, label="Reference (manual mask)"),
                        plt.Line2D([], [], color=OOD_C, lw=2, label=f"Prediction ({name})")],
               loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Top: ID (training centres) - bottom: OOD (unseen centres); "
                 "cases at the 25th / 50th / 75th Dice percentile", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(OUT / "fig3_examples.png", facecolor=SURF)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    full = pd.concat([pd.read_csv(f) for f in sorted(RES.glob("*_cases.csv"))], ignore_index=True)
    full = full.merge(load_index()[["case", "site"]], on="case", how="left")
    common = set.intersection(*(set(d["case"]) for _, d in full.groupby("model")))
    df = full[full["case"].isin(common)]
    s = summarise(df)
    s.to_csv(OUT / "results.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    cols = ["description", "dice_ID", "dice_OOD", "gap", "gap_ci_low", "gap_ci_high", "gap_size_adj",
            "hd95_ID", "hd95_OOD", "vol_diff_ml_ID", "vol_diff_ml_OOD", "lesion_f1_ID", "lesion_f1_OOD",
            "count_diff_ID", "count_diff_OOD"]
    L = ["Stroke lesion segmentation - validation metrics, ID vs OOD (sorted by ID Dice)", "=" * 110,
         f"ID  = {s['n_ID'].iloc[0]} new patients from {df.loc[df['split'] == 'ID', 'site'].nunique()} training centres",
         f"OOD = {s['n_OOD'].iloc[0]} patients from {df.loc[df['split'] == 'OOD', 'site'].nunique()} centres never seen in training",
         "Dice, lesion F1, volume diff (mL), count diff = means; HD95 = median (mm); "
         "gap = Dice(ID) - Dice(OOD), 95 % bootstrap CI", "", s[cols].round(3).to_string(index=False), ""]

    cnn = full[full["model"].str.startswith("cnn")]
    # pick the model on ID validation only - choosing on OOD would select on the test set
    pool = s[s["model"].str.startswith("cnn")] if len(cnn) else s
    best = pool.sort_values("dice_ID", ascending=False)["model"].iloc[0]
    if len(cnn) and len(common) < cnn["case"].nunique():
        sc = summarise(cnn)
        L += [f"CNN models on all evaluation cases ({sc['n_ID'].iloc[0]} ID + {sc['n_OOD'].iloc[0]} OOD):",
              sc[cols].round(3).to_string(index=False), ""]

    d = full[(full["model"] == best) & (full["gt_ml"] > 0)]
    L += [f"Best CNN on ID validation: {LABELS[best]} (all its evaluation cases)", "-" * 110,
          "Median lesion volume: " + ", ".join(f"{k} {v.median():.1f} mL" for k, v in d.groupby("split")["gt_ml"])]
    for sp in ["OOD", "ID"]:
        c = (d[d["split"] == sp].groupby("site").agg(n=("dice", "size"), dice=("dice", "mean"),
                                                     median_lesion_ml=("gt_ml", "median"))
             .query("n >= 3").sort_values("dice").round(3))
        L += ["", f"{sp} centres with >= 3 cases, worst to best:", c.to_string()]
    txt = "\n".join(L)
    print(txt)
    (OUT / "results.txt").write_text(txt + "\n", encoding="utf-8")

    fig1(s)
    fig2(d, LABELS[best])
    fig3(full, best, LABELS[best])
    print("\nfigures:", *[OUT / f for f in ("fig1_dice_id_vs_ood.png", "fig2_dice_by_lesion_size.png",
                                           "fig3_examples.png")], sep="\n  ")


if __name__ == "__main__":
    main()
