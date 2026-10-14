"""Per-site summary table + exploration figures (week 1-2 deliverable: statistiques et histogrammes par centre).

Reads outputs/subjects.parquet and outputs/histograms.npz (from 01_compute_stats.py) and writes:
  outputs/site_summary.csv
  figures/01_site_fingerprint.png     per-site distributions: n, voxel size, raw intensity, lesion volume, days post stroke
  figures/02_intensity_histograms.png normalized in-brain intensity histogram per site vs. all sites
  figures/02b_..._selected.png        same, 8 contrasting sites (slide-sized)
  figures/03_lesions.png              lesion volume distribution, by stage, number of lesion components
  figures/04_time_since_stroke.png    days post stroke and derived stage per cohort
  figures/05_examples.png             one representative axial slice per large site, lesion outlined
  figures/06_qc_anomalies.png         QC findings: R032 header spacing, R029 non-zero background
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT, FIG = HERE / "outputs", HERE / "figures"

# palette: validated reference instance (dataviz skill, light mode)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
STAGE_COLORS = {"acute (≤7 d)": "#86b6ef", "subacute (8–180 d)": "#3987e5", "chronic (>180 d)": "#1c5cab",
                "unknown": "#c3c2b7"}
COHORT_COLORS = {"ATLAS v2.0": BLUE, "new in ATLAS3": ORANGE}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 8.5,
})

V2_LABELS = {"Training", "Testing", "Training_ATLAS2", "Testing_ATLAS2"}


def load():
    d = pd.read_parquet(OUT / "subjects.parquet")
    # SOOP is entirely new in ATLAS3; its 2 empty metadata files inherit that
    d["cohort"] = np.where(d["atlas_dataset"].isin(V2_LABELS), "ATLAS v2.0", "new in ATLAS3")
    days = d["days_post_stroke"]
    d["stage"] = pd.cut(days, [-np.inf, 7, 180, np.inf], labels=list(STAGE_COLORS)[:3]).astype(str)
    d.loc[days.isna(), "stage"] = "unknown"
    d["voxel_str"] = d[["vox_x", "vox_y", "vox_z"]].round(2).astype(str).agg("×".join, axis=1)
    h = np.load(OUT / "histograms.npz", allow_pickle=True)
    assert (h["subject"] == d["subject"].to_numpy()).all()
    return d, h["hist"], h["edges"]


def site_summary(d):
    g = d.groupby("site")
    s = pd.DataFrame({
        "n": g.size(),
        "n_atlas_v2": g["cohort"].apply(lambda c: (c == "ATLAS v2.0").sum()),
        "n_new_v3": g["cohort"].apply(lambda c: (c == "new in ATLAS3").sum()),
        "n_voxel_sizes": g["voxel_str"].nunique(),
        "main_voxel_mm": g["voxel_str"].agg(lambda v: v.value_counts().index[0]),
        "voxel_mm3_median": g["voxel_mm3"].median(),
        "n_shapes": d.drop_duplicates(["site", "dim_x", "dim_y", "dim_z"]).groupby("site").size(),
        "orientations": g["orientation"].agg(lambda v: ",".join(sorted(v.unique()))),
        "t1_dtypes": g["t1_dtype"].agg(lambda v: ",".join(sorted(v.unique()))),
        "brain_ml_median": g["brain_ml"].median(),
        "int_p50_median": g["int_p50"].median(),
        "int_p99_median": g["int_p99"].median(),
        "int_cv_median": g["int_cv"].median(),
        "lesion_ml_median": g["lesion_ml"].median(),
        "lesion_ml_q25": g["lesion_ml"].quantile(0.25),
        "lesion_ml_q75": g["lesion_ml"].quantile(0.75),
        "n_components_median": g["n_components"].median(),
        "lesion_vs_shell_median": g["lesion_vs_shell"].median(),
        "days_median": g["days_post_stroke"].median(),
        "days_missing_frac": g["days_post_stroke"].apply(lambda x: x.isna().mean()),
        **{f"n_{k.split(' ')[0]}": g["stage"].apply(lambda x, k=k: (x == k).sum()) for k in STAGE_COLORS},
    }).sort_values("n", ascending=False)
    s.round(3).to_csv(OUT / "site_summary.csv")
    return s


def strip(ax, d, order, col, log=True, clip_low=None, color=BLUE):
    rng = np.random.default_rng(0)
    for i, site in enumerate(order):
        v = d.loc[d["site"] == site, col].dropna().to_numpy()
        if clip_low is not None:
            v = np.clip(v, clip_low, None)
        if not len(v):
            continue
        y = i + rng.uniform(-0.28, 0.28, len(v))
        ax.scatter(v, y, s=7, color=color, alpha=0.45, linewidths=0)
        ax.plot([np.median(v)] * 2, [i - 0.38, i + 0.38], color=INK, lw=1.6, solid_capstyle="round")
    if log:
        ax.set_xscale("log")
    ax.grid(axis="y", visible=False)


def fig_site_fingerprint(d, s):
    order = list(s.index)
    fig, axes = plt.subplots(1, 5, figsize=(17, 13.5), sharey=True,
                             gridspec_kw={"width_ratios": [1.1, 1, 1, 1, 1], "wspace": 0.08})
    ax = axes[0]
    y = np.arange(len(order))
    left = np.zeros(len(order))
    for cohort, col in [("n_atlas_v2", "ATLAS v2.0"), ("n_new_v3", "new in ATLAS3")]:
        vals = s.loc[order, cohort].to_numpy()
        ax.barh(y, vals, left=left, height=0.72, color=COHORT_COLORS[col], label=col,
                edgecolor=SURFACE, linewidth=1)
        left += vals
    for yi, n in zip(y, s.loc[order, "n"]):
        ax.text(n + 2, yi, str(n), va="center", fontsize=7.5, color=INK2)
    ax.set_title("Subjects per site")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, s["n"].max() * 1.12)

    strip(axes[1], d, order, "voxel_mm3")
    axes[1].set_title("Voxel volume (mm³)")
    strip(axes[2], d, order, "int_p50")
    axes[2].set_title("Median brain intensity (raw units)")
    strip(axes[3], d, order, "lesion_ml", clip_low=0.01)
    axes[3].set_title("Lesion volume (mL)")
    strip(axes[4], d, order, "days_post_stroke", clip_low=1)
    axes[4].set_title("Days post stroke")
    for site, i in zip(order, y):
        if d.loc[d["site"] == site, "days_post_stroke"].isna().all():
            axes[4].text(1.2, i, "no data", va="center", fontsize=7.5, color=MUTED, style="italic")
    axes[4].axvspan(7, 180, color=GRID, alpha=0.5, lw=0, zorder=0)
    axes[4].text(36, -1.3, "subacute", ha="center", fontsize=7.5, color=MUTED)

    axes[0].set_yticks(y, [f"{st}" for st in order])
    axes[0].set_ylim(len(order) - 0.4, -1.8)
    fig.subplots_adjust(top=0.955)
    fig.suptitle("ATLAS3 training set — site fingerprint (one row per site, dots = subjects, bar = median)",
                 x=0.07, y=0.99, ha="left", fontsize=12, fontweight="bold")
    fig.savefig(FIG / "01_site_fingerprint.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_histograms(d, H, edges, s, sites=None, fname="02_intensity_histograms.png", ncol=8, panel_h=2.0):
    centers = (edges[:-1] + edges[1:]) / 2
    ok = ~np.isnan(H).any(axis=1)
    pooled = np.median(H[ok], axis=0)
    order = sites or list(s.index)
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(17, panel_h * nrow + 0.6), sharex=True, sharey=True, squeeze=False)
    ymax = np.percentile(H[ok].max(axis=1), 95) * 1.1
    for ax, site in zip(axes.flat, order):
        m = (d["site"] == site).to_numpy() & ok
        ax.plot(centers, pooled, color=MUTED, lw=1.2, label="all sites (median)")
        if m.any():
            q = np.percentile(H[m], [25, 50, 75], axis=0)
            ax.fill_between(centers, q[0], q[2], color=BLUE, alpha=0.18, lw=0)
            ax.plot(centers, q[1], color=BLUE, lw=1.8, label="this site (median, IQR)")
        ax.set_title(f"{site}  n={m.sum()}", fontsize=8.5 if sites is None else 10)
        ax.set_xlim(0, 1.25)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_ylim(0, ymax)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    for ax in axes.flat[len(order):]:
        ax.axis("off")
    axes.flat[0].legend(loc="upper left", fontsize=7, bbox_to_anchor=(0, 1.02))
    fig.supxlabel("in-brain intensity / 99th percentile of that scan", color=INK2, fontsize=9)
    fig.suptitle("In-brain T1 intensity histograms by site (each scan scaled by its own p99)",
                 x=0.07, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.01, 1, 0.97 if sites is None else 0.93))
    fig.savefig(FIG / fname, dpi=120)
    plt.close(fig)


def fig_lesions(d):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={"wspace": 0.28})
    v = d["lesion_ml"].clip(lower=0.01)
    bins = np.logspace(-2, 3, 41)
    axes[0].hist(v, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=1)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("lesion volume (mL, log scale)")
    axes[0].set_ylabel("subjects")
    top = axes[0].get_ylim()[1]
    for q, lab, yf in zip(np.percentile(d["lesion_ml"], [25, 50, 75]), ["Q1", "median", "Q3"], [0.97, 0.80, 0.97]):
        axes[0].axvline(q, color=INK, lw=0.8)
        axes[0].text(q, top * yf, f" {lab}\n {q:.1f} mL", fontsize=7.5, color=INK2, va="top",
                     bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1, alpha=0.85))
    axes[0].set_title(f"Lesion volume, all {len(d)} subjects")

    ax = axes[1]
    stages = list(STAGE_COLORS)
    rng = np.random.default_rng(1)
    for i, st in enumerate(stages):
        vv = v[d["stage"] == st].to_numpy()
        ax.scatter(vv, i + rng.uniform(-0.3, 0.3, len(vv)), s=7, color=STAGE_COLORS[st], alpha=0.55, linewidths=0)
        if len(vv):
            ax.plot([np.median(vv)] * 2, [i - 0.4, i + 0.4], color=INK, lw=1.8)
            ax.text(1.4e3, i, f"n={len(vv)}\nmed {np.median(vv):.1f}", fontsize=7.5, color=INK2, va="center")
    ax.set_xscale("log")
    ax.set_xlim(0.008, 3e3)
    ax.set_yticks(range(len(stages)), stages)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("lesion volume (mL, log scale)")
    ax.set_title("Lesion volume by stage (from DAYS_POST_STROKE)")

    ax = axes[2]
    nc = d["n_components"].clip(upper=15)
    counts = nc.value_counts().sort_index()
    ax.bar(counts.index, counts.to_numpy(), color=BLUE, width=0.8, edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(range(0, 16), [str(i) for i in range(15)] + ["15+"])
    ax.set_xlabel("connected lesion components per scan (26-connectivity)")
    ax.set_ylabel("subjects")
    ax.grid(axis="x", visible=False)
    ax.set_title("Number of separate lesions")
    fig.savefig(FIG / "03_lesions.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_time(d):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.3})
    ax = axes[0]
    bins = np.logspace(0, np.log10(12000), 45)
    xs = [d.loc[d["cohort"] == c, "days_post_stroke"].dropna().clip(lower=1) for c in COHORT_COLORS]
    ax.hist(xs, bins=bins, stacked=True, color=list(COHORT_COLORS.values()), edgecolor=SURFACE, linewidth=0.8,
            label=[f"{c} (n={len(x)} with days)" for c, x in zip(COHORT_COLORS, xs)])
    ax.set_xscale("log")
    for x0 in (7, 180):
        ax.axvline(x0, color=INK2, lw=0.8)
    for x0, lab in [(2.6, "acute"), (36, "subacute"), (1500, "chronic")]:
        ax.text(x0, 1.01, lab, color=INK2, fontsize=8.5, ha="center", transform=ax.get_xaxis_transform())
    ax.set_title("Time since stroke", pad=18)
    ax.set_xlabel("days post stroke (log scale; ≤0 shown at 1)")
    ax.set_ylabel("subjects")
    ax.legend(loc="upper left")

    ax = axes[1]
    ct = pd.crosstab(d["cohort"], d["stage"]).reindex(columns=list(STAGE_COLORS), fill_value=0)
    y = np.arange(len(ct))
    left = np.zeros(len(ct))
    for st in ct.columns:
        ax.barh(y, ct[st], left=left, color=STAGE_COLORS[st], label=st, height=0.6, edgecolor=SURFACE, linewidth=2)
        for yi, (l, w) in enumerate(zip(left, ct[st])):
            if w >= 40:
                ax.text(l + w / 2, yi, str(w), ha="center", va="center", fontsize=8,
                        color="#ffffff" if st in ("chronic (>180 d)", "subacute (8–180 d)") else INK)
        left += ct[st].to_numpy()
    ax.set_yticks(y, ct.index)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("subjects")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.set_title("Stage per cohort")
    fig.savefig(FIG / "04_time_since_stroke.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def pick_slice(t1, mk):
    area = mk.reshape(-1, mk.shape[2]).sum(axis=0)
    return int(area.argmax()) if area.max() > 0 else mk.shape[2] // 2


def fig_examples(d, s, n=10):
    sites = list(s.index[:n])
    fig, axes = plt.subplots(2, n // 2, figsize=(17, 8.2))
    for ax, site in zip(axes.flat, sites):
        g = d[(d["site"] == site) & (d["lesion_ml"] > 0)]
        row = g.iloc[(g["lesion_ml"] - g["lesion_ml"].median()).abs().argsort().iloc[0]]
        t1_img = nib.as_closest_canonical(nib.load(row["t1_path"]))
        mk_img = nib.as_closest_canonical(nib.load(row["mask_path"]))
        t1 = np.asarray(t1_img.dataobj, dtype=np.float32)
        mk = np.asarray(mk_img.dataobj) > 0
        z = pick_slice(t1, mk)
        sl, ml = t1[:, :, z].T, mk[:, :, z].T  # RAS: rows = A, cols = R
        rows, cols = np.nonzero(sl > 0)
        if rows.size:  # crop to the brain in this slice
            r0, r1, c0, c1 = max(rows.min() - 5, 0), rows.max() + 6, max(cols.min() - 5, 0), cols.max() + 6
            sl, ml = sl[r0:r1, c0:c1], ml[r0:r1, c0:c1]
        zooms = t1_img.header.get_zooms()[:3]
        ax.imshow(sl, cmap="gray", origin="lower", vmin=0, vmax=np.percentile(t1[t1 > 0], 99.5),
                  aspect=zooms[1] / zooms[0], interpolation="nearest")
        ax.contour(ml, levels=[0.5], colors=[ORANGE], linewidths=1.0)
        ax.set_title(f"{site} · {row['subject'].replace('sub-', '')}", fontsize=9)
        ax.text(0.02, 0.02, f"{row['voxel_str']} mm · {row['orientation']} · lesion {row['lesion_ml']:.1f} mL",
                transform=ax.transAxes, fontsize=7.2, color="#ffffff", va="bottom")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    for ax in axes.flat[len(sites):]:
        ax.axis("off")
    fig.suptitle("One subject per site (10 largest sites, median lesion size) — axial slice with the largest "
                 "lesion area, reoriented to RAS; lesion outlined in orange; image right = patient right",
                 x=0.02, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / "05_examples.png", dpi=110)
    plt.close(fig)


def fig_anomalies(d):
    """QC findings: R032 header spacing (brain stretched front-back) and R029 non-zero background."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1, 1.6, 1]})
    for ax, site in zip(axes[:2], ["R009", "R032"]):
        row = d[d["site"] == site].iloc[0]
        img = nib.as_closest_canonical(nib.load(row["t1_path"]))
        t1 = np.asarray(img.dataobj, dtype=np.float32)
        zooms = img.header.get_zooms()[:3]
        x = int(np.argmax((t1 > 0).sum(axis=(1, 2))))  # sagittal slice with the most brain
        sl = t1[x].T  # rows = S, cols = A
        ax.imshow(sl, cmap="gray", origin="lower", aspect="equal",  # extent is in mm -> equal aspect
                  extent=(0, sl.shape[1] * zooms[1], 0, sl.shape[0] * zooms[2]),
                  vmin=0, vmax=np.percentile(t1[t1 > 0], 99.5))
        ax.set_title(f"{site} · sagittal at header spacing ({row['voxel_str']} mm)\n"
                     f"front-back extent {row['bbox_y_mm']:.0f} mm, brain {row['brain_ml']:.0f} mL", fontsize=9)
        ax.set_xlabel("mm (posterior → anterior)")
        ax.grid(False)
    row = d[(d["site"] == "R029") & (d["background_value"] > 0)].iloc[0]
    t1 = np.asarray(nib.load(row["t1_path"]).dataobj, dtype=np.float32)
    z = t1.shape[2] // 2
    axes[2].imshow((t1[:, :, z] > 0).T, cmap="gray", origin="lower", interpolation="nearest", vmin=0, vmax=1)
    axes[2].contour((t1[:, :, z] > row["background_value"]).T, levels=[0.5], colors=[ORANGE], linewidths=1.2)
    axes[2].set_title(f"R029 · naive brain mask 'T1 > 0' (white)\nbackground = {row['background_value']:.4f}, "
                      "not 0; correct mask in orange", fontsize=9)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].grid(False)
    fig.suptitle("Quality control catches real problems", x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "06_qc_anomalies.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_one_subject(d, subject="sub-r040s080"):
    """What one subject looks like: T1 slice, lesion mask, overlay."""
    row = d[d["subject"] == subject].iloc[0]
    t1_img = nib.as_closest_canonical(nib.load(row["t1_path"]))
    t1 = np.asarray(t1_img.dataobj, dtype=np.float32)
    mk = np.asarray(nib.as_closest_canonical(nib.load(row["mask_path"])).dataobj) > 0.5
    z = pick_slice(t1, mk)
    sl, ml = t1[:, :, z].T, mk[:, :, z].T
    rows, cols = np.nonzero(sl > 0)
    sl = sl[rows.min() - 4:rows.max() + 5, cols.min() - 4:cols.max() + 5]
    ml = ml[rows.min() - 4:rows.max() + 5, cols.min() - 4:cols.max() + 5]
    vmax = np.percentile(t1[t1 > 0], 99.5)
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.2), facecolor="#000000")
    axes[0].imshow(sl, cmap="gray", origin="lower", vmin=0, vmax=vmax)
    axes[1].imshow(ml, cmap="gray", origin="lower", vmin=0, vmax=1, interpolation="nearest")
    axes[2].imshow(sl, cmap="gray", origin="lower", vmin=0, vmax=vmax)
    axes[2].imshow(np.ma.masked_where(~ml, ml), cmap=matplotlib.colors.ListedColormap([ORANGE]),
                   origin="lower", alpha=0.75, interpolation="nearest")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "00_one_subject.png", dpi=150, facecolor="#000000")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    d, H, edges = load()
    s = site_summary(d)
    fig_site_fingerprint(d, s)
    fig_histograms(d, H, edges, s)
    fig_histograms(d, H, edges, s, sites=["SOOP", "R009", "R038", "R040", "R031", "R008", "R032", "R029"],
                   fname="02b_intensity_histograms_selected.png", ncol=4, panel_h=2.6)
    fig_lesions(d)
    fig_time(d)
    fig_examples(d, s)
    fig_anomalies(d)
    fig_one_subject(d)
    print(f"wrote {OUT / 'site_summary.csv'} and figures in {FIG}")


if __name__ == "__main__":
    main()
