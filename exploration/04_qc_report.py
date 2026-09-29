"""Step 7 (contrôle qualité): prints a QC + key-numbers report from outputs/subjects.parquet
and writes outputs/qc_flags.csv (one row per flagged subject).

Usage:  python exploration/04_qc_report.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs"


def main():
    d = pd.read_parquet(OUT / "subjects.parquet")
    err = d["error"].notna() if "error" in d else pd.Series(False, index=d.index)

    flags = {
        "read_error": err,
        "empty_metadata": d["meta_rows"] == 0,
        "grid_mismatch_t1_mask": ~d["same_grid"].fillna(False).astype(bool),
        # 2 values with max ~1 is binary (some masks store 1 as 0.99999998 because of header scaling)
        "mask_not_binary": ~((d["mask_n_values"] <= 2) & (np.isclose(d["mask_max"], 1) | (d["mask_max"] == 0))),
        "empty_mask": d["lesion_vox"] == 0,
        "lesion_under_0.1ml": d["lesion_ml"] < 0.1,
        "lesion_over_300ml": d["lesion_ml"] > 300,
        "lesion_20pct_outside_brain": d["lesion_outside_brain_frac"] > 0.2,
        "negative_intensities": d["n_negative"] > 0,
        "background_not_zero": d["bg_not_zero"].eq(True) if "bg_not_zero" in d else False,
        "brain_under_800ml": d["brain_ml"] < 800,
        "brain_over_2000ml": d["brain_ml"] > 2000,
        "anisotropy_over_2": d["anisotropy"] > 2,
        "negative_days_post_stroke": d["days_post_stroke"] < 0,
    }
    f = pd.DataFrame(flags).fillna(False)
    print(f"{len(d)} subjects, {d['site'].nunique()} sites, {int(err.sum())} read errors\n")
    print("QC flags (count, sites involved):")
    for name in f.columns:
        idx = f[name]
        sites = d.loc[idx, "site"].value_counts()
        print(f"  {name:<30} {int(idx.sum()):>5}   {dict(sites.head(6))}")
    flagged = d.loc[f.any(axis=1), ["site", "subject"]].join(f[f.any(axis=1)])
    flagged.to_csv(OUT / "qc_flags.csv", index=False)

    print("\nGeometry:")
    print("  orientations:", d["orientation"].value_counts().to_dict())
    print("  T1 vs mask orientation differs:", int((d["orientation"] != d["mask_orientation"]).sum()))
    print("  T1 dtypes:", d["t1_dtype"].value_counts().to_dict())
    print("  voxel volume mm3 (p5/p50/p95):", np.percentile(d["voxel_mm3"], [5, 50, 95]).round(3).tolist())
    print("  slice thickness (max voxel side) mm:", d["vox_max"].round(1).value_counts().head(8).to_dict())

    print("\nBrain:")
    print("  volume mL (p5/p50/p95):", np.percentile(d["brain_ml"], [5, 50, 95]).round(0).tolist())
    print("  raw intensity p99 (p5/p50/p95 across scans):", np.percentile(d["int_p99"], [5, 50, 95]).round(1).tolist())
    site_p99 = d.groupby("site")["int_p99"].median().sort_values()
    print("  site median p99, lowest 3:", site_p99.head(3).round(1).to_dict(), " highest 3:", site_p99.tail(3).round(1).to_dict())

    print("\nLesions:")
    q = d["lesion_ml"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(2)
    print("  volume mL p10/p25/p50/p75/p90:", q.tolist(), " max:", round(d["lesion_ml"].max(), 1))
    print("  components median:", d["n_components"].median(), " single lesion:", f"{(d['n_components'] == 1).mean():.1%}",
          " >=10 components:", int((d["n_components"] >= 10).sum()))
    print("  lesion/surrounding-tissue intensity (median):", round(d["lesion_vs_shell"].median(), 3),
          " -> by site, highest (least visible) 5:",
          d.groupby("site")["lesion_vs_shell"].median().sort_values().tail(5).round(2).to_dict())
    print("  lesion on patient's right:", f"{(d['lesion_dx_mm'] > 0).mean():.1%}")


if __name__ == "__main__":
    main()
