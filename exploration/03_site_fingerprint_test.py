"""Quick "empreinte du centre" check (PDF section 5.5) on hand-crafted features, before any deep model.

Can a simple classifier tell which site a scan comes from? Three feature sets, sites with >= MIN_N subjects:
  A. shape of the in-brain intensity histogram only (each scan scaled by its own p99)
  B. raw intensity scale (log percentiles) - what z-score normalization removes
  C. acquisition geometry (voxel size, matrix, orientation) - what resampling to 1 mm removes
Reports balanced accuracy (5-fold stratified CV) against chance. Writes outputs/site_fingerprint_test.txt.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent / "outputs"
MIN_N = 20


def main():
    d = pd.read_parquet(OUT / "subjects.parquet")
    h = np.load(OUT / "histograms.npz", allow_pickle=True)
    H = h["hist"]
    ok = ~np.isnan(H).any(axis=1)
    counts = d.loc[ok, "site"].value_counts()
    keep_sites = counts[counts >= MIN_N].index
    m = ok & d["site"].isin(keep_sites).to_numpy()
    y = d.loc[m, "site"].to_numpy()

    pct = [c for c in d.columns if c.startswith("int_p")]
    geo = pd.get_dummies(d.loc[m, "orientation"], prefix="or").astype(float)
    feats = {
        "A. histogram shape (scale-free)": H[m, :126],  # bins up to 1.25 x p99
        "B. raw intensity scale (log percentiles)": np.log1p(d.loc[m, pct].clip(lower=0).to_numpy()),
        "C. geometry (voxel size, matrix, orientation)": np.hstack([
            d.loc[m, ["vox_x", "vox_y", "vox_z", "dim_x", "dim_y", "dim_z"]].to_numpy(float), geo.to_numpy()]),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    models = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0)),
        "random forest": RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1),
    }
    lines = [f"{m.sum()} scans from {len(keep_sites)} sites with >= {MIN_N} subjects "
             f"(chance balanced accuracy = {1 / len(keep_sites):.3f})", ""]
    for fname, X in feats.items():
        for mname, model in models.items():
            s = cross_val_score(model, X, y, cv=cv, scoring="balanced_accuracy")
            lines.append(f"{fname:<48} {mname:<14} balanced acc = {s.mean():.3f} ± {s.std():.3f}")
    txt = "\n".join(lines)
    print(txt)
    (OUT / "site_fingerprint_test.txt").write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
