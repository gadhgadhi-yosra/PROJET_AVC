"""Build the nnU-Net v2 raw dataset (Dataset601_ATLAS3) + the centre-based splits (PDF section 3.3).

Split design (one training run gives the central measure Dice(ID) - Dice(OOD)):
  - OOD test   : whole centres removed from training -> nnUNet_raw/.../imagesTs + labelsTs
  - ID val     : other patients from the training centres -> fold 0 of splits_final.json
  - train      : the rest
Folds 1-4 are also written (stratified by centre) so more folds can be trained later.

Data fixes (exploration/outputs/qc_report.txt):
  - R032 excluded (inconsistent header spacing, 2 T1/mask grid mismatches)
  - R029: background is a tiny non-zero offset -> set to exactly 0 so nnU-Net's nonzero crop works
  - masks thresholded at > 0.5 (41 masks store 1 as 0.99999998) and written as uint8
  - scans > 64 M voxels (105 SOOP, 0.25-0.44 mm in-plane) resampled to 1 mm here (PDF step 3): at native
    resolution one scan is 1.3 GB in float32 and nnU-Net's fingerprint/preprocessing runs out of 16 GB RAM
  - other images are hard-linked (no extra disk); mixed orientations handled by NibabelIOWithReorient (-> RAS)
Writes outputs/split.csv (subject, site, set, fold).
"""
import json
import os
import shutil
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_to_output
from scipy import ndimage
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parents[1]  # nnunet/ root
PROJ = HERE.parent
DATASET = "Dataset601_ATLAS3"
RAW = HERE / "nnUNet_raw" / DATASET
PREP = HERE / "nnUNet_preprocessed" / DATASET
OUT = HERE / "outputs"
EXCLUDE_SITES = {"R032"}
BG_FIX_SITES = {"R029"}
OOD_FRAC = 0.20          # share of subjects in held-out centres
OOD_MAX_SITE = 60        # don't let one big centre (SOOP=169, R009=111) be the whole OOD set
SEED = 0
BIG_NVOX = 64e6          # 105 SOOP scans (up to 1008x1008x348 at 0.25 mm) -> pre-resampled to 1 mm


def case_id(subject):
    return subject.replace("sub-", "")  # e.g. r001s001


def choose_ood_sites(d):
    counts = d.groupby("site").size()
    rng = np.random.default_rng(SEED)
    cand = [s for s in rng.permutation(sorted(counts.index)) if counts[s] <= OOD_MAX_SITE]
    target, picked, n = OOD_FRAC * len(d), [], 0
    for s in cand:
        if n >= target:
            break
        picked.append(s)
        n += counts[s]
    return sorted(picked)


def is_big(row):
    return row["dim_x"] * row["dim_y"] * row["dim_z"] > BIG_NVOX


def write_resampled_case(row, img_dst, lbl_dst):
    """Very high-res scans -> 1 mm isotropic (anti-aliased linear for T1, linear + > 0.5 for the mask)."""
    t1_img = nib.load(row["t1_path"])
    zooms = np.array(t1_img.header.get_zooms()[:3], float)
    t1 = np.asarray(t1_img.dataobj, dtype=np.float32)
    sigma = np.maximum(1.0 / zooms - 1.0, 0) / 2.0  # skimage-style anti-aliasing for the downsampling factor
    t1 = ndimage.gaussian_filter(t1, sigma, output=np.float32)
    out = resample_to_output(nib.Nifti1Image(t1, t1_img.affine), (1.0, 1.0, 1.0), order=1)
    del t1
    nib.save(nib.Nifti1Image(np.asarray(out.dataobj, dtype=np.float32), out.affine), img_dst)
    mk = (np.asarray(nib.load(row["mask_path"]).dataobj) > 0.5).astype(np.float32)
    lbl = resample_to_output(nib.Nifti1Image(mk, t1_img.affine), (1.0, 1.0, 1.0), order=1)
    hdr = nib.load(img_dst).header.copy()
    hdr.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image((np.asarray(lbl.dataobj) > 0.5).astype(np.uint8), out.affine, hdr), lbl_dst)


def write_case(args):
    row, img_dir, lbl_dir = args
    cid = case_id(row["subject"])
    img_dst = img_dir / f"{cid}_0000.nii.gz"
    lbl_dst = lbl_dir / f"{cid}.nii.gz"
    if is_big(row):
        if img_dst.exists() and img_dst.stat().st_nlink > 1:  # earlier run hard-linked the native scan
            img_dst.unlink()
            lbl_dst.unlink(missing_ok=True)
        if not (img_dst.exists() and lbl_dst.exists()):
            write_resampled_case(row, img_dst, lbl_dst)
        return cid
    t1_img = nib.load(row["t1_path"])
    if not img_dst.exists():
        if row["site"] in BG_FIX_SITES:
            t1 = np.asarray(t1_img.dataobj, dtype=np.float32)
            t1[t1 == row["background_value"]] = 0.0
            out = nib.Nifti1Image(t1, t1_img.affine, t1_img.header)
            out.set_data_dtype(np.float32)
            nib.save(out, img_dst)
        else:
            try:
                os.link(row["t1_path"], img_dst)
            except OSError:
                shutil.copy2(row["t1_path"], img_dst)
    if not lbl_dst.exists():
        mk = np.asarray(nib.load(row["mask_path"]).dataobj)
        if mk.ndim > 3:
            mk = mk[..., 0]
        # copy the T1 header: building from the affine alone recomputes pixdim (0.99999 vs 1.0 mm),
        # which fails nnU-Net's exact image/label spacing check
        hdr = t1_img.header.copy()
        hdr.set_data_dtype(np.uint8)
        hdr.set_slope_inter(1, 0)
        nib.save(nib.Nifti1Image((mk > 0.5).astype(np.uint8), t1_img.affine, hdr), lbl_dst)
    return cid


def main():
    d = pd.read_parquet(PROJ / "exploration" / "outputs" / "subjects.parquet")
    d = d[~d["site"].isin(EXCLUDE_SITES)].reset_index(drop=True)
    d["case"] = d["subject"].map(case_id)

    ood_sites = choose_ood_sites(d)
    d["set"] = np.where(d["site"].isin(ood_sites), "ood_test", "train")
    pool = d[d["set"] == "train"]
    d["fold"] = -1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # centres with < 5 subjects
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        for k, (_, va) in enumerate(skf.split(pool, pool["site"])):
            d.loc[pool.index[va], "fold"] = k
    d.loc[(d["set"] == "train") & (d["fold"] == 0), "set"] = "id_val"

    dirs = {s: (RAW / f"images{s}", RAW / f"labels{s}") for s in ("Tr", "Ts")}
    for a, b in dirs.values():
        a.mkdir(parents=True, exist_ok=True)
        b.mkdir(parents=True, exist_ok=True)
    jobs = [(r, *dirs["Ts" if r["set"] == "ood_test" else "Tr"]) for r in d.to_dict("records")]
    small = [j for j in jobs if not is_big(j[0])]
    big = [j for j in jobs if is_big(j[0])]
    with ProcessPoolExecutor(6) as ex:
        for i, _ in enumerate(ex.map(write_case, small, chunksize=8), 1):
            if i % 200 == 0 or i == len(small):
                print(f"  {i}/{len(small)} cases written", flush=True)
    with ProcessPoolExecutor(2) as ex:  # up to ~6 GB RAM each
        for i, _ in enumerate(ex.map(write_case, big), 1):
            if i % 20 == 0 or i == len(big):
                print(f"  {i}/{len(big)} very high-res cases resampled to 1 mm", flush=True)

    n_tr = int((d["set"] != "ood_test").sum())
    (RAW / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "T1"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": n_tr,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIOWithReorient",
        "name": "ATLAS3 / ISLES'26 raw T1 (R032 excluded)",
    }, indent=2), encoding="utf-8")

    tr = d[d["set"] != "ood_test"]
    splits = [{"train": sorted(tr.loc[tr["fold"] != k, "case"]), "val": sorted(tr.loc[tr["fold"] == k, "case"])}
              for k in range(5)]
    PREP.mkdir(parents=True, exist_ok=True)
    (PREP / "splits_final.json").write_text(json.dumps(splits, indent=1), encoding="utf-8")

    OUT.mkdir(exist_ok=True)
    d[["case", "subject", "site", "set", "fold", "days_post_stroke", "chronicity", "lesion_ml"]].to_csv(
        OUT / "split.csv", index=False)
    print(f"OOD centres ({len(ood_sites)}): {', '.join(ood_sites)}")
    print(d["set"].value_counts().to_string())
    print(f"ID/OOD centres: {pool['site'].nunique()} / {len(ood_sites)}")


if __name__ == "__main__":
    main()
