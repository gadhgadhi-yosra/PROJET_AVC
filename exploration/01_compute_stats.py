"""Step 1 (inventory) + per-subject statistics for ATLAS3_Training_Raw (ISLES'26 training set).

Walks data/ATLAS3_Training_Raw/<SITE>/sub-*/ses-*/anat/, reads the metadata CSV and both NIfTI
volumes of every session, and writes:
  outputs/subjects.parquet   one row per session: metadata, geometry, lesion + intensity stats, QC flags
  outputs/histograms.npz     in-brain intensity histogram per session (intensity / in-brain p99)

Usage:  python exploration/01_compute_stats.py [--workers N] [--limit N]
"""
import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent / "data" / "ATLAS3_Training_Raw"
OUT = Path(__file__).resolve().parent / "outputs"
HIST_EDGES = np.linspace(0.0, 1.5, 151)  # normalized intensity (value / p99 of brain)


def find_sessions():
    sessions = []
    for anat in sorted(ROOT.glob("*/sub-*/ses-*/anat")):
        t1 = sorted(anat.glob("*_desc-brain_T1w.nii.gz"))
        mask = sorted(anat.glob("*_label-lesion_*mask.nii.gz"))
        meta = sorted(anat.glob("*_metadata.csv"))
        nvox = int(np.prod(nib.load(t1[0]).shape)) if t1 else 0
        sessions.append(dict(
            site=anat.parts[-4], subject=anat.parts[-3], session=anat.parts[-2], nvox=nvox,
            t1_path=str(t1[0]) if t1 else None, mask_path=str(mask[0]) if mask else None,
            meta_path=str(meta[0]) if meta else None,
            n_t1=len(t1), n_mask=len(mask), n_meta=len(meta),
        ))
    return sessions


def read_metadata(path):
    if path is None:
        return {"meta_rows": 0}
    df = pd.read_csv(path)
    row = df.iloc[0].to_dict() if len(df) else {}
    return {
        "meta_rows": len(df),
        "atlas_dataset": row.get("ATLAS2_DATASET"),
        "days_post_stroke": row.get("DAYS_POST_STROKE"),
        "chronicity": row.get("CHRONICITY"),
        "meta_site": row.get("SITE"),
    }


def process(s):
    r = {k: v for k, v in s.items()}
    hist = np.full(len(HIST_EDGES) - 1, np.nan, dtype=np.float32)
    t0 = time.perf_counter()
    try:
        r.update(read_metadata(s["meta_path"]))
        r["t1_mb"] = Path(s["t1_path"]).stat().st_size / 1e6
        t1_img, mask_img = nib.load(s["t1_path"]), nib.load(s["mask_path"])

        # --- geometry (header only) ---
        zooms = [float(z) for z in t1_img.header.get_zooms()[:3]]
        r.update(
            dim_x=t1_img.shape[0], dim_y=t1_img.shape[1], dim_z=t1_img.shape[2], ndim=len(t1_img.shape),
            vox_x=zooms[0], vox_y=zooms[1], vox_z=zooms[2],
            vox_min=min(zooms), vox_max=max(zooms), voxel_mm3=float(np.prod(zooms)),
            anisotropy=max(zooms) / min(zooms),
            fov_x_mm=t1_img.shape[0] * zooms[0], fov_y_mm=t1_img.shape[1] * zooms[1],
            fov_z_mm=t1_img.shape[2] * zooms[2],
            orientation="".join(nib.aff2axcodes(t1_img.affine)),
            mask_orientation="".join(nib.aff2axcodes(mask_img.affine)),
            t1_dtype=str(t1_img.get_data_dtype()), mask_dtype=str(mask_img.get_data_dtype()),
            same_grid=bool(t1_img.shape == mask_img.shape and np.allclose(t1_img.affine, mask_img.affine, atol=1e-3)),
            qform_code=int(t1_img.header["qform_code"]), sform_code=int(t1_img.header["sform_code"]),
        )

        # --- voxel data ---
        t1 = np.asarray(t1_img.dataobj, dtype=np.float32)
        mk = np.asarray(mask_img.dataobj)
        if t1.ndim > 3:
            t1 = t1[..., 0]
        if mk.ndim > 3:
            mk = mk[..., 0]
        mask_vals = np.unique(mk)
        r["mask_n_values"] = len(mask_vals)
        # some masks store 1 as 0.99999998 (header scaling): binary = two values, close to 0 and 1
        r["mask_binary"] = bool(len(mask_vals) <= 2 and np.all(np.isclose(mask_vals, 0) | np.isclose(mask_vals, 1)))
        r["mask_max"] = float(mask_vals.max())
        r["t1_min"], r["t1_max"] = float(t1.min()), float(t1.max())
        r["n_negative"] = int((t1 < 0).sum())
        fov_vox = t1.size

        # background value = most frequent value. Usually exactly 0, but some scans (R029) carry a tiny
        # scaling offset (e.g. 0.0007), so "t1 > 0" would select the whole field of view.
        u, c = np.unique(t1[::4, ::4, ::4], return_counts=True)
        bg = float(u[c.argmax()])
        r["background_value"] = bg
        r["bg_not_zero"] = bg != 0.0
        # crop to the brain+lesion bounding box (with a margin) to keep memory low on huge volumes
        if t1.shape == mk.shape:
            keep = (t1 > bg) | (mk > 0)
            if keep.any():
                sl = []
                for ax in range(3):
                    nz = np.nonzero(keep.any(axis=tuple(a for a in range(3) if a != ax)))[0]
                    sl.append(slice(max(nz[0] - 4, 0), nz[-1] + 5))
                sl = tuple(sl)
                offset = np.array([s.start for s in sl], dtype=float)
                t1, mk = t1[sl].copy(), mk[sl].copy()
            else:
                offset = np.zeros(3)
            del keep
        else:
            offset = np.zeros(3)

        brain = t1 > bg  # scans are skull-stripped: everything above the background value is brain
        lesion = (mk > 0.5) if mk.shape == t1.shape else np.zeros_like(brain)
        n_brain, n_les = int(brain.sum()), int(lesion.sum())
        vml = r["voxel_mm3"] / 1000.0
        r["brain_ml"] = n_brain * vml
        r["brain_frac_of_fov"] = n_brain / fov_vox
        r["lesion_vox"] = n_les
        r["lesion_ml"] = n_les * vml
        r["lesion_frac_of_brain"] = n_les / max(n_brain, 1)
        r["lesion_outside_brain_frac"] = float((lesion & ~brain).sum() / n_les) if n_les else np.nan

        # lesion components (26-connectivity), as in the ISLES lesion-wise metrics
        if n_les:
            lab, ncomp = ndimage.label(lesion, structure=np.ones((3, 3, 3)))
            sizes = np.bincount(lab.ravel())[1:]
            r["n_components"] = int(ncomp)
            r["largest_comp_ml"] = float(sizes.max() * vml)
            r["n_components_ge_10vox"] = int((sizes >= 10).sum())
        else:
            r["n_components"], r["largest_comp_ml"], r["n_components_ge_10vox"] = 0, 0.0, 0

        # brain bounding box (for the crop step) in mm
        if n_brain:
            for ax, name in enumerate("xyz"):
                nz = np.nonzero(brain.any(axis=tuple(a for a in range(3) if a != ax)))[0]
                r[f"bbox_{name}_mm"] = (nz[-1] - nz[0] + 1) * zooms[ax]
            # lesion side: lesion centroid vs brain centroid along world x (RAS+: >0 = right)
            if n_les:
                aff = t1_img.affine
                c_brain = aff[:3, :3] @ (np.array(ndimage.center_of_mass(brain)) + offset) + aff[:3, 3]
                c_les = aff[:3, :3] @ (np.array(ndimage.center_of_mass(lesion)) + offset) + aff[:3, 3]
                r["lesion_dx_mm"] = float(c_les[0] - c_brain[0])
                r["lesion_dz_mm"] = float(c_les[2] - c_brain[2])

        # --- intensity inside brain ---
        vals = t1[brain]
        if vals.size:
            p = np.percentile(vals, [1, 5, 25, 50, 75, 95, 99, 99.9])
            for k, v in zip(["p01", "p05", "p25", "p50", "p75", "p95", "p99", "p999"], p):
                r[f"int_{k}"] = float(v)
            r["int_mean"], r["int_std"] = float(vals.mean()), float(vals.std())
            r["int_cv"] = r["int_std"] / r["int_mean"] if r["int_mean"] else np.nan
            r["int_n_unique"] = int(np.unique(vals[:: max(1, vals.size // 200_000)]).size)
            h, _ = np.histogram(vals / p[6], bins=HIST_EDGES)
            hist = (h / vals.size).astype(np.float32)
            if n_les:
                les_vals = t1[lesion & brain]
                if les_vals.size:
                    # lesion contrast: <1 means the lesion is darker than typical brain tissue
                    r["lesion_int_median_rel"] = float(np.median(les_vals) / p[3])
                    # contrast vs a 3-voxel shell of tissue around the lesion
                    shell = ndimage.binary_dilation(lesion, iterations=3) & ~lesion & brain
                    if shell.any():
                        r["lesion_vs_shell"] = float(np.median(les_vals) / np.median(t1[shell]))
    except Exception as e:  # keep going; the QC table will list failures
        r["error"] = f"{type(e).__name__}: {e}"
    r["seconds"] = time.perf_counter() - t0
    return r, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--big-workers", type=int, default=1, help="workers for volumes > --big-vox voxels")
    ap.add_argument("--big-vox", type=int, default=60_000_000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sites", nargs="*", default=None, help="only these site folders")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    sessions = find_sessions()
    if args.sites:
        sessions = [s for s in sessions if s["site"] in args.sites]
    sessions = sessions[: args.limit]
    small = [s for s in sessions if s["nvox"] <= args.big_vox]
    big = [s for s in sessions if s["nvox"] > args.big_vox]
    print(f"{len(sessions)} sessions under {ROOT}: {len(small)} normal ({args.workers} workers), "
          f"{len(big)} large > {args.big_vox:,} voxels ({args.big_workers} workers)")

    t0 = time.perf_counter()
    rows, hists = [], []
    for batch, workers in [(small, args.workers), (big, args.big_workers)]:
        if not batch:
            continue
        with Pool(workers) as pool:
            for r, h in pool.imap(process, batch, chunksize=1):
                rows.append(r)
                hists.append(h)
                if len(rows) % 100 == 0 or len(rows) == len(sessions):
                    print(f"  {len(rows)}/{len(sessions)}  ({time.perf_counter() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    H = np.stack(hists)
    suffix = "" if args.limit is None else f"_limit{args.limit}"
    if args.sites and args.limit is None and (OUT / "subjects.parquet").exists():
        # recompute only some sites: replace their rows in the existing full outputs
        old = pd.read_parquet(OUT / "subjects.parquet")
        old_h = np.load(OUT / "histograms.npz", allow_pickle=True)["hist"]
        keep = ~old["subject"].isin(df["subject"]).to_numpy()
        df = pd.concat([old[keep], df], ignore_index=True)
        H = np.concatenate([old_h[keep], H])
        order = np.lexsort((df["subject"].to_numpy(), df["site"].to_numpy()))
        df, H = df.iloc[order].reset_index(drop=True), H[order]
    df.to_parquet(OUT / f"subjects{suffix}.parquet", index=False)
    np.savez_compressed(OUT / f"histograms{suffix}.npz", hist=H, edges=HIST_EDGES,
                        subject=df["subject"].to_numpy(), site=df["site"].to_numpy())
    n_err = int(df["error"].notna().sum()) if "error" in df else 0
    print(f"done in {time.perf_counter() - t0:.0f}s, {n_err} errors -> {OUT}")


if __name__ == "__main__":
    main()
