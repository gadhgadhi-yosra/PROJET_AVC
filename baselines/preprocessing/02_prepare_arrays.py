"""Fast common preprocessing for the quick baselines (2.5D CNN + classical ML), same splits as nnU-Net.

Per case (from nnUNet_raw, so the 01 fixes apply): reorient to RAS, resample to 1 mm (T1 linear, mask
linear + > 0.5), crop to the brain bounding box, z-score inside the brain (background = 0, clip +-5).
Saved as data_fast/<case>_img.npy (float16) and <case>_seg.npy (uint8), axes (z, y, x) = axial slices first.
Metrics are then computed in this common 1 mm space. Writes data_fast/index.csv.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

HERE = Path(__file__).resolve().parents[1]  # nnunet/ root
RAW = HERE / "nnUNet_raw" / "Dataset601_ATLAS3"
OUT = HERE / "data_fast"


def prep(args):
    case, s = args
    sub = "Ts" if s == "ood_test" else "Tr"
    img_dst, seg_dst = OUT / f"{case}_img.npy", OUT / f"{case}_seg.npy"
    if img_dst.exists() and seg_dst.exists():
        try:
            seg = np.load(seg_dst)
            nz, ny, nx = seg.shape
            return {"case": case, "nz": nz, "ny": ny, "nx": nx, "lesion_vox": int(seg.sum())}
        except (ValueError, OSError):  # partially written by an interrupted run
            pass
    im =nib.as_closest_canonical(nib.load(RAW / f"images{sub}" / f"{case}_0000.nii.gz"))
    lb = nib.as_closest_canonical(nib.load(RAW / f"labels{sub}" / f"{case}.nii.gz"))
    x = np.asarray(im.dataobj, dtype=np.float32)
    y = np.asarray(lb.dataobj).astype(np.float32)
    z = np.array(im.header.get_zooms()[:3], float)
    if np.abs(z - 1).max() > 0.05:
        x = ndimage.zoom(x, z, order=1)
        y = ndimage.zoom(y, z, order=1)
    y = y > 0.5
    brain = x != 0
    idx = np.argwhere(brain | y)
    lo, hi = np.maximum(idx.min(0) - 2, 0), idx.max(0) + 3
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    x, y, brain = x[sl], y[sl], brain[sl]
    v = x[brain]
    x = np.where(brain, np.clip((x - v.mean()) / (v.std() + 1e-6), -5, 5), 0)
    np.save(img_dst, x.transpose(2, 1, 0).astype(np.float16))
    np.save(seg_dst, y.transpose(2, 1, 0).astype(np.uint8))
    return {"case": case, "nz": x.shape[2], "ny": x.shape[1], "nx": x.shape[0], "lesion_vox": int(y.sum())}


def main():
    OUT.mkdir(exist_ok=True)
    split = pd.read_csv(HERE / "outputs" / "split.csv")
    jobs = list(zip(split["case"], split["set"]))
    rows = []
    with ProcessPoolExecutor(7) as ex:
        for i, r in enumerate(ex.map(prep, jobs, chunksize=4), 1):
            rows.append(r)
            if i % 100 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)
    idx = split.merge(pd.DataFrame(rows), on="case")
    idx.to_csv(OUT / "index.csv", index=False)
    print(idx.groupby("set")[["nz", "ny", "nx"]].median())


if __name__ == "__main__":
    main()
