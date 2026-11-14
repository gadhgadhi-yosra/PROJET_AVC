"""Inventaire des entrées C1 : une ligne par sujet de splits.json, écrite dans un CSV (rien dans la console).

    .venv-dl\\Scripts\\python -m adaptation.check_inputs                      # data/processed, 4 processus
    python -m adaptation.check_inputs --data-root /kaggle/working/processed --output stats.csv

Pour chaque volume (t1, mask, brain) : extension, type stocké sur disque, `scl_slope` et `scl_inter` de
l'en-tête, valeurs brutes (sans la pente) et valeurs lues par le chargeur d'entraînement (`read_image`,
`read_label` de generalization.data_processed). Pour t1 : min, max, moyenne, NaN, Inf, voxels de
|z| > 10 et > 100, moyenne et écart-type dans le cerveau, voxels non nuls hors du cerveau. Pour les
masques : valeurs distinctes, non binaires, lésion hors du cerveau. Le CSV se relit avec pandas.
"""
import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generalization.calibration import DEFAULT_SPLITS, load_splits  # noqa: E402
from generalization.data_processed import (DEFAULT_PROCESSED, VOLUMES, read_image, read_label,  # noqa: E402
                                           volume_path)

DEFAULT_OUTPUT = REPO_ROOT / "adaptation" / "runs" / "diagnostic" / "inventaire_c1.csv"


def header_scaling(image):
    """(`scl_slope`, `scl_inter`) du fichier.

    nibabel les retire de `image.header` au chargement (remis à NaN) et les garde dans `image.dataobj` :
    ce sont eux que `get_fdata` et `np.asarray(dataobj)` appliquent.
    """
    return float(image.dataobj.slope), float(image.dataobj.inter)


def volume_info(path, prefix):
    image = nib.load(path)
    slope, inter = header_scaling(image)
    raw = np.asarray(image.dataobj.get_unscaled())
    return image, {f"{prefix}_extension": "".join(Path(path).suffixes), f"{prefix}_dtype": str(image.get_data_dtype()),
                   f"{prefix}_slope": slope, f"{prefix}_inter": inter,
                   f"{prefix}_brut_min": float(raw.min()), f"{prefix}_brut_max": float(raw.max())}


def subject_row(task):
    """Statistiques d'un sujet ; `statut` = ok, manquant (volumes absents) ou erreur (message)."""
    root, case = task
    row = dict(case_id=case["case_id"], site=case["site"], subject=case["subject"], decoupage=case["decoupage"],
               role_pli0=case["role_pli0"])
    folder = Path(root) / case["subject"]
    paths = {name: volume_path(folder, name) for name in VOLUMES}
    if not all(path.is_file() for path in paths.values()):
        return {**row, "statut": "manquant"}
    try:
        _, info = volume_info(paths["t1"], "t1")
        row.update(info)
        t1 = read_image(paths["t1"])[0][0].numpy()       # exactement ce que reçoit le réseau
        brain = read_label(paths["brain"])[0][0].numpy().astype(bool)
        mask = read_label(paths["mask"])[0][0].numpy().astype(bool)
        for name in ("mask", "brain"):
            image, info = volume_info(paths[name], name)
            row.update(info)
            values = np.unique(np.asarray(image.dataobj))
            row[f"{name}_valeurs_distinctes"] = len(values)
            row[f"{name}_non_binaire"] = bool(np.any(~np.isclose(values, 0) & ~np.isclose(values, 1, atol=1e-6)))
        finite = np.isfinite(t1)
        inside = t1[brain & finite]
        row.update(forme="x".join(map(str, t1.shape)), t1_lu_dtype=str(t1.dtype),
                   t1_min=float(t1[finite].min()) if finite.any() else float("nan"),
                   t1_max=float(t1[finite].max()) if finite.any() else float("nan"),
                   t1_moyenne=float(t1[finite].mean()) if finite.any() else float("nan"),
                   t1_nan=int(np.isnan(t1).sum()), t1_inf=int(np.isinf(t1).sum()),
                   t1_abs_sup_10=int((np.abs(t1[finite]) > 10).sum()),
                   t1_abs_sup_100=int((np.abs(t1[finite]) > 100).sum()),
                   t1_moyenne_cerveau=float(inside.mean()) if inside.size else float("nan"),
                   t1_ecart_type_cerveau=float(inside.std()) if inside.size else float("nan"),
                   t1_non_nul_hors_cerveau=int((t1[~brain] != 0).sum()),
                   voxels_cerveau=int(brain.sum()), voxels_lesion=int(mask.sum()),
                   lesion_hors_cerveau=int((mask & ~brain).sum()), statut="ok")
    except Exception as error:  # un sujet illisible ne doit pas arrêter l'inventaire
        row.update(statut="erreur", erreur=f"{type(error).__name__}: {error}")
    return row


def cases_of(splits):
    fold0 = {role: set(splits["folds"][0][role]) for role in ("train", "val", "id_test")}
    split_of = {case: split for split in ("train", "id_test", "ood_test") for case in splits[split]}
    for case in splits["cases"]:
        role = next((r for r, ids in fold0.items() if case["case_id"] in ids), "ood_test"
                    if split_of.get(case["case_id"]) == "ood_test" else "")
        yield dict(case_id=case["case_id"], site=case["site"], subject=case["subject"],
                   decoupage=split_of.get(case["case_id"], ""), role_pli0=role)


def inventory(root, splits, output, workers=4):
    cases = list(cases_of(load_splits(splits)))
    tasks = [(str(root), case) for case in cases]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            rows = list(pool.map(subject_row, tasks, chunksize=8))
    else:
        rows = [subject_row(task) for task in tasks]
    columns = list(dict.fromkeys(key for row in rows for key in row))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        writer.writerows(rows)
    return output, len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=str(DEFAULT_PROCESSED))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output, n = inventory(args.data_root, args.splits, args.output, args.workers)
    print(f"{n} sujets -> {output}")


if __name__ == "__main__":
    main()
