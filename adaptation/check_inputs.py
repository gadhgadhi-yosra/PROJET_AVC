"""Inventaire des entrées C1 : une ligne par sujet de splits.json, ajoutée au CSV dès qu'elle est prête.

    .venv-dl\\Scripts\\python -m adaptation.check_inputs                      # data/processed, 4 processus
    python -m adaptation.check_inputs --data-root /kaggle/working/processed --output stats.csv

Reprise : relancer la même commande ; les sujets déjà présents dans le CSV (hors `erreur`) sont sautés,
les lignes `erreur` sont retirées puis recalculées. `--recommencer` repart d'un CSV vide.

Pour chaque volume (t1, mask, brain) : extension, type stocké sur disque, `scl_slope` et `scl_inter`
(lus dans `dataobj`), valeurs stockées (sans la pente) et valeurs lues par le chargeur d'entraînement
(`read_image`, `read_label` de generalization.data_processed). Pour t1 : min, max, moyenne, NaN, Inf,
voxels de |z| > 10 et > 100, moyenne et écart-type dans le cerveau, voxels non nuls hors du cerveau.
Pour les masques : valeurs distinctes, non binaires, lésion hors du cerveau. Chaque processus ne
garde qu'un sujet en mémoire à la fois ; seules les statistiques remontent.
"""
import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generalization.calibration import DEFAULT_SPLITS, load_splits  # noqa: E402
from generalization.data_processed import (DEFAULT_PROCESSED, VOLUMES, read_image, read_label,  # noqa: E402
                                           volume_path)

DEFAULT_OUTPUT = REPO_ROOT / "adaptation" / "runs" / "diagnostic" / "inventaire_c1.csv"
MAX_WORKERS = 4

COLUMNS = (["case_id", "site", "subject", "decoupage", "role_pli0", "statut", "erreur", "forme"]
           + [f"{v}_{k}" for v in VOLUMES for k in ("extension", "dtype", "slope", "inter", "brut_min", "brut_max")]
           + ["t1_lu_dtype", "t1_min", "t1_max", "t1_moyenne", "t1_nan", "t1_inf", "t1_abs_sup_10",
              "t1_abs_sup_100", "t1_moyenne_cerveau", "t1_ecart_type_cerveau", "t1_non_nul_hors_cerveau",
              "mask_valeurs_distinctes", "mask_non_binaire", "brain_valeurs_distinctes", "brain_non_binaire",
              "voxels_cerveau", "voxels_lesion", "lesion_hors_cerveau"])


def header_info(path, prefix):
    """En-tête et valeurs stockées (sans pente) ; `scl_slope`/`scl_inter` viennent de `dataobj`.

    nibabel les retire de `image.header` au chargement (remis à NaN) : ce sont ceux de `dataobj`
    que `get_fdata` et `np.asarray(dataobj)` appliquent.
    """
    image = nib.load(path)
    raw = np.asanyarray(image.dataobj.get_unscaled())
    return {f"{prefix}_extension": "".join(Path(path).suffixes), f"{prefix}_dtype": str(image.get_data_dtype()),
            f"{prefix}_slope": float(image.dataobj.slope), f"{prefix}_inter": float(image.dataobj.inter),
            f"{prefix}_brut_min": float(raw.min()), f"{prefix}_brut_max": float(raw.max())}


def label_values(path, prefix):
    values = np.unique(np.asanyarray(nib.load(path).dataobj))
    return {f"{prefix}_valeurs_distinctes": len(values),
            f"{prefix}_non_binaire": bool(np.any(~np.isclose(values, 0) & ~np.isclose(values, 1, atol=1e-6)))}


def subject_row(task):
    """Statistiques d'un sujet ; `statut` = ok, manquant (volumes absents) ou erreur (message)."""
    root, case = task
    row = dict(case)
    folder = Path(root) / case["subject"]
    paths = {name: volume_path(folder, name) for name in VOLUMES}
    if not all(path.is_file() for path in paths.values()):
        return {**row, "statut": "manquant"}
    try:
        for name in VOLUMES:
            row.update(header_info(paths[name], name))
        for name in ("mask", "brain"):
            row.update(label_values(paths[name], name))
        t1 = read_image(paths["t1"])[0][0].numpy()       # exactement ce que reçoit le réseau
        brain = read_label(paths["brain"])[0][0].numpy().astype(bool)
        mask = read_label(paths["mask"])[0][0].numpy().astype(bool)
        finite = np.isfinite(t1)
        values = t1[finite]
        inside = t1[brain & finite]
        row.update(forme="x".join(map(str, t1.shape)), t1_lu_dtype=str(t1.dtype),
                   t1_min=float(values.min()) if values.size else float("nan"),
                   t1_max=float(values.max()) if values.size else float("nan"),
                   t1_moyenne=float(values.mean()) if values.size else float("nan"),
                   t1_nan=int(np.isnan(t1).sum()), t1_inf=int(np.isinf(t1).sum()),
                   t1_abs_sup_10=int((np.abs(values) > 10).sum()),
                   t1_abs_sup_100=int((np.abs(values) > 100).sum()),
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


def done_cases(output):
    """case_id déjà inventoriés ; réécrit le CSV sans ses lignes `erreur` pour les recalculer."""
    if not output.is_file() or output.stat().st_size == 0:
        return set()
    with open(output, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            raise SystemExit(f"{output} n'a pas les colonnes attendues (ancien format ?) : "
                             "choisir un autre --output ou relancer avec --recommencer")
        rows = list(reader)
    kept = [row for row in rows if row["statut"] != "erreur"]
    if len(kept) != len(rows):
        with open(output, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, COLUMNS)
            writer.writeheader()
            writer.writerows(kept)
    return {row["case_id"] for row in kept}


def inventory(root, splits, output, workers=MAX_WORKERS, restart=False):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if restart and output.exists():
        output.unlink()
    done = done_cases(output)
    cases = [case for case in cases_of(load_splits(splits)) if case["case_id"] not in done]
    tasks = [(str(root), case) for case in cases]
    workers = max(1, min(workers, MAX_WORKERS))
    counts = {}
    with open(output, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, COLUMNS)
        if not done:
            writer.writeheader()
        bar = tqdm(total=len(done) + len(tasks), initial=len(done), unit="sujet", dynamic_ncols=True)

        def record(row):
            writer.writerow(row)
            handle.flush()
            counts[row["statut"]] = counts.get(row["statut"], 0) + 1
            bar.set_postfix(counts, refresh=False)
            bar.update()

        if workers > 1:
            with ProcessPoolExecutor(workers) as pool:
                for future in as_completed([pool.submit(subject_row, task) for task in tasks]):
                    record(future.result())
        else:
            for task in tasks:
                record(subject_row(task))
        bar.close()
    return output, len(done), counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=str(DEFAULT_PROCESSED))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help=f"au plus {MAX_WORKERS}")
    parser.add_argument("--recommencer", action="store_true", help="vider le CSV au lieu de le reprendre")
    args = parser.parse_args()
    output, skipped, counts = inventory(args.data_root, args.splits, args.output, args.workers, args.recommencer)
    print(f"{skipped} sujets repris, nouveaux : {counts} -> {output}")


if __name__ == "__main__":
    main()
