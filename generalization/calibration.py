"""Calibrage des transformations sur les seuls sujets d'entraînement de `splits.json`.

Aucun réglage ne doit voir les centres de test OOD ni le centre exclu : ce module lit `splits.json`,
refuse tout centre tenu à l'écart et recalcule, sujet par sujet sur les données brutes, les grandeurs
qui fixent les plages de gamma et de contraste et le centre de référence de FDA et de l'histogram
matching. Le résultat agrégé par centre va dans `calibration.json` (versionné) ; les mesures par sujet
restent en local, dans `generalization/figures/calibration/` (ignoré par Git).

    .venv-dl\\Scripts\\python -m generalization.calibration
"""
import argparse
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from .data_raw import DEFAULT_ROOT, background_value, list_subjects

HERE = Path(__file__).resolve().parent
DEFAULT_SPLITS = HERE.parent / "splits.json"
CALIBRATION_FILE = HERE / "calibration.json"
LOCAL_DIR = HERE / "figures" / "calibration"
# Même histogramme que exploration/01_compute_stats.py : intensité / p99 du cerveau.
HIST_EDGES = np.linspace(0.0, 1.5, 151)
HIST_WIDTH = float(HIST_EDGES[1] - HIST_EDGES[0])
# Critères du centre de référence : assez de sujets, 1 mm isotrope, pas d'histogramme en peigne (R029).
REFERENCE_MIN_SUBJECTS = 20
REFERENCE_MIN_LEVELS = 200


# --- splits.json ---

def load_splits(path=DEFAULT_SPLITS):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def held_out_sites(splits):
    """Centres interdits pour tout réglage : centres de test OOD et centres exclus."""
    return sorted(set(splits["ood_centers"]) | set(splits.get("excluded_centers", [])))


def training_cases(splits, fold=None):
    """Table (case_id, subject, site, session) des sujets d'entraînement, sans test ID ni OOD.

    Avec `fold`, seulement la partie entraînement de ce pli (sa validation est retirée).
    """
    cases = pd.DataFrame(splits["cases"])
    keep = splits["train"] if fold is None else splits["folds"][fold]["train"]
    return cases[cases["case_id"].isin(set(keep))].reset_index(drop=True)


def training_sites(splits, fold=None):
    return sorted(training_cases(splits, fold)["site"].unique())


def check_training_sites(sites, splits, purpose="un réglage", fold=None):
    """Erreur si un centre n'appartient pas au pool d'entraînement (OOD, exclu ou inconnu).

    Avec `fold`, les centres de validation de ce pli sont aussi interdits.
    """
    sites = [sites] if isinstance(sites, str) else list(sites)
    held_out = sorted(set(sites) & set(held_out_sites(splits)))
    if held_out:
        raise ValueError(f"Centres de test ou exclus interdits pour {purpose} : {held_out}")
    if fold is not None:
        validation = sorted(set(sites) & set(splits["folds"][fold]["validation_centers"]))
        if validation:
            raise ValueError(f"Centres de validation du pli {fold} interdits pour {purpose} : {validation}")
    unknown = sorted(set(sites) - set(training_sites(splits, fold)))
    if unknown:
        raise ValueError(f"Centres absents du pool d'entraînement de splits.json pour {purpose} : {unknown}")
    return sites


def restrict_to_training(records, splits, fold=None):
    """Garde les lignes qui sont des sujets d'entraînement (même centre, sujet et session), du pli `fold` si donné."""
    keys = set(zip(*(training_cases(splits, fold)[column] for column in ("site", "subject", "session"))))
    keep = [key in keys for key in zip(records["site"], records["subject"], records["session"])]
    return records[keep].reset_index(drop=True)


# --- Mesures par sujet ---

def subject_stats(record):
    """Grandeurs d'intensité d'un sujet brut, définies comme dans exploration/01_compute_stats.py."""
    image = nib.load(record["t1_path"])
    data = np.asarray(image.dataobj, dtype=np.float32)
    if data.ndim > 3:
        data = data[..., 0]
    values = data[data > background_value(torch.from_numpy(data))]
    p50, p99 = np.percentile(values, [50, 99])
    histogram, _ = np.histogram(values / p99, bins=HIST_EDGES)
    zooms = image.header.get_zooms()[:3]
    return dict(site=record["site"], subject=record["subject"], session=record["session"],
                p50=float(p50), p99=float(p99), cv=float(values.std() / values.mean()),
                levels=int(np.unique(values[::max(1, values.size // 200_000)]).size),
                voxel_min=float(min(zooms)), voxel_max=float(max(zooms)),
                **{f"h{index:03d}": value for index, value in enumerate(histogram / values.size)})


def compute_stats(records, workers=3):
    rows = [row for _, row in records.iterrows()]
    if workers <= 1:
        return pd.DataFrame([subject_stats(row) for row in rows])
    with ProcessPoolExecutor(workers) as pool:
        return pd.DataFrame(list(pool.map(subject_stats, rows, chunksize=4)))


# --- Calibrage ---

def site_table(stats, splits):
    """Médianes par centre. Refuse toute mesure d'un centre tenu à l'écart."""
    check_training_sites(stats["site"].unique(), splits, "le calibrage")
    grouped = stats.groupby("site")
    table = pd.DataFrame({
        "n": grouped.size(),
        "p50": grouped["p50"].median(), "p99": grouped["p99"].median(), "cv": grouped["cv"].median(),
        "levels": grouped["levels"].median(),
        "voxel_min": grouped["voxel_min"].median(), "voxel_max": grouped["voxel_max"].median(),
    })
    # Position de la médiane du cerveau sur [0, p99], comme le calibrage d'origine.
    table["position"] = table["p50"] / table["p99"]
    return table


def outward(low, high, step=0.01):
    """Arrondit une plage vers l'extérieur."""
    return [math.floor(low / step + 1e-9) * step, math.ceil(high / step - 1e-9) * step]


def gamma_range(table):
    """Gamma qui amène le centre médian sur chaque centre : ln(position) / ln(position médiane)."""
    reference = table["position"].median()
    gamma = np.log(table["position"]) / np.log(reference)
    return dict(range=[round(v, 2) for v in outward(gamma.min(), gamma.max())],
                raw=[round(float(gamma.min()), 3), round(float(gamma.max()), 3)],
                sites=[gamma.idxmin(), gamma.idxmax()], position_median=round(float(reference), 3))


def contrast_range(table):
    """Coefficient de variation du cerveau rapporté à sa médiane sur les centres."""
    reference = table["cv"].median()
    factor = table["cv"] / reference
    return dict(range=[round(v, 2) for v in outward(factor.min(), factor.max())],
                raw=[round(float(factor.min()), 3), round(float(factor.max()), 3)],
                sites=[factor.idxmin(), factor.idxmax()], cv_median=round(float(reference), 3))


def histogram_columns(stats):
    return [column for column in stats.columns if column.startswith("h") and column[1:].isdigit()]


def reference_ranking(stats, table):
    """Centres candidats classés par distance W1 de leur histogramme moyen au reste du pool d'entraînement.

    Un centre « typique » change le moins possible les images qu'on harmonise vers lui. La distance est
    en fraction du p99 du cerveau ; la référence exclut le centre mesuré.
    """
    columns = histogram_columns(stats)
    eligible = table[(table["n"] >= REFERENCE_MIN_SUBJECTS) & (table["voxel_max"] <= 1.05)
                     & (table["voxel_min"] >= 0.95) & (table["levels"] >= REFERENCE_MIN_LEVELS)]
    rows = []
    for site in eligible.index:
        inside = stats.loc[stats["site"] == site, columns].to_numpy().mean(axis=0)
        outside = stats.loc[stats["site"] != site, columns].to_numpy().mean(axis=0)
        distance = np.abs(np.cumsum(inside) - np.cumsum(outside)).sum() * HIST_WIDTH
        rows.append(dict(site=site, n=int(eligible.loc[site, "n"]), w1=round(float(distance), 4)))
    return sorted(rows, key=lambda row: row["w1"])


def splits_fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def fold_references(stats, splits):
    """Référence de chaque pli : même critère, sur les seuls sujets d'entraînement du pli.

    Les centres de validation du pli ne sont ni candidats ni dans le pool qui sert à mesurer la distance.
    """
    references = []
    for fold, content in enumerate(splits["folds"]):
        fold_stats = restrict_to_training(stats, splits, fold)
        ranking = reference_ranking(fold_stats, site_table(fold_stats, splits))
        check_training_sites(ranking[0]["site"], splits, f"la référence du pli {fold}", fold)
        references.append(dict(fold=fold, site=ranking[0]["site"], n_subjects=int(len(fold_stats)),
                               validation_centers=content["validation_centers"], ranking=ranking[:4]))
    return references


def calibrate(stats, splits, splits_path):
    table = site_table(stats, splits)
    ranking = reference_ranking(stats, table)
    return table, {
        "date": date.today().isoformat(),
        "splits": dict(file=Path(splits_path).name, version=splits.get("version"), seed=splits.get("seed"),
                       sha256_16=splits_fingerprint(splits_path)),
        "held_out_sites": held_out_sites(splits),
        "training_sites": sorted(table.index),
        "n_training_subjects": int(len(stats)),
        "gamma": gamma_range(table),
        "contrast": contrast_range(table),
        "reference": dict(site=ranking[0]["site"], criteria=(
            f"au moins {REFERENCE_MIN_SUBJECTS} sujets d'entraînement, 1 mm isotrope, au moins "
            f"{REFERENCE_MIN_LEVELS} niveaux de gris ; plus petite distance W1 au reste du pool"),
            ranking=ranking[:8]),
        # build_transforms(config, fold=k) prend fold_references[k] ; `reference` sert à l'entraînement final.
        "fold_references": fold_references(stats, splits),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", type=Path, default=CALIBRATION_FILE)
    parser.add_argument("--stats", type=Path, default=LOCAL_DIR / "stats_entrainement.csv",
                        help="cache local des mesures par sujet (jamais dans Git)")
    args = parser.parse_args()

    splits = load_splits(args.splits)
    if args.stats.is_file():
        stats = pd.read_csv(args.stats)
        print(f"Mesures relues dans {args.stats}")
    else:
        records = restrict_to_training(list_subjects(args.data_root, with_metadata=False), splits)
        print(f"{len(records)} sujets d'entraînement sur {records['site'].nunique()} centres : mesure en cours…")
        stats = compute_stats(records, args.workers)
        args.stats.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(args.stats, index=False)
    table, result = calibrate(stats, splits, args.splits)
    # Les sections écrites à la main (balayages de FDA et de BigAug) sont conservées.
    previous = json.loads(args.out.read_text(encoding="utf-8")) if args.out.is_file() else {}
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({**previous, **result}, ensure_ascii=False, indent=2) + "\n")
    table.to_csv(LOCAL_DIR / "centres_entrainement.csv")
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(table.round(3).sort_values("position"))
    print(json.dumps({key: result[key] for key in ("gamma", "contrast", "reference")}, ensure_ascii=False, indent=2))
    print("Référence par pli :", [reference["site"] for reference in result["fold_references"]])


if __name__ == "__main__":
    main()
