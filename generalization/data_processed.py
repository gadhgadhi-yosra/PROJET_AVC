"""Chargeur des données prétraitées du contrat C1 (pipeline de Yosra).

    subjects = load_processed("data/processed", split="train", fold=0)   # liste de torchio.Subject
    subject = subjects[0]                                                 # clés : t1, mask, brain, site…

Format attendu : `<root>/<sujet>/{t1,mask,brain}.nii.gz` (RAS, 1 mm, z-score dans le cerveau stocké en
int16 avec `scl_slope`, fond à 0, recadré) et `qc.json`, plus la table `<root>/subjects.parquet`.
Les images sont lues avec nibabel, à la demande : la liste ne charge que les en-têtes, qui sont
vérifiés tout de suite. `data_raw.py` reste le repli sur les données brutes.
"""
import json
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torchio as tio

from .calibration import DEFAULT_SPLITS, load_splits
from .data_raw import EXCLUDED_SITES

DEFAULT_PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
VOLUMES = ("t1", "mask", "brain")
SPLITS = ("train", "val", "id_test", "ood_test")


class ProcessedFormatError(ValueError):
    """Volume prétraité qui ne respecte pas le contrat C1 (orientation, espacement, grille, fichier)."""


# --- Lecteurs nibabel (fonctions de module : sérialisables pour les DataLoader) ---

def read_image(path):
    """Image z-score : float32 (1, X, Y, Z), `scl_slope` appliqué par nibabel."""
    image = nib.load(path)
    return torch.from_numpy(np.asarray(image.get_fdata(dtype=np.float32)))[None], image.affine


def read_label(path):
    """Masque binaire : seuil `> 0.5`, jamais `== 1` (masques codés 0,99999998)."""
    image = nib.load(path)
    return torch.from_numpy(np.asarray(image.dataobj) > 0.5).to(torch.uint8)[None], image.affine


# --- Contrôles ---

def check_geometry(paths, case):
    """Vérifie sur les en-têtes : RAS, 1 mm isotrope, même grille pour t1, mask et brain."""
    reference = None
    for name, path in paths.items():
        header = nib.load(path)
        if len(header.shape) != 3:
            raise ProcessedFormatError(f"{case} : {name}.nii.gz a {len(header.shape)} dimensions au lieu de 3")
        codes = "".join(nib.aff2axcodes(header.affine))
        if codes != "RAS":
            raise ProcessedFormatError(f"{case} : {name}.nii.gz est orienté {codes} au lieu de RAS "
                                       "(contrat C1) ; refaire le prétraitement ou passer par data_raw")
        spacing = np.asarray(header.header.get_zooms()[:3], dtype=float)
        if not np.allclose(spacing, 1.0, atol=1e-3):
            raise ProcessedFormatError(f"{case} : {name}.nii.gz a des voxels de {tuple(spacing.round(3))} mm "
                                       "au lieu de 1 mm isotrope (contrat C1)")
        if reference is None:
            reference = (header.shape, header.affine)
        elif header.shape != reference[0] or not np.allclose(header.affine, reference[1], atol=1e-3):
            raise ProcessedFormatError(f"{case} : {name}.nii.gz n'est pas sur la grille de t1.nii.gz")


def split_case_ids(splits, split, fold=None):
    """Identifiants d'un découpage : train, id_test, ood_test ; avec `fold`, aussi val (et train du pli)."""
    if split not in SPLITS:
        raise ValueError(f"split inconnu : {split!r} (attendus : {', '.join(SPLITS)})")
    if split == "ood_test":
        return list(splits["ood_test"])
    if fold is None:
        if split == "val":
            raise ValueError("split='val' exige un pli : load_processed(..., split='val', fold=k)")
        return list(splits[split])
    if not 0 <= fold < len(splits["folds"]):
        raise ValueError(f"Pli {fold} absent de splits.json ({len(splits['folds'])} plis)")
    return list(splits["folds"][fold][split])


def read_table(root, table=None):
    """Table des sujets traités (`subjects.parquet`, ou `.csv`), restreinte aux statuts `ok`."""
    path = Path(table) if table else Path(root) / "subjects.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Table des sujets introuvable : {path} (sortie de preprocessing/week4.py)")
    frame = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    if "status" in frame:
        frame = frame[frame["status"] == "ok"]
    if "case_id" not in frame:
        frame = frame.assign(case_id=frame["site"] + "_" + frame["subject"] + "_" + frame["session"])
    return frame[~frame["site"].isin(EXCLUDED_SITES)].reset_index(drop=True)


# --- Liste et chargement ---

def list_processed(root=DEFAULT_PROCESSED, split="train", fold=None, splits=DEFAULT_SPLITS, table=None,
                   on_missing="error", sites=None, warn_empty=True):
    """Table des sujets d'un découpage : case_id, subject, site, session, empty_mask, chemins.

    on_missing : "error" (défaut) si un sujet du découpage n'a pas ses volumes, "warn" pour l'écarter.
    sites : ne garder que ces centres. Les masques vides sont signalés par un avertissement
    (`warn_empty`), jamais une erreur.
    """
    root = Path(root)
    splits = load_splits(splits) if isinstance(splits, (str, Path)) else splits
    wanted = split_case_ids(splits, split, fold)
    frame = read_table(root, table).set_index("case_id")
    cases = pd.DataFrame(splits["cases"]).set_index("case_id")
    rows, missing = [], []
    for case in wanted:
        site = cases.loc[case, "site"]
        if site in EXCLUDED_SITES or (sites is not None and site not in sites):
            continue
        folder = root / cases.loc[case, "subject"]
        paths = {name: folder / f"{name}.nii.gz" for name in VOLUMES}
        if case not in frame.index or not all(path.is_file() for path in paths.values()):
            missing.append(case)
            continue
        check_geometry(paths, case)
        rows.append(dict(case_id=case, subject=cases.loc[case, "subject"], site=site,
                         session=cases.loc[case, "session"], empty_mask=_empty_mask(frame.loc[case], folder),
                         **{f"{name}_path": str(path) for name, path in paths.items()},
                         **_normalization(folder)))
    if missing:
        message = (f"{len(missing)} sujets du découpage {split!r}" + (f" (pli {fold})" if fold is not None else "")
                   + f" sans volumes prétraités dans {root} : {missing[:5]}{' …' if len(missing) > 5 else ''}")
        if on_missing == "error":
            raise ProcessedFormatError(message + " ; passer on_missing='warn' pour les écarter")
        warnings.warn(message + " : écartés")
    empty = [row["case_id"] for row in rows if row["empty_mask"]]
    if empty and warn_empty:
        warnings.warn(f"{len(empty)} sujets à masque vide (gardés comme cas négatifs) : {empty}")
    return pd.DataFrame(rows, columns=["case_id", "subject", "site", "session", "empty_mask", "t1_path",
                                       "mask_path", "brain_path", "norm_mean", "norm_std"])


def _qc(folder):
    path = folder / "qc.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _empty_mask(row, folder):
    """Masque vide d'après la table, sinon d'après qc.json, sinon en lisant le masque."""
    for value in (row.get("empty_mask"), _qc(folder).get("empty_mask")):
        if value is not None and not pd.isna(value):
            return bool(value)
    return not bool(read_label(folder / "mask.nii.gz")[0].any())


def _normalization(folder):
    """Moyenne et écart-type bruts du cerveau (qc.json) : brut = z × norm_std + norm_mean."""
    qc = _qc(folder)
    return dict(norm_mean=float(qc.get("raw_mean", 0.0)), norm_std=float(qc.get("raw_std", 1.0)))


def load_processed_subject(record):
    """`torchio.Subject` paresseux d'une ligne de `list_processed` : t1, mask, brain, site, subject…"""
    record = dict(record)
    subject = tio.Subject(
        t1=tio.ScalarImage(record["t1_path"], reader=read_image),
        mask=tio.LabelMap(record["mask_path"], reader=read_label),
        # Le cerveau vient de brain.nii.gz, jamais de « > 0 » : le z-score vaut 0 à l'intérieur aussi.
        brain=tio.LabelMap(record["brain_path"], reader=read_label),
    )
    for key in ("case_id", "subject", "site", "session", "empty_mask", "norm_mean", "norm_std"):
        subject[key] = record[key]
    return subject


def load_processed(root=DEFAULT_PROCESSED, split="train", fold=None, splits=DEFAULT_SPLITS, table=None,
                   on_missing="error"):
    """Liste de `torchio.Subject` (t1, mask, brain, site…) d'un découpage de splits.json.

    split : "train", "id_test" ou "ood_test" ; avec `fold=k`, "train" et "val" sont ceux du pli k et
    "id_test" le test ID du pli. Les images ne sont lues qu'à l'accès (`subject["t1"].data`).
    """
    records = list_processed(root, split, fold, splits, table, on_missing)
    return [load_processed_subject(record) for _, record in records.iterrows()]
