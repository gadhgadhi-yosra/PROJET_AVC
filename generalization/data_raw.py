"""Chargement provisoire des données ATLAS v3 brutes, en attendant le contrat C1 (fin S4).

    records = list_subjects("data/ATLAS3_Training_Raw")
    subject = load_subject(records.iloc[0])     # torchio.Subject : t1, mask, brain

Chaque sujet est réorienté en RAS, rééchantillonné à 1 mm, son masque binarisé avec `> 0.5`
et son image normalisée en z-score dans le cerveau. Le chemin des données est un paramètre :
les transformations ne dépendent que du `torchio.Subject` produit ici.
"""
import csv
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchio as tio

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "ATLAS3_Training_Raw"
# R032 : espacement des voxels incohérent dans l'en-tête (cerveaux de 2,7 L), décision QC en attente.
EXCLUDED_SITES = ("R032",)
# Masques vides relevés dans exploration/outputs/qc_flags.csv.
KNOWN_EMPTY_MASKS = frozenset({"sub-soop0910", "sub-soop0961", "sub-r041s108", "sub-r053s016", "sub-r065s002"})


def read_metadata(path):
    """Lit le CSV d'un sujet. Deux fichiers SOOP n'ont que l'en-tête : `metadata_empty` vaut alors True."""
    result = dict(days_post_stroke=float("nan"), chronicity=False, atlas2_dataset="", metadata_empty=True)
    if not path:
        return result
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return result
    row = rows[0]
    result["metadata_empty"] = False
    days = (row.get("DAYS_POST_STROKE") or "").strip()
    if days and days.lower() not in ("nan", "n/a", "na"):
        result["days_post_stroke"] = float(days)
    # CHRONICITY est un drapeau codé '1' ou '1.0', vide sinon : ce n'est pas un stade.
    flag = (row.get("CHRONICITY") or "").strip()
    result["chronicity"] = bool(flag) and float(flag) == 1
    result["atlas2_dataset"] = (row.get("ATLAS2_DATASET") or "").strip()
    return result


def list_subjects(root=DEFAULT_ROOT, sites=None, exclude_sites=EXCLUDED_SITES, with_metadata=True):
    """Table des sujets : subject, site, session, chemins, métadonnées, `known_empty_mask`.

    Le centre vient du nom du dossier (colonne SITE des métadonnées, pas CENTER).
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dossier de données introuvable : {root}")
    rows = []
    for anat in sorted(root.glob("*/sub-*/ses-*/anat")):
        site, subject, session = anat.parts[-4], anat.parts[-3], anat.parts[-2]
        if site in exclude_sites or (sites is not None and site not in sites):
            continue
        found = {key: sorted(anat.glob(pattern)) for key, pattern in [
            ("t1_path", "*_desc-brain_T1w.nii.gz"),
            ("mask_path", "*_label-lesion_*mask.nii.gz"),
            ("metadata_path", "*_metadata.csv")]}
        if len(found["t1_path"]) != 1 or len(found["mask_path"]) != 1:
            warnings.warn(f"{subject} ignoré : paire image/masque manquante ou ambiguë dans {anat}")
            continue
        row = dict(subject=subject, site=site, session=session,
                   **{key: str(files[0]) if files else None for key, files in found.items()},
                   known_empty_mask=subject in KNOWN_EMPTY_MASKS)
        if with_metadata:
            row.update(read_metadata(row["metadata_path"]))
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"Aucun sujet trouvé sous {root} (sites={sites}, exclus={exclude_sites})")
    return pd.DataFrame(rows)


def background_value(data):
    """Valeur du fond = valeur la plus fréquente (0 en général, ~0,0007 à R029)."""
    values, counts = torch.unique(data[..., ::4, ::4, ::4], return_counts=True)
    return float(values[counts.argmax()])


def zscore_in_brain(subject, background=0.0):
    """Z-score calculé dans le cerveau ; le fond est mis à `background` (convention nnU-Net)."""
    data = subject["t1"].data.float()
    brain = subject["brain"].data.bool()
    values = data[brain]
    mean, std = float(values.mean()), float(values.std())
    normalized = (data - mean) / max(std, 1e-8)
    normalized[~brain] = background
    subject["t1"].set_data(normalized)
    subject["norm_mean"], subject["norm_std"] = mean, std
    return subject


def load_subject(record, spacing=1.0, normalize=True):
    """Charge une ligne de `list_subjects` en `torchio.Subject` (t1, mask, brain) en RAS.

    spacing : taille de voxel cible en mm, ou None pour garder la grille d'origine.
    Le sujet porte aussi `subject`, `site`, `empty_mask` et, si `normalize`, `norm_mean` /
    `norm_std` pour revenir aux intensités brutes (brut = z * norm_std + norm_mean).
    """
    record = dict(record)
    image = tio.ScalarImage(record["t1_path"])
    mask = tio.LabelMap(record["mask_path"])
    if image.spatial_shape != mask.spatial_shape or not np.allclose(image.affine, mask.affine, atol=1e-3):
        raise ValueError(f"{record['subject']} : image et masque sur des grilles différentes")
    data = image.data[:1].float()
    # Cerveau = voxels au-dessus du fond, pas « > 0 » : à R029 le fond vaut ~0,0007.
    brain = data > background_value(data)
    # Jamais « == 1 » : certains masques codent la lésion 0,99999998.
    lesion = mask.data[:1] > 0.5
    lesion_voxels_raw = int(lesion.sum())
    affine = image.affine
    subject = tio.Subject(
        t1=tio.ScalarImage(tensor=data, affine=affine),
        mask=tio.LabelMap(tensor=lesion.to(torch.uint8), affine=affine),
        brain=tio.LabelMap(tensor=brain.to(torch.uint8), affine=affine),
    )
    subject = tio.ToCanonical()(subject)
    if spacing is not None:
        # Interpolation linéaire pour l'image, plus proche voisin pour les masques.
        subject = tio.Resample(spacing)(subject)
    lesion_voxels = int(subject["mask"].data.sum())
    if lesion_voxels == 0:
        if lesion_voxels_raw:
            warnings.warn(f"{record['subject']} : lésion de {lesion_voxels_raw} voxels perdue au rééchantillonnage")
        elif not record.get("known_empty_mask"):
            warnings.warn(f"{record['subject']} : masque vide non répertorié dans qc_flags.csv")
    for key in ("subject", "site", "days_post_stroke", "chronicity"):
        if key in record:
            subject[key] = record[key]
    subject["empty_mask"] = lesion_voxels == 0
    if normalize:
        zscore_in_brain(subject)
    return subject


class RawAtlasDataset(torch.utils.data.Dataset):
    """Jeu PyTorch minimal : charge chaque sujet à la demande puis applique `transform`."""

    def __init__(self, records, transform=None, spacing=1.0, normalize=True):
        self.records = records.reset_index(drop=True)
        self.transform = transform
        self.spacing = spacing
        self.normalize = normalize

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        subject = load_subject(self.records.iloc[index], self.spacing, self.normalize)
        return self.transform(subject) if self.transform is not None else subject
