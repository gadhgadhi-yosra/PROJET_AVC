"""Volumes synthétiques imitant ATLAS v3 brut, pour tester sans aucune donnée patient.

Un « cerveau » ellipsoïde (substance blanche, cortex, ventricules, texture) avec une petite
lésion sombre, écrit dans la même arborescence que les données réelles :
    <root>/<SITE>/sub-*/ses-1/anat/{*_desc-brain_T1w.nii.gz, *_label-lesion_desc-T1lesion_mask.nii.gz, *_metadata.csv}

Les centres générés reprennent les pièges relevés pendant l'exploration : orientation LAS/PSR,
voxels anisotropes de SOOP, fond non nul et valeurs négatives de R029, masque codé 0,99999998,
masque vide, métadonnées vides et centre R032 à exclure.
"""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

FOV_MM = (72.0, 84.0, 68.0)  # petit champ de vue : les tests restent rapides
FOV_REALISTIC_MM = (176.0, 208.0, 176.0)  # proche d'un cerveau réel, pour les mesures de temps
METADATA_COLUMNS = ["SESSION_ID", "SITE", "DAYS_POST_STROKE", "CHRONICITY", "ATLAS2_DATASET"]


def _smooth_noise(rng, shape, sigma_mm, spacing):
    noise = ndimage.gaussian_filter(rng.standard_normal(shape).astype(np.float32),
                                    [sigma_mm / s for s in spacing])
    return noise / noise.std()


def synthetic_volume(spacing=(1.0, 1.0, 1.0), fov_mm=FOV_MM, seed=0, lesion_radius_mm=5.0,
                     lesion_ratio=0.7, scale=800.0):
    """Retourne (image float32, lésion bool, affine RAS). Le fond vaut exactement 0."""
    rng = np.random.default_rng(seed)
    spacing = tuple(float(s) for s in spacing)
    shape = tuple(int(round(f / s)) for f, s in zip(fov_mm, spacing))
    axes = [(np.arange(n) - (n - 1) / 2) * s for n, s in zip(shape, spacing)]
    x, y, z = np.meshgrid(*axes, indexing="ij", sparse=True)
    rx, ry, rz = (0.45 * f for f in fov_mm)
    rho = np.sqrt((x / rx) ** 2 + (y / ry) ** 2 + (z / rz) ** 2)
    brain = rho < 1.0

    coarse = _smooth_noise(rng, shape, 5.0, spacing)
    fine = _smooth_noise(rng, shape, 1.5, spacing)
    image = np.where(rho + 0.12 * coarse < 0.72, 1.0, 0.68).astype(np.float32)  # blanche / grise
    for side in (-1, 1):  # ventricules
        ventricle = ((x - side * 0.13 * rx) / (0.09 * rx)) ** 2 + (y / (0.35 * ry)) ** 2 + (z / (0.2 * rz)) ** 2 < 1
        image[ventricle] = 0.25
    image *= 1.0 + 0.06 * fine
    image = ndimage.gaussian_filter(image, [0.6 / s for s in spacing])  # volume partiel
    image += 0.02 * rng.standard_normal(shape).astype(np.float32)

    lesion = np.zeros(shape, dtype=bool)
    if lesion_radius_mm > 0:
        centre = (0.42 * rx, 0.15 * ry, 0.1 * rz)
        distance = np.sqrt((x - centre[0]) ** 2 + (y - centre[1]) ** 2 + (z - centre[2]) ** 2)
        lesion = (distance + 0.8 * fine < lesion_radius_mm) & brain
        image[lesion] *= lesion_ratio

    image = np.where(brain, np.clip(image, 0.05, None) * scale, 0.0).astype(np.float32)
    affine = np.diag([*spacing, 1.0])
    affine[:3, 3] = [-(n - 1) / 2 * s for n, s in zip(shape, spacing)]
    return image, lesion, affine


def _reorient(data, affine, orientation):
    """Stocke le volume RAS dans une autre orientation (LAS, PSR…) sans changer sa géométrie."""
    image = nib.Nifti1Image(data, affine)
    target = nib.orientations.axcodes2ornt(tuple(orientation))
    transform = nib.orientations.ornt_transform(nib.orientations.axcodes2ornt(("R", "A", "S")), target)
    image = image.as_reoriented(transform)
    return np.asarray(image.dataobj), image.affine


def write_subject(root, site, subject, image, lesion, affine, orientation="RAS", dtype=np.int16,
                  mask_value=1, mask_dtype=np.uint8, metadata=None):
    """Écrit un sujet au format ATLAS brut. `metadata=None` écrit un CSV sans ligne (cas SOOP)."""
    anat = Path(root) / site / subject / "ses-1" / "anat"
    anat.mkdir(parents=True, exist_ok=True)
    image, image_affine = _reorient(image.astype(dtype), affine, orientation)
    mask, mask_affine = _reorient((lesion * mask_value).astype(mask_dtype), affine, orientation)
    nib.save(nib.Nifti1Image(image, image_affine), anat / f"{subject}_ses-1_desc-brain_T1w.nii.gz")
    nib.save(nib.Nifti1Image(mask, mask_affine), anat / f"{subject}_ses-1_label-lesion_desc-T1lesion_mask.nii.gz")
    lines = [",".join(METADATA_COLUMNS)]
    if metadata is not None:
        row = {"SESSION_ID": f"{subject}_ses-1", "SITE": site, **metadata}
        lines.append(",".join(str(row.get(column, "")) for column in METADATA_COLUMNS))
    (anat / f"{subject}_ses-1_metadata.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return anat


def write_synthetic_dataset(root, fov_mm=FOV_MM, seed=0):
    """Écrit un petit jeu multi-centres et retourne sa racine (à passer à `data_raw.list_subjects`)."""
    root = Path(root)
    chronic = {"DAYS_POST_STROKE": 426, "CHRONICITY": 1, "ATLAS2_DATASET": "Training"}

    # R009 : 1 mm isotrope, stocké en LAS (40 % des fichiers réels).
    for index in (1, 2):
        volume = synthetic_volume(fov_mm=fov_mm, seed=seed + index)
        write_subject(root, "R009", f"sub-r009s00{index}", *volume, orientation="LAS", metadata=chronic)

    # SOOP : 0,5 x 0,47 x 0,47 mm, pas de délai post-AVC, un fichier en PSR.
    soop = {"DAYS_POST_STROKE": "", "CHRONICITY": "1.0", "ATLAS2_DATASET": ""}
    for index, orientation in ((1, "RAS"), (2, "PSR")):
        volume = synthetic_volume(spacing=(0.5, 0.47, 0.47), fov_mm=fov_mm, seed=seed + 10 + index, scale=1300.0)
        write_subject(root, "SOOP", f"sub-soop000{index}", *volume, orientation=orientation, metadata=soop)
    # Masque vide et métadonnées vides, comme sub-soop0910.
    volume = synthetic_volume(fov_mm=fov_mm, seed=seed + 13, lesion_radius_mm=0)
    write_subject(root, "SOOP", "sub-soop0003", *volume, metadata=None)

    # R029 : ~30 niveaux de gris, fond à 0,0007 au lieu de 0 et quelques valeurs négatives.
    image, lesion, affine = synthetic_volume(fov_mm=fov_mm, seed=seed + 20, scale=31.0)
    image = np.round(image) + 0.0007
    image[2:4, 2:4, 2:5] = -0.5
    write_subject(root, "R029", "sub-r029s001", image, lesion, affine, dtype=np.float32,
                  metadata={"DAYS_POST_STROKE": 326, "CHRONICITY": 1, "ATLAS2_DATASET": "Training"})

    # R014 : lésion presque invisible (rapport ~0,91), masque codé 0,99999998.
    volume = synthetic_volume(fov_mm=fov_mm, seed=seed + 30, lesion_ratio=0.91, lesion_radius_mm=3.5, scale=370.0)
    write_subject(root, "R014", "sub-r014s001", *volume, mask_value=0.99999998, mask_dtype=np.float64,
                  metadata={"DAYS_POST_STROKE": "", "CHRONICITY": "", "ATLAS2_DATASET": "Training"})

    # R032 : espacement d'en-tête incohérent dans les données réelles, centre exclu au chargement.
    volume = synthetic_volume(spacing=(0.94, 3.0, 0.94), fov_mm=fov_mm, seed=seed + 40, scale=70.0)
    write_subject(root, "R032", "sub-r032s005", *volume, orientation="LAS",
                  metadata={"DAYS_POST_STROKE": 128, "CHRONICITY": "", "ATLAS2_DATASET": ""})
    return root


# --- Format prétraité du contrat C1 (preprocessing/week4.py) ---

PROCESSED_FOV_MM = (56.0, 60.0, 52.0)  # après recadrage (marge de 5 voxels) : moins de 64³
Z_STEP = 1e-3  # pas du z-score stocké en int16 (scl_slope), comme week4.py


def write_processed_subject(root, site, subject, image, lesion, affine, margin=5, mask_value=1,
                            mask_dtype=np.uint8, orientation="RAS"):
    """Écrit `<root>/<sujet>/{t1,mask,brain}.nii.gz` et `qc.json` comme le pipeline C1.

    Cerveau = image > 0 (fond synthétique exactement nul), z-score dans le cerveau, fond à 0,
    recadrage sur le cerveau avec affine translatée, z-score en int16 avec scl_slope = 0,001.
    """
    brain = image > 0
    values = image[brain]
    mean, std = float(values.mean()), float(values.std())
    normalized = np.zeros(image.shape, np.float32)
    normalized[brain] = (values - mean) / std
    bounds = []
    for axis in range(3):
        index = np.flatnonzero(brain.any(axis=tuple(a for a in range(3) if a != axis)))
        bounds.append((max(int(index[0]) - margin, 0), min(image.shape[axis], int(index[-1]) + margin + 1)))
    crop = tuple(slice(low, high) for low, high in bounds)
    translation = np.eye(4)
    translation[:3, 3] = [low for low, _ in bounds]
    affine = affine @ translation
    out = Path(root) / subject
    out.mkdir(parents=True, exist_ok=True)
    volumes = {"t1": np.rint(normalized[crop] / Z_STEP).astype(np.int16),
               "mask": (lesion[crop] * mask_value).astype(mask_dtype),
               "brain": brain[crop].astype(np.uint8)}
    for name, data in volumes.items():
        data, stored_affine = _reorient(data, affine, orientation)
        result = nib.Nifti1Image(data, stored_affine)
        result.set_qform(stored_affine, code=1)
        result.set_sform(stored_affine, code=1)
        if name == "t1":
            result.set_data_dtype(np.int16)
            result.header.set_slope_inter(Z_STEP, 0)
        nib.save(result, out / f"{name}.nii.gz")
    qc = dict(raw_mean=mean, raw_std=std, shape=list(volumes["t1"].shape), empty_mask=not bool(lesion.any()),
              lesion_ml=float(lesion.sum() / 1000))
    (out / "qc.json").write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
    return out


def write_processed_dataset(root, fov_mm=PROCESSED_FOV_MM, seed=0):
    """Petit jeu prétraité + `subjects.parquet` + `splits.json` cohérents. Retourne (racine, chemin des splits).

    Entraînement : R009 (2 sujets), R014 (dont un masque vide et un masque codé 0,99999998), R029 ;
    test ID : un sujet de R009 ; test OOD : SOOP ; R032 traité mais exclu. Deux plis : R014 en
    validation au pli 0, R009 au pli 1.
    """
    root = Path(root)
    plan = [  # centre, sujet, rapport de la lésion, rayon (0 = masque vide), valeur du masque, statut
        ("R009", "sub-r009s001", 0.7, 5.0, 1, "ok"),
        ("R009", "sub-r009s002", 0.7, 5.0, 1, "ok"),
        ("R009", "sub-r009s003", 0.7, 5.0, 1, "ok"),
        ("R014", "sub-r014s001", 0.91, 3.5, 0.99999998, "ok"),
        ("R014", "sub-r014s002", 0.7, 0.0, 1, "ok"),
        ("R029", "sub-r029s001", 0.7, 5.0, 1, "ok"),
        ("SOOP", "sub-soop0001", 0.7, 5.0, 1, "ok"),
        ("R032", "sub-r032s005", 0.7, 5.0, 1, "excluded"),
    ]
    rows = []
    for index, (site, subject, ratio, radius, mask_value, status) in enumerate(plan):
        image, lesion, affine = synthetic_volume(fov_mm=fov_mm, seed=seed + index, lesion_ratio=ratio,
                                                 lesion_radius_mm=radius, scale=300.0 + 100 * index)
        dtype = np.float64 if mask_value != 1 else np.uint8
        write_processed_subject(root, site, subject, image, lesion, affine, mask_value=mask_value, mask_dtype=dtype)
        rows.append(dict(case_id=f"{site}_{subject}_ses-1", subject=subject, site=site, session="ses-1",
                         status=status, empty_mask=not bool(lesion.any())))
    pd.DataFrame(rows).to_parquet(root / "subjects.parquet", index=False)
    ids = {row["subject"]: row["case_id"] for row in rows}
    pick = lambda *subjects: [ids[subject] for subject in subjects]  # noqa: E731
    train = pick("sub-r009s001", "sub-r009s002", "sub-r014s001", "sub-r014s002", "sub-r029s001")
    splits = {
        "version": 1, "seed": 2026, "ood_centers": ["SOOP"], "excluded_centers": ["R032"],
        "cases": [{key: row[key] for key in ("case_id", "subject", "site", "session")}
                  for row in rows if row["site"] != "R032"],
        "train": train, "id_test": pick("sub-r009s003"), "ood_test": pick("sub-soop0001"),
        "folds": [
            {"train": pick("sub-r009s001", "sub-r009s002", "sub-r029s001"),
             "val": pick("sub-r014s001", "sub-r014s002"), "validation_centers": ["R014"],
             "id_test": pick("sub-r009s003")},
            {"train": pick("sub-r014s001", "sub-r014s002", "sub-r029s001"),
             "val": pick("sub-r009s001", "sub-r009s002"), "validation_centers": ["R009"], "id_test": []},
        ],
    }
    splits_path = root.parent / "splits.json"
    splits_path.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    return root, splits_path
