"""Adaptateur de la démo (`demo/`) pour le modèle C8 : `predict(chemin_irm) -> {"mask_path": ...}`.

    AVC_PREDICT_ADAPTER=adaptation.demo_adapter:predict python demo/app.py

La démo appelle `predict(str(chemin))` et attend un dictionnaire avec `mask_path` : un masque binaire NIfTI dans
la grille de l'IRM reçue. Cette fonction appelle `adaptation.inference.predict` (prétraitement S4, modèle
intensité) puis enregistre le masque en uint8, avec la forme et l'affine de l'image d'entrée.

- Dossier de sortie : `AVC_MASK_DIR` s'il est défini, sinon un dossier temporaire créé pour l'appel.
- Nom : celui de l'IRM sans extension, suivi de `_masque_c8.nii.gz`.
- Checkpoint : résolu comme dans `inference.py` (`PROJET_AVC_MODELE`, sinon
  `data/modeles/unet3d_intensity_pli0_best.pt`).
Pas de carte d'incertitude (`uncertainty_path` absent) : le modèle ne la produit pas.
"""
import os
import tempfile
from pathlib import Path

import numpy as np

from . import inference

SUFFIX = "_masque_c8.nii.gz"


def mask_name(path):
    """Nom du masque dérivé de celui de l'IRM (sans .nii ni .nii.gz)."""
    name = Path(path).name
    for extension in (".nii.gz", ".nii"):
        if name.lower().endswith(extension):
            name = name[: -len(extension)]
            break
    return name + SUFFIX


def predict(chemin_irm: str) -> dict:
    """Segmente l'IRM T1 (sans crâne) `chemin_irm` et renvoie {"mask_path": chemin du masque NIfTI uint8}."""
    import nibabel as nib
    image = nib.load(str(chemin_irm))
    mask, _ = inference.predict(image)
    if mask.shape != image.shape[:3]:
        raise ValueError(f"masque {mask.shape} différent de la grille de l'IRM {image.shape}")
    folder = Path(os.environ.get("AVC_MASK_DIR") or tempfile.mkdtemp(prefix="avc_c8_"))
    folder.mkdir(parents=True, exist_ok=True)
    output = nib.Nifti1Image((mask > 0).astype(np.uint8), image.affine)
    output.set_qform(image.affine, code=1)
    output.set_sform(image.affine, code=1)
    output.set_data_dtype(np.uint8)
    path = folder / mask_name(chemin_irm)
    nib.save(output, path)
    return {"mask_path": str(path)}
