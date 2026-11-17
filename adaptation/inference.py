"""Contrat C8 : segmentation de la lésion d'un T1 avec le modèle final RQ3, pour la démo.

    from adaptation.inference import predict

    masque, volume_ml = predict("sub-r034s001_ses-1_space-orig_desc-brain_T1w.nii.gz")
    print(f"Lésion estimée : {volume_ml:.1f} ml")

Entrée : un T1 **déjà débarrassé du crâne**, au format des images ATLAS (`*_desc-brain_T1w.nii.gz`,
fond constant). Le chemin d'un NIfTI, une image nibabel, ou un tableau numpy 3D avec son `affine`
(défaut : identité, soit des voxels de 1 mm).

Sortie : `(masque, volume_ml)`.
- `masque` : tableau numpy uint8 (0/1) **de la même forme que l'entrée, dans la même grille** (même
  affine), prêt à être superposé à l'image ou enregistré :
  `nib.save(nib.Nifti1Image(masque, nib.load(chemin).affine), "masque.nii.gz")`.
- `volume_ml` : volume de la lésion prédite, mesuré dans l'espace de travail du modèle (voxels de 1 mm³,
  1 000 voxels = 1 ml). C'est la grandeur `pred_ml` des résultats RQ3. Le volume recompté dans la grille
  d'origine peut différer légèrement (rééchantillonnage au plus proche voisin).

Étapes, identiques au prétraitement S4 (`preprocessing/week4.py`, version s4-v1.2) appliqué aux données
d'entraînement : orientation RAS, fond = valeur la plus fréquente, cerveau = voxels au-dessus du fond,
rééchantillonnage à 1 mm (linéaire), recadrage sur le cerveau avec 5 voxels de marge, z-score dans le
cerveau, quantification au millième. Puis fenêtre glissante 96³ (recouvrement 0,25, pondération
gaussienne), sigmoïde, mise à zéro hors du cerveau, seuil 0,5 ; le masque est replacé dans la grille
d'entrée.

Checkpoint : `checkpoint=` ou la variable d'environnement `PROJET_AVC_MODELE`, sinon
`data/modeles/unet3d_intensity_pli0_best.pt`. Le modèle est chargé une fois puis gardé en mémoire.
GPU si disponible, sinon CPU (environ une minute par volume).

Limites : modèle du pli 0 (entraîné sans les centres de validation et de test du pli), lésions de moins
de 1 ml mal détectées (Dice OOD ≈ 0,3), lésions aiguës (J1) invisibles en T1 non détectées (centre R008,
voir adaptation/outputs/rq3/echecs/echecs.md). Outil de démonstration, pas de diagnostic.
"""
import functools
import os
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "data" / "modeles" / "unet3d_intensity_pli0_best.pt"
# Constantes du prétraitement S4 (preprocessing/week4.py, s4-v1.2).
SPACING_MM = (1.0, 1.0, 1.0)
CROP_MARGIN = 5
Z_STEP = 1e-3
THRESHOLD = 0.5


# --- Prétraitement ---

def as_image(volume, affine=None):
    """Image nibabel à partir d'un chemin, d'une image nibabel ou d'un tableau 3D (+ affine)."""
    import nibabel as nib
    if isinstance(volume, (str, Path)):
        return nib.load(str(volume))
    if isinstance(volume, nib.spatialimages.SpatialImage):
        return volume
    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError(f"volume 3D attendu, reçu la forme {array.shape}")
    return nib.Nifti1Image(array.astype(np.float32), np.eye(4) if affine is None else np.asarray(affine))


def preprocess(image):
    """Prétraitement S4 de l'image seule.

    Renvoie un dict : `t1` (tenseur 1 × X × Y × Z, z-score), `brain` (tenseur booléen de même forme),
    `grid` (image nibabel 1 mm non recadrée, sert à replacer le masque) et `crop` (tranches du recadrage).
    """
    import nibabel as nib
    from nibabel.processing import resample_from_to, resample_to_output
    if len(image.shape) != 3:
        raise ValueError(f"volume 3D attendu, reçu la forme {image.shape}")
    canonical = nib.as_closest_canonical(image)
    x = canonical.get_fdata(dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("voxels non finis dans l'image")
    values, counts = np.unique(x[::4, ::4, ::4], return_counts=True)
    background = float(values[counts.argmax()])
    brain = x > background
    if np.count_nonzero(brain) < 10:
        raise ValueError("cerveau vide : l'image doit être un T1 sans crâne, sur fond constant")
    resampled = resample_to_output(nib.Nifti1Image(x, canonical.affine), SPACING_MM, order=1, cval=background)
    support = np.asarray(resample_from_to(nib.Nifti1Image(brain.astype(np.uint8), canonical.affine),
                                          (resampled.shape, resampled.affine), order=0).dataobj, dtype=bool)
    x = resampled.get_fdata(dtype=np.float32)
    if not support.any():
        raise ValueError("cerveau perdu au rééchantillonnage")
    bounds = []
    for axis in range(3):
        indices = np.flatnonzero(support.any(axis=tuple(i for i in range(3) if i != axis)))
        bounds.append(slice(max(0, int(indices[0]) - CROP_MARGIN), min(x.shape[axis], int(indices[-1]) + CROP_MARGIN + 1)))
    crop = tuple(bounds)
    mean, std = float(x[support].mean(dtype=np.float64)), float(x[support].std(dtype=np.float64))
    if not np.isfinite(std) or std < 1e-8:
        raise ValueError("intensité constante dans le cerveau")
    normalized = np.zeros(x.shape, np.float32)
    normalized[support] = (x[support] - mean) / std
    normalized = (np.rint(normalized[crop] / Z_STEP) * Z_STEP).astype(np.float32)  # comme le stockage int16
    return dict(t1=torch.from_numpy(normalized)[None], brain=torch.from_numpy(support[crop].copy())[None],
                grid=resampled, crop=crop)


def to_input_grid(mask, prepared, image):
    """Replace le masque (espace 1 mm recadré) dans la grille de l'image d'entrée (plus proche voisin)."""
    import nibabel as nib
    from nibabel.processing import resample_from_to
    full = np.zeros(prepared["grid"].shape, np.uint8)
    full[prepared["crop"]] = mask
    back = resample_from_to(nib.Nifti1Image(full, prepared["grid"].affine), (image.shape, image.affine), order=0)
    return np.asarray(back.dataobj, dtype=np.uint8)


# --- Modèle ---

@functools.lru_cache(maxsize=2)
def load_segmenter(checkpoint=None, device=None):
    """(réseau de segmentation en mode évaluation, RunConfig) d'un checkpoint de `adaptation.train`.

    Seuls les poids et la configuration sont lus : ni données, ni splits.json. Pour un modèle DANN,
    seul le réseau de segmentation est gardé (le classifieur de centre ne sert pas à la prédiction).
    """
    from dataclasses import fields

    from .train import RunConfig, build_network
    path = Path(checkpoint or os.environ.get("PROJET_AVC_MODELE") or DEFAULT_CHECKPOINT)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint introuvable : {path} (best.pt du run, voir la docstring)")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    saved = torch.load(path, map_location="cpu", weights_only=False)
    known = {f.name for f in fields(RunConfig)}
    config = {k: v for k, v in saved["config"].items() if k in known}
    for key in ("channels", "strides", "evaluation_splits"):
        if key in config:
            config[key] = tuple(config[key])
    cfg = RunConfig(**config)
    network = build_network(cfg)
    weights = saved["model"]
    if any(key.startswith("unet.") for key in weights):  # DANNWrapper : poids du réseau sous « unet. »
        weights = {key[len("unet."):]: value for key, value in weights.items() if key.startswith("unet.")}
    network.load_state_dict(weights)
    return network.to(device).eval(), cfg


# --- Prédiction ---

def predict(volume, checkpoint=None, affine=None, device=None, threshold=THRESHOLD):
    """Segmente la lésion d'un T1 sans crâne ; renvoie `(masque, volume_ml)`.

    volume : chemin d'un NIfTI, image nibabel, ou tableau numpy 3D (avec `affine`, défaut identité = 1 mm).
    masque : uint8 0/1, même forme et même grille que l'entrée.
    volume_ml : volume prédit en ml, compté sur la grille de 1 mm du modèle (= `pred_ml` des résultats RQ3).

    Exemple :
        >>> import nibabel as nib
        >>> from adaptation.inference import predict
        >>> chemin = "sub-r034s001_ses-1_space-orig_desc-brain_T1w.nii.gz"
        >>> masque, volume_ml = predict(chemin)
        >>> nib.save(nib.Nifti1Image(masque, nib.load(chemin).affine), "lesion_predite.nii.gz")
    """
    from .train import predict as sliding_window
    network, cfg = load_segmenter(None if checkpoint is None else str(checkpoint),
                                  None if device is None else str(device))
    image = as_image(volume, affine)
    prepared = preprocess(image)
    _, probability = sliding_window(network, prepared["t1"], prepared["brain"], cfg)
    mask = (probability[0] > threshold).numpy().astype(np.uint8)
    volume_ml = float(mask.sum()) * float(np.prod(SPACING_MM)) / 1000.0
    return to_input_grid(mask, prepared, image), volume_ml
