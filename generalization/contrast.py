"""Visibilité de la lésion : rapport d'intensité lésion / tissu voisin.

Même définition que `lesion_vs_shell` de exploration/01_compute_stats.py : médiane dans la lésion
divisée par la médiane d'une couronne de 3 voxels autour d'elle (dans le cerveau). Une valeur
proche de 1 signifie une lésion presque invisible (R014, R008, R039 ≈ 0,91).
"""
import numpy as np
from scipy import ndimage


def lesion_shell_ratio(image, lesion, brain, norm_mean=0.0, norm_std=1.0, shell_voxels=3):
    """Rapport lésion / couronne, ou NaN sans lésion.

    image, lesion, brain : tableaux (X, Y, Z). Pour une image en z-score, passer `norm_mean` et
    `norm_std` (fournis par `data_raw.load_subject`) : le rapport est calculé sur les intensités
    brutes équivalentes, les seules où un rapport a un sens.
    """
    image = np.asarray(image, dtype=np.float64) * norm_std + norm_mean
    lesion, brain = np.asarray(lesion, dtype=bool), np.asarray(brain, dtype=bool)
    inside = lesion & brain
    shell = ndimage.binary_dilation(lesion, iterations=shell_voxels) & ~lesion & brain
    if not inside.any() or not shell.any():
        return float("nan")
    return float(np.median(image[inside]) / np.median(image[shell]))


def subject_ratio(subject):
    """Rapport lésion / couronne d'un `torchio.Subject` produit par `data_raw.load_subject`."""
    return lesion_shell_ratio(subject["t1"].data[0].numpy(), subject["mask"].data[0].numpy(),
                              subject["brain"].data[0].numpy(),
                              subject.get("norm_mean", 0.0), subject.get("norm_std", 1.0))
