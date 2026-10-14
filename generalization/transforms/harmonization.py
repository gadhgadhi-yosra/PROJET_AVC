"""Harmonisation (§3.5 d) : histogram matching vers un centre de référence.

C'est une baseline de prétraitement déterministe, pas une augmentation aléatoire : chaque image
reçoit la distribution d'intensité moyenne du centre de référence (dans le cerveau).
"""
import torch

from .base import BrainIntensityTransform


def fit_reference_quantiles(subjects, n_quantiles=1001, image_key="t1", brain_key="brain", max_voxels=1_000_000):
    """Fonction quantile moyenne du cerveau sur les sujets de référence, tenseur (n_quantiles,).

    Moyenner les fonctions quantiles donne l'histogramme « moyen » du centre (barycentre de
    Wasserstein en 1D). Les sujets doivent être normalisés comme les images à harmoniser.
    """
    levels = torch.linspace(0, 1, n_quantiles)
    curves = []
    for subject in subjects:
        values = subject[image_key].data.float()[subject[brain_key].data.bool()]
        values = values[:: max(1, values.numel() // max_voxels)]
        curves.append(torch.quantile(values, levels))
    if not curves:
        raise ValueError("Aucun sujet de référence")
    return torch.stack(curves).mean(dim=0)


def interpolate(x, xp, fp):
    """Équivalent de numpy.interp pour des abscisses `xp` strictement croissantes."""
    index = torch.searchsorted(xp, x).clamp(1, len(xp) - 1)
    weight = (x - xp[index - 1]) / (xp[index] - xp[index - 1])
    return fp[index - 1] + weight * (fp[index] - fp[index - 1])


class HistogramMatching(BrainIntensityTransform):
    """Applique aux intensités du cerveau la fonction quantile de référence.

    Chaque valeur est remplacée par le quantile de référence de même rang. Le rang moyen des
    valeurs égales est utilisé, ce qui reste correct pour les images à peu de niveaux de gris (R029).
    """

    def __init__(self, reference_quantiles, **kwargs):
        kwargs.setdefault("p", 1.0)
        super().__init__(**kwargs)
        self.reference_quantiles = torch.as_tensor(reference_quantiles, dtype=torch.float32)

    def augment(self, data, brain, spacing):
        values, inverse, counts = torch.unique(data[brain], return_inverse=True, return_counts=True)
        rank = (counts.cumsum(0) - 0.5 * counts) / inverse.numel()
        levels = torch.linspace(0, 1, len(self.reference_quantiles))
        matched = data.clone()
        matched[brain] = interpolate(rank.float(), levels, self.reference_quantiles)[inverse]
        return matched
