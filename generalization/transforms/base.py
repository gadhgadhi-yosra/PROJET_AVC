"""Socle commun des transformations d'intensité : elles n'agissent que dans le cerveau.

Règles garanties par `BrainIntensityTransform` pour toutes les classes dérivées :
  - seules les images scalaires (`torchio.ScalarImage`) sont modifiées, jamais les masques ;
  - les voxels hors cerveau gardent exactement leur valeur d'origine ;
  - la forme et l'affine sont conservées ;
  - l'aléa vient du générateur de PyTorch (`torch.manual_seed` suffit pour reproduire un tirage).
"""
import math

import torch
from scipy import ndimage
from torchio.transforms import IntensityTransform
from torchio.transforms.augmentation import RandomTransform


def mode_value(data):
    """Valeur la plus fréquente, calculée sur un sous-échantillon : c'est la valeur du fond."""
    values, counts = torch.unique(data[..., ::4, ::4, ::4], return_counts=True)
    return values[counts.argmax()]


def quantile(values, q):
    """Quantile sans la limite de 16 millions d'éléments de `torch.quantile`."""
    values = values.flatten()
    k = min(max(int(round(q * (values.numel() - 1))) + 1, 1), values.numel())
    return values.kthvalue(k).values


def unit_range(data, brain):
    """(lo, hi) = minimum et 99e percentile dans le cerveau, pour ramener l'image sur [0, 1]."""
    values = data[brain]
    lo, hi = values.min(), quantile(values, 0.99)
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def uniform(low, high):
    return low + (high - low) * torch.rand(1).item()


def log_uniform(low, high):
    return math.exp(uniform(math.log(low), math.log(high)))


def gaussian_blur(volume, sigma_mm, spacing):
    """Flou gaussien d'un tenseur (C, X, Y, Z), écart-type exprimé en mm."""
    sigma = [0.0] + [sigma_mm / s for s in spacing]
    return torch.from_numpy(ndimage.gaussian_filter(volume.numpy(), sigma=sigma, mode="nearest"))


class BrainIntensityTransform(RandomTransform, IntensityTransform):
    """Classe de base : appelle `augment(data, brain, spacing)` sur chaque image scalaire.

    brain_key : nom du masque du cerveau dans le sujet (produit par `data_raw.load_subject`).
        S'il est absent, le cerveau est estimé par « voxels différents de la valeur du fond ».
    """

    def __init__(self, brain_key="brain", **kwargs):
        super().__init__(**kwargs)
        self.brain_key = brain_key

    def get_brain(self, subject, data):
        if self.brain_key in subject:
            return subject[self.brain_key].data.bool().expand_as(data)
        return data != mode_value(data)

    def apply_transform(self, subject):
        for image in self.get_images_dict(subject).values():
            data = image.data.float()
            brain = self.get_brain(subject, data)
            if not brain.any():
                continue
            augmented = self.augment(data, brain, image.spacing)
            image.set_data(torch.where(brain, augmented.to(data.dtype), data))
        return subject

    def augment(self, data, brain, spacing):
        raise NotImplementedError

    @staticmethod
    def fill_background(data, brain):
        """Fond mis au niveau le plus sombre du cerveau avant un filtrage spatial.

        Après un z-score, le fond vaut 0 (la moyenne du cerveau) : filtrer tel quel éclaircirait
        les bords. Avec ce remplissage le résultat est le même sur données brutes ou normalisées.
        """
        return torch.where(brain, data, data[brain].min())
