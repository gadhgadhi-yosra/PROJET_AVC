"""Augmentations d'intensité (§3.5 a) : gamma, contraste, bruit, flou, biais de champ, basse résolution.

Les plages par défaut viennent de exploration/outputs/site_summary.csv (voir le README).
`RandomSharpen` et `RandomBrightness` complètent le preset de type BigAug.
"""
import itertools

import torch
import torch.nn.functional as F

from .base import BrainIntensityTransform, gaussian_blur, log_uniform, uniform, unit_range


class RandomBrainGamma(BrainIntensityTransform):
    """Correction gamma appliquée sur le cerveau ramené à [0, 1], puis retour à l'échelle d'origine.

    Le passage par [0, 1] rend le gamma valide quel que soit le signe des intensités
    (valeurs négatives à R029, z-score). gamma > 1 assombrit les tissus intermédiaires.
    """

    def __init__(self, gamma=(0.7, 1.85), **kwargs):
        super().__init__(**kwargs)
        self.gamma = tuple(gamma)

    def augment(self, data, brain, spacing):
        lo, hi = unit_range(data, brain)
        unit = ((data - lo) / (hi - lo)).clamp(min=0)
        return lo + (hi - lo) * unit ** log_uniform(*self.gamma)


class RandomBrainContrast(BrainIntensityTransform):
    """Étire ou resserre les intensités autour de la moyenne du cerveau."""

    def __init__(self, factor=(0.78, 1.56), **kwargs):
        super().__init__(**kwargs)
        self.factor = tuple(factor)

    def augment(self, data, brain, spacing):
        mean = data[brain].mean()
        return mean + log_uniform(*self.factor) * (data - mean)


class RandomBrainNoise(BrainIntensityTransform):
    """Bruit gaussien, écart-type exprimé en fraction de l'écart-type du cerveau."""

    def __init__(self, std=(0.0, 0.1), **kwargs):
        super().__init__(**kwargs)
        self.std = tuple(std)

    def augment(self, data, brain, spacing):
        return data + uniform(*self.std) * data[brain].std() * torch.randn(data.shape)


class RandomBrainBlur(BrainIntensityTransform):
    """Flou gaussien isotrope, écart-type en mm."""

    def __init__(self, sigma_mm=(0.25, 1.0), **kwargs):
        super().__init__(**kwargs)
        self.sigma_mm = tuple(sigma_mm)

    def augment(self, data, brain, spacing):
        return gaussian_blur(self.fill_background(data, brain), uniform(*self.sigma_mm), spacing)


class RandomSharpen(BrainIntensityTransform):
    """Masque flou : x + amount * (x - flou(x))."""

    def __init__(self, amount=(0.5, 2.0), sigma_mm=1.0, **kwargs):
        super().__init__(**kwargs)
        self.amount = tuple(amount)
        self.sigma_mm = sigma_mm

    def augment(self, data, brain, spacing):
        filled = self.fill_background(data, brain)
        return filled + uniform(*self.amount) * (filled - gaussian_blur(filled, self.sigma_mm, spacing))


class RandomBrightness(BrainIntensityTransform):
    """Décalage et facteur d'échelle sur le cerveau ramené à [0, 1] (perturbation de BigAug)."""

    def __init__(self, shift=(-0.1, 0.1), scale=(0.9, 1.1), **kwargs):
        super().__init__(**kwargs)
        self.shift = tuple(shift)
        self.scale = tuple(scale)

    def augment(self, data, brain, spacing):
        lo, hi = unit_range(data, brain)
        return lo + uniform(*self.scale) * (data - lo) + uniform(*self.shift) * (hi - lo)


class RandomBrainBiasField(BrainIntensityTransform):
    """Biais de champ multiplicatif lisse : exponentielle d'un polynôme aléatoire.

    Le champ multiplie l'intensité mesurée depuis le niveau le plus sombre du cerveau (le « zéro
    signal »), et son logarithme est centré dans le cerveau pour ne pas changer l'échelle globale.
    """

    def __init__(self, coefficients=0.2, order=3, **kwargs):
        super().__init__(**kwargs)
        self.coefficients = coefficients
        self.order = order

    def augment(self, data, brain, spacing):
        # Un polynôme d'ordre 3 est lisse : on l'évalue sur une grille grossière puis on l'agrandit.
        axes = torch.meshgrid(*[torch.linspace(-1, 1, 16)] * 3, indexing="ij")
        log_field = torch.zeros(16, 16, 16)
        for i, j, k in itertools.product(range(self.order + 1), repeat=3):
            if 0 < i + j + k <= self.order:
                log_field += uniform(-self.coefficients, self.coefficients) * axes[0] ** i * axes[1] ** j * axes[2] ** k
        log_field = F.interpolate(log_field[None, None], size=data.shape[1:], mode="trilinear", align_corners=True)[0]
        log_field = log_field.expand_as(data)
        log_field = log_field - log_field[brain].mean()
        lo = data[brain].min()
        return lo + (data - lo) * log_field.exp()


class RandomLowResolution(BrainIntensityTransform):
    """Simule une acquisition moins résolue : moyenne vers une grille grossière, puis retour.

    Tire soit une résolution isotrope (`isotropic_mm`), soit des coupes épaisses sur un seul axe
    (`axis_mm`). Les valeurs sont des tailles de voxel en mm : rien ne se passe si l'image est
    déjà moins résolue. Seule l'image est touchée, le masque garde sa résolution.
    """

    def __init__(self, isotropic_mm=(1.0, 1.5), axis_mm=(1.0, 3.0), axis_probability=0.5, **kwargs):
        super().__init__(**kwargs)
        self.isotropic_mm = tuple(isotropic_mm)
        self.axis_mm = tuple(axis_mm)
        self.axis_probability = axis_probability

    def augment(self, data, brain, spacing):
        target = list(spacing)
        if torch.rand(1).item() < self.axis_probability:
            axis = int(torch.randint(3, (1,)))
            target[axis] = max(target[axis], uniform(*self.axis_mm))
        else:
            size = uniform(*self.isotropic_mm)
            target = [max(s, size) for s in spacing]
        shape = data.shape[1:]
        coarse = [max(1, round(n * s / t)) for n, s, t in zip(shape, spacing, target)]
        if coarse == list(shape):
            return data
        filled = self.fill_background(data, brain)[None]
        low = F.interpolate(filled, size=coarse, mode="area")
        return F.interpolate(low, size=shape, mode="trilinear", align_corners=False)[0]
