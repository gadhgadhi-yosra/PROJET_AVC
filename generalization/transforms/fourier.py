"""Méthode de Fourier (§3.5 c) : FDA, Fourier Domain Adaptation (Yang & Soatto, CVPR 2020).

L'amplitude basse fréquence de l'image est remplacée par celle d'une image d'un autre centre ;
la phase, qui porte l'anatomie, est conservée.
"""
import torch
import torch.nn.functional as F

from .base import BrainIntensityTransform, uniform


def fda_swap(source, target, beta):
    """Échange l'amplitude basse fréquence de `source` (X, Y, Z) avec celle de `target`.

    Le cube échangé a pour demi-côté b = floor(beta * min(X, Y, Z)) autour de la fréquence nulle.
    b = 0 (donc beta = 0) renvoie l'image d'origine.
    """
    half = int(beta * min(source.shape))
    if half == 0:
        return source.clone()
    spectrum = torch.fft.fftshift(torch.fft.fftn(source))
    amplitude, phase = spectrum.abs(), spectrum.angle()
    target_amplitude = torch.fft.fftshift(torch.fft.fftn(target)).abs()
    # Cube symétrique autour de la fréquence nulle : le résultat reste réel.
    cube = tuple(slice(n // 2 - half, n // 2 + half + 1) for n in source.shape)
    amplitude[cube] = target_amplitude[cube]
    return torch.fft.ifftn(torch.fft.ifftshift(torch.polar(amplitude, phase))).real


class ReferenceBank:
    """Images de référence (tenseurs ou fonctions de chargement), chargées à la demande et gardées en mémoire."""

    def __init__(self, sources):
        if not len(sources):
            raise ValueError("Aucune image de référence")
        self.sources = list(sources)
        self.cache = {}

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, index):
        if index not in self.cache:
            source = self.sources[index]
            self.cache[index] = source() if callable(source) else source
        return self.cache[index]

    def sample(self):
        return self[int(torch.randint(len(self), (1,)))]


class FDA(BrainIntensityTransform):
    """Échange d'amplitude basse fréquence avec une image tirée dans `references`.

    references : `ReferenceBank` ou liste de tenseurs (X, Y, Z) d'un autre centre, normalisés comme
        l'image d'entrée. La référence est redimensionnée à la forme de l'image avant la FFT.
    beta : plage du paramètre de FDA (fraction du plus petit côté) ; 0 = identité.
    """

    def __init__(self, references, beta=(0.0, 0.03), **kwargs):
        super().__init__(**kwargs)
        self.references = references if isinstance(references, ReferenceBank) else ReferenceBank(references)
        self.beta = tuple(beta)

    def augment(self, data, brain, spacing):
        target = self.references.sample().float()
        target = F.interpolate(target[None, None], size=data.shape[1:], mode="trilinear", align_corners=False)[0, 0]
        beta = uniform(*self.beta)
        return torch.stack([fda_swap(channel, target, beta) for channel in data])
