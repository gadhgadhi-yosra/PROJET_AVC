"""Randomisation du style (§3.5 b) : convolutions aléatoires (RandConv) et GIN.

Les deux changent la texture et la correspondance des intensités en gardant l'anatomie.
Le preset de type BigAug est une configuration : generalization/configs/bigaug.yaml.
"""
import torch
import torch.nn.functional as F

from .base import BrainIntensityTransform, uniform, unit_range


def blend_with_original(data, styled, brain, alpha, allow_inversion):
    """Mélange alpha * stylisé + (1 - alpha) * original, puis retour aux statistiques d'origine.

    Le mélange se fait entre images standardisées dans le cerveau, et le résultat retrouve la
    moyenne et l'écart-type de l'image d'origine. Sans `allow_inversion`, une image stylisée
    anti-corrélée à l'original est retournée : le mélange ne peut alors pas annuler le contraste.
    """
    mean, std = data[brain].mean(), data[brain].std()
    original = (data - mean) / std
    styled = (styled - styled[brain].mean()) / styled[brain].std().clamp_min(1e-8)
    if not allow_inversion and (original[brain] * styled[brain]).mean() < 0:
        styled = -styled
    mixed = alpha * styled + (1 - alpha) * original
    mixed = (mixed - mixed[brain].mean()) / mixed[brain].std().clamp_min(1e-3)
    return mean + std * mixed


def unit_image(data, brain):
    lo, hi = unit_range(data, brain)
    return torch.where(brain, (data - lo) / (hi - lo), torch.zeros(()))


class RandConv(BrainIntensityTransform):
    """Convolution aléatoire à une couche (Xu et al., ICLR 2021), noyau 3D tiré à chaque appel."""

    def __init__(self, kernel_sizes=(1, 3, 5, 7), mix=(0.0, 0.5), allow_inversion=False, **kwargs):
        super().__init__(**kwargs)
        self.kernel_sizes = tuple(kernel_sizes)
        self.mix = tuple(mix)
        self.allow_inversion = allow_inversion

    def augment(self, data, brain, spacing):
        size = self.kernel_sizes[int(torch.randint(len(self.kernel_sizes), (1,)))]
        channels = data.shape[0]
        weight = torch.randn(channels, 1, size, size, size) / size ** 1.5
        styled = F.conv3d(unit_image(data, brain)[None], weight, padding=size // 2, groups=channels)[0]
        return blend_with_original(data, styled, brain, uniform(*self.mix), self.allow_inversion)


class GIN(BrainIntensityTransform):
    """Global Intensity Non-linear augmentation (Ouyang et al., IEEE TMI 2022).

    Petit réseau convolutif aux poids tirés à chaque appel : `n_layers` couches, noyaux de taille
    1 ou 3, LeakyReLU entre les couches, `channels` canaux intermédiaires.
    """

    def __init__(self, n_layers=4, channels=2, kernel_sizes=(1, 3), mix=(0.0, 0.5), allow_inversion=False, **kwargs):
        super().__init__(**kwargs)
        self.n_layers = n_layers
        self.channels = channels
        self.kernel_sizes = tuple(kernel_sizes)
        self.mix = tuple(mix)
        self.allow_inversion = allow_inversion

    def augment(self, data, brain, spacing):
        hidden = unit_image(data, brain)[None]
        for layer in range(self.n_layers):
            last = layer == self.n_layers - 1
            size = self.kernel_sizes[int(torch.randint(len(self.kernel_sizes), (1,)))]
            out_channels = data.shape[0] if last else self.channels
            weight = torch.randn(out_channels, hidden.shape[1], size, size, size)
            hidden = F.conv3d(hidden, weight, torch.randn(out_channels), padding=size // 2)
            if not last:
                hidden = F.leaky_relu(hidden, 0.01)
        return blend_with_original(data, hidden[0], brain, uniform(*self.mix), self.allow_inversion)
