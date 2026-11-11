"""TENT : adaptation au test par minimisation de l'entropie (Wang et al., ICLR 2021).

    model = configure_tent(model)                      # seules les couches de normalisation apprennent
    tent = Tent(model, torch.optim.Adam(collect_params(model), lr=1e-3))
    probabilities = tent(images, brain=cerveaux)       # adapte sur le lot, sans étiquettes
    tent.reset()                                       # retour exact à l'état d'origine

Sans étiquettes : sur un nouveau centre, le modèle rend ses prédictions plus tranchées en ne
modifiant que les paramètres affines (gamma, beta) des couches de normalisation. BatchNorm utilise
les statistiques du lot courant au lieu des moyennes d'entraînement.
"""
from copy import deepcopy

import torch
from torch import nn

NORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm1d, nn.InstanceNorm2d,
              nn.InstanceNorm3d, nn.GroupNorm)
BATCH_NORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def norm_layers(model):
    return [module for module in model.modules() if isinstance(module, NORM_TYPES)]


def collect_params(model):
    """Paramètres affines (weight, bias) des couches de normalisation, dans un ordre stable."""
    return [parameter for module in norm_layers(model) for name, parameter in module.named_parameters(recurse=False)
            if name in ("weight", "bias")]


def configure_tent(model):
    """Prépare `model` pour TENT et le renvoie.

    - mode entraînement (BatchNorm sur les statistiques du lot, sans mise à jour des moyennes) ;
    - tout gelé sauf les paramètres affines des BatchNorm, InstanceNorm et GroupNorm.
    Erreur si le réseau n'a aucune normalisation affine : TENT n'aurait rien à adapter (cas du
    UNet MONAI par défaut, dont les InstanceNorm3d n'ont pas de paramètres affines).
    """
    model.train()
    model.requires_grad_(False)
    for module in norm_layers(model):
        if isinstance(module, BATCH_NORM_TYPES):
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
        for name, parameter in module.named_parameters(recurse=False):
            if name in ("weight", "bias"):
                parameter.requires_grad_(True)
    if not collect_params(model):
        raise ValueError("Aucune couche de normalisation affine : TENT n'a rien à adapter "
                         "(InstanceNorm avec affine=True, BatchNorm ou GroupNorm nécessaires)")
    return model


def voxel_entropy(logits, brain=None):
    """Entropie moyenne par voxel, dans le cerveau si `brain` est donné (forme N × 1 × X × Y × Z).

    Un canal : sigmoïde (lésion / fond) ; plusieurs canaux : softmax.
    """
    if logits.shape[1] == 1:
        probability = torch.sigmoid(logits)
        entropy = -(probability * torch.log(probability.clamp_min(1e-8))
                    + (1 - probability) * torch.log((1 - probability).clamp_min(1e-8)))
    else:
        log_probability = torch.log_softmax(logits, dim=1)
        entropy = -(log_probability.exp() * log_probability).sum(dim=1, keepdim=True)
    if brain is None:
        return entropy.mean()
    weight = brain.to(entropy.dtype)
    if weight.sum() == 0:
        raise ValueError("Masque du cerveau vide : entropie indéfinie")
    return (entropy * weight).sum() / weight.sum()


class Tent(nn.Module):
    """Adapte `model` (déjà passé dans `configure_tent`) à chaque appel ; `reset()` restaure l'état d'origine.

    steps : pas de minimisation par lot. L'appel renvoie les logits du dernier passage avant mise à
    jour, comme l'implémentation de référence (le lot est évalué puis sert à l'adaptation).
    """

    def __init__(self, model, optimizer, steps=1):
        super().__init__()
        if steps < 1:
            raise ValueError("steps doit valoir au moins 1")
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self._model_state = deepcopy(model.state_dict())
        self._optimizer_state = deepcopy(optimizer.state_dict())
        self.history = []

    def forward(self, images, brain=None):
        for _ in range(self.steps):
            logits = self.model(images)
            loss = voxel_entropy(logits, brain)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.history.append(float(loss.detach()))
        return logits.detach()

    def reset(self):
        """Remet exactement les poids et l'état de l'optimiseur d'origine (avant toute adaptation)."""
        self.model.load_state_dict(self._model_state)
        self.optimizer.load_state_dict(self._optimizer_state)
        self.history = []
