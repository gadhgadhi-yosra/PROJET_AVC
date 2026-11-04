"""DANN : apprentissage adversarial de caractéristiques indépendantes du centre (Ganin et al., JMLR 2016).

    model = DANNWrapper(unet, n_centers=len(centres))          # unet : monai.networks.nets.UNet
    segmentation, logits_centre = model(images)
    model.set_progress(epoch / n_epochs)                       # planification de lambda (Ganin)
    loss = dann_loss(segmentation, masks, logits_centre, centres_du_lot)

Un classifieur de centre lit les caractéristiques du bottleneck à travers une couche d'inversion
de gradient : il apprend à reconnaître le centre, tandis que l'encodeur reçoit le gradient opposé,
multiplié par lambda, et apprend donc à effacer l'empreinte du centre.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(context, features, lambda_):
        context.lambda_ = lambda_
        return features.view_as(features)

    @staticmethod
    def backward(context, grad_output):
        return -context.lambda_ * grad_output, None


class GradientReversal(nn.Module):
    """Identité à l'aller ; au retour, le gradient est multiplié par -lambda."""

    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = float(lambda_)

    def forward(self, features):
        return _GradientReversalFunction.apply(features, self.lambda_)

    def extra_repr(self):
        return f"lambda_={self.lambda_:g}"


def ganin_lambda(progress, gamma=10.0):
    """Planification de Ganin : lambda = 2 / (1 + exp(-gamma × p)) - 1, de 0 (p = 0) vers 1 (p = 1).

    La montée progressive évite que le classifieur, encore aléatoire, perturbe l'encodeur au début
    (solution de repli prévue si DANN est instable).
    """
    progress = min(max(float(progress), 0.0), 1.0)
    return 2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0


class CenterClassifier(nn.Module):
    """Classifieur de centre sur les caractéristiques du bottleneck : moyenne spatiale puis petit MLP."""

    def __init__(self, in_channels, n_centers, hidden=64, dropout=0.25):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(in_channels, hidden), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout), nn.Linear(hidden, n_centers))

    def forward(self, features):
        return self.head(self.pool(features))


def find_bottleneck(unet):
    """Module le plus profond d'un `monai.networks.nets.UNet` (sous-module de la dernière SkipConnection)."""
    from monai.networks.layers.simplelayers import SkipConnection
    block = unet.model
    while True:
        skips = [module for module in block.children() if isinstance(module, SkipConnection)]
        if not skips:
            raise ValueError("Pas de SkipConnection : passer `bottleneck=` explicitement")
        inner = skips[0].submodule
        if isinstance(inner, nn.Sequential) and any(isinstance(m, SkipConnection) for m in inner.children()):
            block = inner
            continue
        return inner


class DANNWrapper(nn.Module):
    """Enveloppe un réseau de segmentation : renvoie (segmentation, logits du centre).

    bottleneck : module dont la sortie alimente le classifieur (défaut : `find_bottleneck(unet)`).
    feature_channels : canaux de cette sortie (défaut : `unet.channels[-1]` pour un UNet MONAI).
    Les caractéristiques sont captées par un hook, sans modifier le réseau ; le gradient du
    classifieur remonte dans l'encodeur à travers la couche d'inversion.
    """

    def __init__(self, unet, n_centers, bottleneck=None, feature_channels=None, hidden=64, gamma=10.0):
        super().__init__()
        self.unet = unet
        self.gamma = gamma
        self.grl = GradientReversal(0.0)
        bottleneck = bottleneck if bottleneck is not None else find_bottleneck(unet)
        feature_channels = feature_channels or getattr(unet, "channels", [None])[-1]
        if feature_channels is None:
            raise ValueError("feature_channels est obligatoire pour ce réseau")
        self.classifier = CenterClassifier(feature_channels, n_centers, hidden)
        self._features = None
        self._hook = bottleneck.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self._features = output

    def set_progress(self, progress):
        """Fixe lambda selon l'avancement de l'entraînement (0 à 1) ; renvoie la valeur."""
        self.grl.lambda_ = ganin_lambda(progress, self.gamma)
        return self.grl.lambda_

    def forward(self, images):
        self._features = None
        segmentation = self.unet(images)
        if self._features is None:
            raise RuntimeError("Le bottleneck n'a pas été traversé : vérifier `bottleneck=`")
        logits = self.classifier(self.grl(self._features))
        self._features = None
        return segmentation, logits


def dann_loss(segmentation, target, center_logits, center_labels, weight=1.0, seg_loss=None):
    """Perte totale = Dice + CE (segmentation) + weight × CE du centre ; renvoie (total, détail).

    Le lambda de Ganin agit dans la couche d'inversion (l'encodeur reçoit -lambda × gradient) ;
    `weight` ne multiplie que la perte du centre. Avec weight = 1 (défaut, comme Ganin), le
    classifieur apprend à pleine vitesse ; mettre aussi lambda ici le ralentirait au début.
    seg_loss : défaut DiceCELoss(sigmoid=True) de MONAI, pour une sortie binaire à un canal.
    """
    if seg_loss is None:
        from monai.losses import DiceCELoss
        seg_loss = DiceCELoss(sigmoid=True)
    segmentation_term = seg_loss(segmentation, target.float())
    center_term = F.cross_entropy(center_logits, center_labels)
    total = segmentation_term + weight * center_term
    parts = {"segmentation": segmentation_term, "centre": center_term, "total": total}
    return total, {name: float(value.detach()) for name, value in parts.items()}


def center_labels(sites, centers):
    """Indices des centres d'un lot dans la liste `centers` (centres d'entraînement du pli)."""
    index = {site: position for position, site in enumerate(centers)}
    unknown = sorted(set(sites) - set(index))
    if unknown:
        raise ValueError(f"Centres absents de la liste d'entraînement : {unknown}")
    return torch.tensor([index[site] for site in sites], dtype=torch.long)
