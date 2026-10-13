"""Adaptateur vers le format dictionnaire de MONAI : {"image", "label"} (+ "brain" facultatif)."""
import numpy as np
import torch
import torchio as tio


class MonaiAdapter:
    """Rend une transformation TorchIO utilisable dans un `monai.transforms.Compose`.

    Les tableaux sont en canal d'abord (C, X, Y, Z), comme après `EnsureChannelFirstd`. L'affine
    vient du `MetaTensor` de l'image s'il existe, sinon l'identité (voxels de 1 mm). Si la clé
    `brain_key` est absente, le cerveau est estimé par les voxels différents du fond.
    """

    def __init__(self, transform, image_key="image", label_key="label", brain_key="brain"):
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key
        self.brain_key = brain_key

    def __call__(self, data):
        data = dict(data)
        image = data[self.image_key]
        affine = getattr(image, "affine", None)
        affine = np.eye(4) if affine is None else np.asarray(affine, dtype=np.float64)
        subject = tio.Subject(t1=tio.ScalarImage(tensor=_plain(image).float(), affine=affine))
        for name, key in (("mask", self.label_key), ("brain", self.brain_key)):
            if key in data:
                subject[name] = tio.LabelMap(tensor=_plain(data[key]), affine=affine)
        transformed = self.transform(subject)
        data[self.image_key] = _like(image, transformed["t1"].data)
        for name, key in (("mask", self.label_key), ("brain", self.brain_key)):
            if key in data:
                data[key] = _like(data[key], transformed[name].data)
        return data


def _plain(array):
    return torch.as_tensor(np.asarray(array)) if not isinstance(array, torch.Tensor) else array.as_subclass(torch.Tensor)


def _like(original, tensor):
    """Redonne au résultat le type, le dtype et les métadonnées MONAI de l'entrée."""
    if not isinstance(original, torch.Tensor):
        return tensor.numpy().astype(original.dtype)
    tensor = tensor.to(original.dtype)
    if type(original) is torch.Tensor:
        return tensor
    from monai.data import MetaTensor
    return MetaTensor(tensor, meta=original.meta, applied_operations=original.applied_operations)
