"""Construction des transformations à partir d'une configuration YAML (contrat C5).

    transform = build_transforms("generalization/configs/intensity.yaml")
    augmented = transform(subject)                      # torchio.Subject
    monai_transform = build_transforms(config, monai=True)   # dictionnaire {"image", "label"}
"""
from pathlib import Path

import torch
import torchio as tio
import yaml

from ..data_raw import DEFAULT_ROOT, list_subjects, load_subject
from .fourier import FDA, ReferenceBank
from .harmonization import HistogramMatching, fit_reference_quantiles
from .intensity import (RandomBrainBiasField, RandomBrainBlur, RandomBrainContrast, RandomBrainGamma,
                        RandomBrainNoise, RandomBrightness, RandomLowResolution, RandomSharpen)
from .monai_adapter import MonaiAdapter
from .style import GIN, RandConv

REGISTRY = {
    "gamma": RandomBrainGamma,
    "contrast": RandomBrainContrast,
    "noise": RandomBrainNoise,
    "blur": RandomBrainBlur,
    "bias_field": RandomBrainBiasField,
    "low_resolution": RandomLowResolution,
    "sharpen": RandomSharpen,
    "brightness": RandomBrightness,
    "randconv": RandConv,
    "gin": GIN,
    "fda": FDA,
    "histogram_matching": HistogramMatching,
    # Transformations spatiales de TorchIO (preset BigAug) : elles déplacent aussi les masques.
    "affine": tio.RandomAffine,
    "elastic": tio.RandomElasticDeformation,
    "flip": tio.RandomFlip,
}
# Seules celles-ci laissent les masques strictement intacts.
INTENSITY_NAMES = tuple(name for name in REGISTRY if name not in ("affine", "elastic", "flip"))


class ReferenceLoader:
    """Charge l'image normalisée d'un sujet de référence (objet sérialisable pour les DataLoader)."""

    def __init__(self, record, spacing):
        self.record = dict(record)
        self.spacing = spacing

    def __call__(self):
        return load_subject(self.record, spacing=self.spacing)["t1"].data[0]


def reference_records(sites, n_references, data_root):
    """`n_references` sujets des centres de référence, répartis régulièrement, sans masque vide connu."""
    if not sites:
        raise ValueError("`reference_sites` est obligatoire : le centre de référence n'est jamais codé en dur")
    records = list_subjects(data_root, sites=list(sites))
    records = records[~records["known_empty_mask"]]
    step = max(1, len(records) // n_references)
    return records.iloc[::step].head(n_references)


def load_config(config):
    """Accepte un chemin YAML, un dictionnaire, ou une liste des deux (transformations concaténées)."""
    if isinstance(config, (list, tuple)):
        configs = [load_config(item) for item in config]
        merged = dict(configs[0])
        merged["transforms"] = [entry for item in configs for entry in item.get("transforms") or []]
        return merged
    if isinstance(config, (str, Path)):
        with open(config, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    return dict(config or {})


def build_transforms(config, data_root=None, monai=False):
    """Retourne la transformation composée décrite par `config`.

    Chaque entrée de `transforms` a un `name` (voir REGISTRY), une probabilité `p` et ses plages
    de paramètres ; `enabled: false` la désactive. FDA et histogram matching lisent leurs images
    de référence dans `data_root` (argument, sinon clé `data_root` de la config, sinon données brutes).
    """
    config = load_config(config)
    data_root = Path(data_root or config.get("data_root") or DEFAULT_ROOT)
    spacing = config.get("spacing", 1.0)
    transforms = []
    for entry in config.get("transforms") or []:
        entry = dict(entry)
        name = entry.pop("name")
        if not entry.pop("enabled", True):
            continue
        if name not in REGISTRY:
            raise KeyError(f"Transformation inconnue : {name}. Disponibles : {sorted(REGISTRY)}")
        if name == "fda":
            records = reference_records(entry.pop("reference_sites", None), entry.pop("n_references", 8), data_root)
            entry["references"] = ReferenceBank([ReferenceLoader(record, spacing) for _, record in records.iterrows()])
        elif name == "histogram_matching":
            entry["reference_quantiles"] = _reference_quantiles(
                entry.pop("reference_sites", None), entry.pop("n_references", 20), data_root, spacing,
                entry.pop("reference_file", None))
        transforms.append(REGISTRY[name](**entry))
    composed = tio.Compose(transforms)
    return MonaiAdapter(composed) if monai else composed


def _reference_quantiles(sites, n_references, data_root, spacing, reference_file):
    """Calcule la fonction quantile de référence, ou la relit depuis `reference_file` s'il existe."""
    if reference_file and Path(reference_file).is_file():
        return torch.load(reference_file)
    records = reference_records(sites, n_references, data_root)
    quantiles = fit_reference_quantiles(load_subject(record, spacing=spacing) for _, record in records.iterrows())
    if reference_file:
        Path(reference_file).parent.mkdir(parents=True, exist_ok=True)
        torch.save(quantiles, reference_file)
    return quantiles
