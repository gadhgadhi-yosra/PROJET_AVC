"""Construction des transformations à partir d'une configuration YAML (contrat C5).

    transform = build_transforms("generalization/configs/intensity.yaml")
    augmented = transform(subject)                      # torchio.Subject
    monai_transform = build_transforms(config, monai=True)   # dictionnaire {"image", "label"}
"""
from pathlib import Path

import torch
import torchio as tio
import yaml

from ..calibration import check_training_sites, load_splits, restrict_to_training
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
REPO_ROOT = Path(__file__).resolve().parents[2]


class ReferenceLoader:
    """Charge l'image normalisée d'un sujet de référence (objet sérialisable pour les DataLoader)."""

    def __init__(self, record, spacing):
        self.record = dict(record)
        self.spacing = spacing

    def __call__(self):
        return load_subject(self.record, spacing=self.spacing)["t1"].data[0]


def reference_records(sites, n_references, data_root, splits=None, fold=None):
    """`n_references` sujets des centres de référence, répartis régulièrement, sans masque vide connu.

    Avec `splits` (chemin de splits.json, relatif à la racine du dépôt), les centres doivent appartenir
    au pool d'entraînement et seuls les sujets d'entraînement servent de référence (jamais le test ID).
    Avec `fold` en plus, les centres de validation du pli sont interdits et seuls les sujets
    d'entraînement du pli servent.
    """
    if not sites:
        raise ValueError("`reference_sites` est obligatoire : le centre de référence n'est jamais codé en dur")
    records = list_subjects(data_root, sites=list(sites))
    if splits:
        splits = load_splits(REPO_ROOT / splits)
        check_training_sites(sites, splits, "les images de référence", fold)
        records = restrict_to_training(records, splits, fold)
    records = records[~records["known_empty_mask"]]
    step = max(1, len(records) // n_references)
    return records.iloc[::step].head(n_references)


def reference_choice(entry, fold=None):
    """Retire de `entry` les clés de référence et renvoie (centres, splits) pour le pli demandé.

    `fold=None` : `reference_sites` (entraînement final sur tous les sujets d'entraînement).
    `fold=k` : `fold_reference_sites[k]`, qui exige la clé `splits` pour être contrôlé.
    """
    sites, by_fold, splits = (entry.pop(key, None) for key in ("reference_sites", "fold_reference_sites", "splits"))
    if fold is None:
        return sites, splits
    if not by_fold or not splits:
        raise ValueError("`fold` exige les clés `fold_reference_sites` et `splits` dans l'entrée")
    if not 0 <= fold < len(by_fold):
        raise ValueError(f"Pli {fold} absent de `fold_reference_sites` ({len(by_fold)} plis)")
    return [by_fold[fold]], splits


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


def build_transforms(config, data_root=None, monai=False, fold=None):
    """Retourne la transformation composée décrite par `config`.

    Chaque entrée de `transforms` a un `name` (voir REGISTRY), une probabilité `p` et ses plages
    de paramètres ; `enabled: false` la désactive. FDA et histogram matching lisent leurs images
    de référence dans `data_root` (argument, sinon clé `data_root` de la config, sinon données brutes).
    `fold=k` prend la référence du pli k (`fold_reference_sites`), choisie hors de ses centres de validation.
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
            sites, splits = reference_choice(entry, fold)
            records = reference_records(sites, entry.pop("n_references", 8), data_root, splits, fold)
            entry["references"] = ReferenceBank([ReferenceLoader(record, spacing) for _, record in records.iterrows()])
        elif name == "histogram_matching":
            sites, splits = reference_choice(entry, fold)
            entry["reference_quantiles"] = _reference_quantiles(
                sites, entry.pop("n_references", 20), data_root, spacing, entry.pop("reference_file", None),
                splits, fold)
        transforms.append(REGISTRY[name](**entry))
    composed = tio.Compose(transforms)
    return MonaiAdapter(composed) if monai else composed


def _reference_quantiles(sites, n_references, data_root, spacing, reference_file, splits=None, fold=None):
    """Calcule la fonction quantile de référence, ou la relit depuis `reference_file` s'il existe."""
    if splits:
        check_training_sites(sites, load_splits(REPO_ROOT / splits), "les images de référence", fold)
    if reference_file and fold is not None:
        reference_file = Path(reference_file).with_stem(f"{Path(reference_file).stem}_pli{fold}")
    if reference_file and Path(reference_file).is_file():
        return torch.load(reference_file)
    records = reference_records(sites, n_references, data_root, splits, fold)
    quantiles = fit_reference_quantiles(load_subject(record, spacing=spacing) for _, record in records.iterrows())
    if reference_file:
        Path(reference_file).parent.mkdir(parents=True, exist_ok=True)
        torch.save(quantiles, reference_file)
    return quantiles
