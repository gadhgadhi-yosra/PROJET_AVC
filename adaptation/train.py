"""Boucle d'entraînement MONAI 3D commune (contrat C4) pour le tableau d'ablation RQ3.

    .venv-dl\\Scripts\\python -m adaptation.train --technique baseline --fold 0 --smoke        # essai local (4 Go)
    python -m adaptation.train --technique fda --fold 0 --data-root /kaggle/input/...   # 40 000 itérations
    python -m adaptation.train --evaluate-only --checkpoint runs/baseline_pli0/best.pt --tent  # TENT au test

Même réseau, mêmes plis, mêmes graines pour toutes les techniques ; seuls les points d'accroche changent :
- `augment` : transformation TorchIO appliquée au volume entier avant le découpage en patches
  (`build_transforms` des configurations de generalization/, calibrées sur C1) ;
- `wrap_model` et `extra_loss` : DANN (classifieur de centre, perte adversariale) ;
- `test_transform` : prétraitement déterministe aussi appliqué à la validation et au test
  (histogram matching) ;
- `--tent` : adaptation au test par centre, sans étiquettes, sur un modèle déjà entraîné.

Sorties dans `--output` (jamais dans Git) : `last.pt`, `best.pt`, `journal_entrainement.csv`,
`resultats_<split>.csv` (une ligne par sujet, métriques de baselines/common.py) et `resume.json`.
"""
import argparse
import collections
import copy
import csv
import functools
import itertools
import json
import math
import os
import pickle
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.common import metrics  # noqa: E402  (métriques ISLES communes à l'équipe)
from generalization.calibration import DEFAULT_SPLITS, load_splits, training_sites  # noqa: E402
from generalization.data_processed import DEFAULT_PROCESSED, list_processed, load_processed_subject, read_table  # noqa: E402
from generalization.transforms import build_transforms  # noqa: E402

from .dann import DANNWrapper, center_labels  # noqa: E402
from .tent import Tent, collect_params, configure_tent  # noqa: E402

CONFIGS = REPO_ROOT / "generalization" / "configs"
# Technique -> configurations d'augmentation, DANN, prétraitement aussi appliqué au test.
TECHNIQUES = {
    "baseline": dict(configs=["baseline.yaml"]),
    "intensity": dict(configs=["intensity.yaml"]),
    "bigaug": dict(configs=["bigaug.yaml"]),
    "fda": dict(configs=["fda.yaml"]),
    "histogram_matching": dict(configs=["histogram_matching.yaml"], test_time=True),
    "dann": dict(configs=["baseline.yaml"], dann=True),
}
METRIC_COLUMNS = ["dice", "hd95_mm", "abs_vol_diff_ml", "lesion_f1", "lesion_count_diff", "gt_ml", "pred_ml"]


@dataclass
class RunConfig:
    technique: str = "baseline"
    fold: int | None = 0                    # None : entraînement final sur tous les sujets d'entraînement
    data_root: str = str(DEFAULT_PROCESSED)
    splits: str = str(DEFAULT_SPLITS)
    output: str = str(REPO_ROOT / "adaptation" / "runs")
    seed: int = 2026
    patch: int = 96
    patches_per_volume: int = 2             # un volume chargé et augmenté donne ce nombre de patches
    volumes_per_batch: int = 2              # lot = volumes_per_batch × patches_per_volume patches
    lesion_fraction: float = 0.5            # part des patches centrés sur la lésion (sinon dans le cerveau)
    channels: tuple = (32, 64, 128, 256, 320)
    strides: tuple = (2, 2, 2, 2)
    num_res_units: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    # Même nombre d'itérations pour toutes les techniques (comparaison équitable) ; le budget de temps
    # n'est qu'un garde-fou pour tenir dans une session Kaggle de 12 h avec l'évaluation.
    max_iterations: int | None = 40000
    time_budget_hours: float = 10.0
    validate_every: int = 2000              # itérations
    log_every: int = 100                    # ligne de progression (durée par itération, temps restant)
    validation_limit: int | None = None     # sujets de validation (None : tous)
    checkpoint_minutes: float = 20.0
    num_workers: int = 4
    # fp32 par défaut : en fp16 (autocast + GradScaler), la passe avant de la baseline donne des patchs entiers
    # NaN dès ~170 itérations et l'échelle du scaler s'effondre (diagnostic rq3-diag-nan, variantes A à D).
    amp: bool = False
    grad_clip: float | None = 12.0          # norme globale des gradients (None : pas de clipping)
    # DANN : la perte du centre vaut ~ln(39) = 3,7 au départ contre ~1 pour la segmentation ; avec un poids 1,
    # elle domine la perte totale et la norme des gradients (clipping à 12). Poids 0,1 : terme adversarial
    # présent mais inférieur à la segmentation ; l'encodeur reçoit au plus 0,1 × le gradient du centre.
    dann_weight: float = 0.1
    dann_gamma: float = 5.0                 # montée de Ganin plus lente (10 dans l'article)
    dann_start: float = 0.05                # lambda nul sur les 5 premiers % (2 000 itérations)
    sw_overlap: float = 0.25
    sw_batch: int = 2
    evaluation_splits: tuple = ("id_test", "ood_test")
    evaluation_limit: int | None = None
    tent_learning_rate: float = 1e-4
    tent_patches: int = 4                   # patches par sujet pour adapter TENT
    smoke: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def name(self):
        fold = "final" if self.fold is None else f"pli{self.fold}"
        return f"unet3d_{self.technique}_{fold}" + ("_smoke" if self.smoke else "")


def smoke_config(cfg):
    """Essai rapide sur un GPU de 4 Go : petit réseau, quelques sujets, quelques itérations."""
    return RunConfig(**{**asdict(cfg), "channels": (16, 32, 64, 128), "strides": (2, 2, 2), "num_res_units": 1,
                        "max_iterations": 20, "validate_every": 10, "log_every": 5, "validation_limit": 2, "evaluation_limit": 2,
                        "num_workers": 0, "time_budget_hours": 0.25, "checkpoint_minutes": 1e9,
                        "tent_patches": 2, "smoke": True})


# --- Graines ---

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_seed(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed((seed + worker_id) % 2**32)


# --- Reprise exacte après une coupure (session Kaggle ou Colab) ---
# Sans reprise, ces outils ne changent rien : mêmes indices tirés, aucun tirage aléatoire supplémentaire.

def rng_state():
    """État sérialisé des générateurs du processus courant (torch, numpy, random) ; ne tire rien."""
    return pickle.dumps((torch.get_rng_state(), np.random.get_state(), random.getstate()))


def set_rng_state(blob):
    torch_state, numpy_state, python_state = pickle.loads(blob)
    torch.set_rng_state(torch_state)
    np.random.set_state(numpy_state)
    random.setstate(python_state)


def resume_worker(worker_id, states=()):
    """Initialisation d'un worker à la reprise : état du worker qui aurait produit le même lot sans coupure."""
    worker_seed(worker_id)
    if worker_id < len(states):
        set_rng_state(states[worker_id])


class TrackedDataset(torch.utils.data.Dataset):
    """Chaque élément avec l'état aléatoire du processus qui vient de le produire (pour last.pt)."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index], rng_state()


def collate_tracked(items):
    batch = collate_patches([item for item, _ in items])
    batch["rng"] = items[-1][1]  # état du worker après le dernier volume du lot
    return batch


class ResumableSampler(torch.utils.data.Sampler):
    """Même suite d'indices que RandomSampler(replacement=True, generator=Generator().manual_seed(seed)),
    à partir du rang `start` (volumes déjà consommés avant la coupure)."""

    def __init__(self, dataset, num_samples, seed, start=0):
        self.dataset, self.num_samples, self.seed, self.start = dataset, num_samples, seed, start

    def __iter__(self):
        base = torch.utils.data.RandomSampler(self.dataset, replacement=True, num_samples=self.num_samples,
                                              generator=torch.Generator().manual_seed(self.seed))
        return itertools.islice(iter(base), self.start, None)

    def __len__(self):
        return self.num_samples - self.start


# --- Données ---

def subject_arrays(subject):
    """(image, lésion, cerveau) en tenseurs 1 × X × Y × Z (float32, float32, bool)."""
    return subject["t1"].data.float(), subject["mask"].data.float(), subject["brain"].data.bool()


def pad_to(tensor, size, value=0.0):
    pads = []
    for n in reversed(tensor.shape[1:]):
        missing = max(size - n, 0)
        pads += [missing // 2, missing - missing // 2]
    return F.pad(tensor, pads, value=value) if any(pads) else tensor


def sample_patch(image, lesion, brain, size, lesion_fraction, generator):
    """Patch `size`³ centré sur un voxel de lésion (proportion `lesion_fraction`) ou du cerveau."""
    image, lesion, brain = (pad_to(t, size) for t in (image, lesion, brain.float()))
    pool = lesion[0] > 0 if lesion.any() and torch.rand(1, generator=generator).item() < lesion_fraction else brain[0] > 0
    candidates = pool.nonzero()
    centre = candidates[torch.randint(len(candidates), (1,), generator=generator).item()]
    starts = [min(max(int(c) - size // 2, 0), n - size) for c, n in zip(centre, image.shape[1:])]
    region = tuple(slice(s, s + size) for s in starts)
    return image[(slice(None), *region)], lesion[(slice(None), *region)], brain[(slice(None), *region)].bool()


class PatchDataset(torch.utils.data.Dataset):
    """Chaque élément : un sujet C1 chargé, augmenté en entier, puis découpé en patches (miroir gauche-droite)."""

    def __init__(self, records, augment, cfg, centers=None):
        self.records = records.reset_index(drop=True)
        self.augment = augment
        self.cfg = cfg
        self.centers = centers

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        subject = load_processed_subject(self.records.iloc[index])
        if self.augment is not None:
            subject = self.augment(subject)
        image, lesion, brain = subject_arrays(subject)
        generator = torch.Generator().manual_seed(int(torch.randint(2**31, (1,)).item()))
        patches = []
        for _ in range(self.cfg.patches_per_volume):
            patch = sample_patch(image, lesion, brain, self.cfg.patch, self.cfg.lesion_fraction, generator)
            if torch.rand(1, generator=generator).item() < 0.5:  # miroir gauche-droite, commun à toutes les runs
                patch = tuple(t.flip(1) for t in patch)
            patches.append(patch)
        site = self.records.iloc[index]["site"]
        return patches, site


def collate_patches(items):
    images, lesions, brains, sites = [], [], [], []
    for patches, site in items:
        for image, lesion, brain in patches:
            images.append(image)
            lesions.append(lesion)
            brains.append(brain)
            sites.append(site)
    return {"image": torch.stack(images), "label": torch.stack(lesions), "brain": torch.stack(brains), "site": sites}


# --- Réseau ---

def build_network(cfg):
    from monai.networks.nets import UNet
    # Normalisation d'instance AFFINE : nécessaire pour TENT (paramètres gamma / beta à adapter).
    return UNet(spatial_dims=3, in_channels=1, out_channels=1, channels=tuple(cfg.channels),
                strides=tuple(cfg.strides), num_res_units=cfg.num_res_units, norm=("instance", {"affine": True}))


def segmentation_network(model):
    """Réseau de segmentation seul (sans classifieur de centre) pour la prédiction."""
    return model.unet if isinstance(model, DANNWrapper) else model


def split_outputs(outputs):
    return outputs if isinstance(outputs, tuple) else (outputs, None)


# --- Points d'accroche par technique ---

def technique_hooks(cfg, centers):
    """(augment, wrap_model, extra_loss, test_transform) de la technique `cfg.technique`."""
    if cfg.technique not in TECHNIQUES:
        raise ValueError(f"Technique inconnue : {cfg.technique} (attendues : {', '.join(TECHNIQUES)})")
    spec = TECHNIQUES[cfg.technique]
    configs = [CONFIGS / name for name in spec["configs"]]
    augment = build_transforms(configs, data_root=cfg.data_root, fold=cfg.fold, source="processed")
    if len(augment) == 0:
        augment = None
    test_transform = augment if spec.get("test_time") else None
    wrap_model = extra_loss = None
    if spec.get("dann"):
        def wrap_model(network):
            return DANNWrapper(network, n_centers=len(centers), gamma=cfg.dann_gamma, start=cfg.dann_start)

        def extra_loss(aux, batch):
            labels = center_labels(batch["site"], centers).to(aux.device)
            loss = F.cross_entropy(aux.float(), labels)
            return cfg.dann_weight * loss, {"centre": float(loss.detach())}
    return augment, wrap_model, extra_loss, test_transform


# --- Prédiction et évaluation ---

@torch.no_grad()
def predict(model, image, brain=None, cfg=None):
    """Masque binaire et probabilité d'un volume 1 × X × Y × Z, par fenêtre glissante de `cfg.patch`³.

    La probabilité est mise à 0 hors du cerveau quand `brain` est donné.
    """
    from monai.inferers import sliding_window_inference
    cfg = cfg or RunConfig()
    network = segmentation_network(model)
    device = next(network.parameters()).device
    volume = pad_to(image.float(), cfg.patch)[None].to(device)
    with torch.autocast(device.type, enabled=cfg.amp and device.type == "cuda"):
        logits = sliding_window_inference(volume, (cfg.patch,) * 3, cfg.sw_batch, network, overlap=cfg.sw_overlap,
                                          mode="gaussian")
    probability = torch.sigmoid(logits.float())[0].cpu()
    probability = unpad(probability, image.shape[1:])
    if brain is not None:
        probability = probability * brain.float()
    return (probability > 0.5).to(torch.uint8), probability


def unpad(tensor, shape):
    region = []
    for n, target in zip(tensor.shape[1:], shape):
        start = (n - target) // 2
        region.append(slice(start, start + target))
    return tensor[(slice(None), *region)]


def evaluate(model, cfg, split, test_transform=None, tent=False, writer=None):
    """Une ligne de métriques par sujet du découpage (du pli si `cfg.fold` est donné)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        records = list_processed(cfg.data_root, split, cfg.fold if split != "ood_test" else None,
                                 splits=cfg.splits, on_missing="warn")
    if cfg.evaluation_limit:
        records = records.groupby("site").head(1).head(cfg.evaluation_limit)
    days = read_table(cfg.data_root).set_index("case_id").get("days_post_stroke")
    rows = []
    for site, group in records.groupby("site", sort=True):
        adapted = tent_model(model, cfg) if tent else None  # TENT repart de zéro pour chaque centre
        for _, record in group.iterrows():
            start = time.perf_counter()
            subject = load_processed_subject(record)
            if test_transform is not None:
                subject = test_transform(subject)
            image, lesion, brain = subject_arrays(subject)
            if adapted is not None:
                adapt_tent(adapted, image, lesion, brain, cfg)
            prediction, _ = predict(adapted.model if adapted is not None else model, image, brain, cfg)
            row = dict(model="unet3d", technique=cfg.technique + ("+tent" if tent else ""),
                       fold="final" if cfg.fold is None else cfg.fold, case_id=record["case_id"], site=site,
                       split=split, days_post_stroke=None if days is None else days.get(record["case_id"]),
                       **metrics(prediction[0].numpy().astype(bool), lesion[0].numpy() > 0),
                       seconds=round(time.perf_counter() - start, 2))
            rows.append(row)
            if writer is not None:
                writer(row)
    return rows


def tent_model(model, cfg):
    network = copy.deepcopy(segmentation_network(model))
    network = configure_tent(network)
    return Tent(network, torch.optim.Adam(collect_params(network), lr=cfg.tent_learning_rate), steps=1)


def adapt_tent(tent, image, lesion, brain, cfg):
    """Adapte TENT sur quelques patches du cerveau du sujet (sans utiliser la lésion)."""
    device = next(tent.model.parameters()).device
    generator = torch.Generator().manual_seed(cfg.seed)
    for _ in range(cfg.tent_patches):
        patch, _, patch_brain = sample_patch(image, torch.zeros_like(lesion), brain, cfg.patch, 0.0, generator)
        with torch.autocast(device.type, enabled=cfg.amp and device.type == "cuda"):
            tent(patch[None].to(device), patch_brain[None].to(device))


def summarize(rows):
    """Dice moyen par découpage et écart ID - OOD (sujets à lésion non vide)."""
    summary = {}
    for split in sorted({row["split"] for row in rows}):
        dice = [row["dice"] for row in rows if row["split"] == split and not math.isnan(row["dice"])]
        summary[split] = dict(sujets=len([r for r in rows if r["split"] == split]),
                              dice_moyen=float(np.mean(dice)) if dice else float("nan"))
    if "id_test" in summary and "ood_test" in summary:
        summary["ecart_id_ood"] = summary["id_test"]["dice_moyen"] - summary["ood_test"]["dice_moyen"]
    return summary


class CsvWriter:
    """Écrit un CSV ligne par ligne ; `append=True` continue un fichier existant (reprise)."""

    def __init__(self, path, columns, append=False):
        self.path, self.columns = Path(path), columns
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not (append and self.path.is_file()):
            with open(self.path, "w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, columns).writeheader()

    def __call__(self, row):
        with open(self.path, "a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, self.columns, extrasaction="ignore").writerow(row)


RESULT_COLUMNS = ["model", "technique", "fold", "case_id", "site", "split", "days_post_stroke", *METRIC_COLUMNS, "seconds"]


# --- Entraînement ---

def training_records(cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train = list_processed(cfg.data_root, "train", cfg.fold, splits=cfg.splits, on_missing="warn")
        validation = (list_processed(cfg.data_root, "val", cfg.fold, splits=cfg.splits, on_missing="warn")
                      if cfg.fold is not None else train.iloc[:0])
    if cfg.smoke:
        train = train[~train["empty_mask"]].groupby("site").head(1).head(4)
    if cfg.validation_limit:
        validation = validation.groupby("site").head(1).head(cfg.validation_limit)
    return train, validation


def validate(model, records, cfg, test_transform):
    model.eval()
    dice = []
    for _, record in records.iterrows():
        subject = load_processed_subject(record)
        if test_transform is not None:
            subject = test_transform(subject)
        image, lesion, brain = subject_arrays(subject)
        if lesion.any():
            prediction, _ = predict(model, image, brain, cfg)
            dice.append(metrics(prediction[0].numpy().astype(bool), lesion[0].numpy() > 0)["dice"])
    model.train()
    return float(np.mean(dice)) if dice else float("nan")


class TrainingDiverged(RuntimeError):
    """Entraînement arrêté : perte ou poids non finis, ou trop de pas sautés d'affilée par le GradScaler."""


def finite_parameters(model):
    """True si tous les paramètres du modèle sont finis (ni NaN ni Inf)."""
    return all(bool(torch.isfinite(p).all()) for p in model.parameters())


def improves(dice, best):
    """Un Dice NaN ou infini n'est jamais une amélioration (le meilleur initial vaut -inf)."""
    return math.isfinite(dice) and dice > best


class DivergenceGuard:
    """Arrêt immédiat si la perte ou les poids ne sont pas finis, ou après `max_skipped` pas sautés d'affilée.

    Un pas sauté (GradScaler qui baisse son échelle) ne modifie pas les poids ; une série de pas sautés
    signifie que l'entraînement est bloqué (échelle qui s'effondre vers 0).
    """

    def __init__(self, max_skipped=50, max_loss=100.0):
        self.max_skipped = max_skipped
        self.max_loss = max_loss  # perte Dice + BCE de l'ordre de 1 : au-delà de 100, l'entraînement diverge
        self.consecutive_skipped = 0
        self.skipped = 0

    def check(self, iteration, loss, model, skipped):
        if not math.isfinite(loss):
            raise TrainingDiverged(f"Itération {iteration} : perte non finie ({loss}) ; arrêt de l'entraînement")
        if self.max_loss is not None and loss > self.max_loss:
            raise TrainingDiverged(f"Itération {iteration} : perte totale {loss:.4g} > {self.max_loss:g} "
                                   "(divergence) ; arrêt de l'entraînement")
        if not finite_parameters(model):
            raise TrainingDiverged(f"Itération {iteration} : poids non finis après la mise à jour ; arrêt de l'entraînement")
        self.skipped += int(skipped)
        self.consecutive_skipped = self.consecutive_skipped + 1 if skipped else 0
        if self.consecutive_skipped > self.max_skipped:
            raise TrainingDiverged(f"Itération {iteration} : {self.consecutive_skipped} pas sautés d'affilée par le "
                                   f"GradScaler (gradients non finis) ; arrêt de l'entraînement")


def save_checkpoint(path, model, optimizer, scaler, state, cfg, resume=None):
    """Écriture atomique (fichier temporaire puis renommage) : une coupure pendant l'écriture laisse
    l'ancien checkpoint intact. `resume` : états aléatoires des derniers lots, workers, LR (last.pt)."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                    state=state, config=asdict(cfg), reprise=resume), temporary)
    os.replace(temporary, path)


def train(cfg, augment=None, wrap_model=None, extra_loss=None, test_transform=None, resume=False):
    """Entraîne, garde le meilleur modèle sur la validation du pli, puis évalue sur les découpages de test.

    Les points d'accroche par défaut viennent de `technique_hooks(cfg)` ; on peut les remplacer.
    """
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_splits(cfg.splits)
    centers = training_sites(splits, cfg.fold)
    hooks = technique_hooks(cfg, centers)
    augment = augment if augment is not None else hooks[0]
    wrap_model = wrap_model if wrap_model is not None else hooks[1]
    extra_loss = extra_loss if extra_loss is not None else hooks[2]
    test_transform = test_transform if test_transform is not None else hooks[3]
    output = Path(cfg.output) / cfg.name
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")

    train_records, validation_records = training_records(cfg)
    network = build_network(cfg)
    model = (wrap_model(network) if wrap_model else network).to(device)
    from monai.losses import DiceCELoss
    seg_loss = DiceCELoss(sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.amp and device.type == "cuda")
    state = dict(iteration=0, elapsed=0.0, best_dice=-math.inf, best_iteration=None)
    guard = DivergenceGuard()
    workers, rng_states = cfg.num_workers, []
    if resume and (output / "last.pt").is_file():
        checkpoint = torch.load(output / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        state = checkpoint["state"]
        saved = checkpoint.get("reprise") or {}
        rng_states = list(saved.get("rng_lots", []))
        if rng_states and saved.get("num_workers") != workers:
            print(f"Reprise avec {workers} workers au lieu de {saved.get('num_workers')} : tirages aléatoires "
                  "différents de la run sans coupure")
            rng_states = []
        print(f"Reprise à l'itération {state['iteration']} ({state['elapsed'] / 3600:.2f} h) ; "
              + ("indices et états aléatoires restaurés" if rng_states else "indices restaurés, états aléatoires neufs"))
    # Données : construites après la reprise (rang de départ des indices) ; aucun tirage aléatoire ici.
    dataset = TrackedDataset(PatchDataset(train_records, augment, cfg, centers))
    sampler = ResumableSampler(dataset, 10**7, cfg.seed, start=state["iteration"] * cfg.volumes_per_batch)
    worker_init = functools.partial(resume_worker, states=tuple(rng_states)) if workers and rng_states else worker_seed
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.volumes_per_batch, sampler=sampler,
                                         num_workers=workers, collate_fn=collate_tracked,
                                         worker_init_fn=worker_init, persistent_workers=workers > 0,
                                         pin_memory=device.type == "cuda")
    recent = collections.deque(rng_states, maxlen=max(workers, 1))  # états après les derniers lots appris
    log = CsvWriter(output / "journal_entrainement.csv", ["iteration", "heures", "lr", "lambda", "perte",
                                                         "perte_segmentation", "perte_centre", "dice_validation"],
                    append=state["iteration"] > 0)
    budget = cfg.time_budget_hours * 3600
    started, last_checkpoint = time.time() - state["elapsed"], time.time()
    print(f"{cfg.name} : {len(train_records)} sujets d'entraînement, {len(validation_records)} de validation, "
          f"{len(centers)} centres, appareil {device}, budget {cfg.time_budget_hours} h")
    model.train()
    running = []
    window_start, window_iteration = time.time(), state["iteration"]
    lr = None

    def resume_info():
        return dict(rng_lots=list(recent), num_workers=workers, lr=lr,
                    volumes_consommes=state["iteration"] * cfg.volumes_per_batch)

    batches = iter(loader)
    if not workers and rng_states:  # chargement dans ce processus : son état après le dernier lot appris
        set_rng_state(rng_states[-1])
    for batch in batches:
        batch_rng = batch.pop("rng")
        # Avancement de 0 à 1 : budget de temps, ou nombre d'itérations s'il est atteint avant.
        progress = (time.time() - started) / budget
        if cfg.max_iterations:
            progress = max(progress, state["iteration"] / cfg.max_iterations)
        if progress >= 1.0:
            break
        recent.append(batch_rng)
        lr = cfg.learning_rate * (1 - progress) ** 0.9  # décroissance polynomiale sur le budget
        for group in optimizer.param_groups:
            group["lr"] = lr
        lambda_ = model.set_progress(progress) if isinstance(model, DANNWrapper) else None
        image, label = batch["image"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=cfg.amp and device.type == "cuda"):
            segmentation, aux = split_outputs(model(image))
            loss = seg_loss(segmentation.float(), label)
            parts = {"segmentation": float(loss.detach())}
            if extra_loss is not None and aux is not None:
                additional, extra_parts = extra_loss(aux, batch)
                loss = loss + additional
                parts.update(extra_parts)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scale = scaler.get_scale()
        if cfg.grad_clip:
            scaler.unscale_(optimizer)  # sans effet en fp32 (scaler inactif) ; indispensable avec --amp
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        state["iteration"] += 1
        guard.check(state["iteration"], float(loss.detach()), model, skipped=scaler.get_scale() < scale)
        running.append((float(loss.detach()), parts))
        if state["iteration"] % cfg.log_every == 0:
            now = time.time()
            seconds = (now - window_start) / max(state["iteration"] - window_iteration, 1)
            remaining = (cfg.max_iterations - state["iteration"]) * seconds / 3600 if cfg.max_iterations else None
            recent = running[-cfg.log_every:]
            line = dict(progression=state["iteration"], heures=round((now - started) / 3600, 3),
                        s_par_iteration=round(seconds, 3), perte=round(float(np.mean([l for l, _ in recent])), 4),
                        lr=f"{lr:.2e}", heures_restantes=None if remaining is None else round(remaining, 2))
            if lambda_ is not None:  # DANN : termes de la perte et lambda à chaque ligne
                line.update(perte_segmentation=round(float(np.mean([p["segmentation"] for _, p in recent])), 4),
                            perte_centre=round(float(np.mean([p.get("centre", np.nan) for _, p in recent])), 4),
                            **{"lambda": round(lambda_, 4)})
            print(line, flush=True)
            window_start, window_iteration = now, state["iteration"]
        if state["iteration"] % cfg.validate_every == 0:
            state["elapsed"] = time.time() - started
            dice = validate(model, validation_records, cfg, test_transform) if len(validation_records) else float("nan")
            if improves(dice, state["best_dice"]) and finite_parameters(model):
                state.update(best_dice=dice, best_iteration=state["iteration"])
                save_checkpoint(output / "best.pt", model, optimizer, scaler, state, cfg)
            mean = lambda key: np.mean([p.get(key, np.nan) for _, p in running]) if running else np.nan  # noqa: E731
            row = dict(iteration=state["iteration"], heures=round(state["elapsed"] / 3600, 3), lr=f"{lr:.2e}",
                       **{"lambda": None if lambda_ is None else round(lambda_, 3)},
                       perte=round(float(np.mean([l for l, _ in running])), 4),
                       perte_segmentation=round(float(mean("segmentation")), 4),
                       perte_centre=None if lambda_ is None else round(float(mean("centre")), 4),
                       dice_validation=None if math.isnan(dice) else round(dice, 4))
            log(row)
            print(row, flush=True)
            running = []
        if time.time() - last_checkpoint > cfg.checkpoint_minutes * 60:
            state["elapsed"] = time.time() - started
            save_checkpoint(output / "last.pt", model, optimizer, scaler, state, cfg, resume_info())
            last_checkpoint = time.time()
    state["elapsed"] = time.time() - started
    save_checkpoint(output / "last.pt", model, optimizer, scaler, state, cfg, resume_info())
    if not len(validation_records) and not (output / "best.pt").is_file():
        save_checkpoint(output / "best.pt", model, optimizer, scaler, state, cfg)  # entraînement final : le dernier
    print(f"Entraînement terminé : {state['iteration']} itérations, {state['elapsed'] / 3600:.2f} h, "
          f"meilleur Dice de validation {state['best_dice']:.4f} (itération {state['best_iteration']})")
    if (output / "best.pt").is_file():
        best = torch.load(output / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(best["model"])
    else:
        print("Aucune validation avec un Dice fini : pas de best.pt, évaluation du dernier modèle")
    return model, run_evaluation(model, cfg, output, test_transform, state=state)


def run_evaluation(model, cfg, output, test_transform=None, tent=False, state=None):
    model.eval()
    rows = []
    suffix = "_tent" if tent else ""
    for split in cfg.evaluation_splits:
        writer = CsvWriter(output / f"resultats_{split}{suffix}.csv", RESULT_COLUMNS)
        rows += evaluate(model, cfg, split, test_transform, tent=tent, writer=writer)
    summary = dict(run=cfg.name, technique=cfg.technique + ("+tent" if tent else ""), fold=cfg.fold,
                   entrainement=state, **summarize(rows))
    (output / f"resume{suffix}.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return summary


def load_model(checkpoint_path, device=None):
    """Recharge un modèle et sa configuration depuis un checkpoint de `train`."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = RunConfig(**{**checkpoint["config"], "channels": tuple(checkpoint["config"]["channels"]),
                       "strides": tuple(checkpoint["config"]["strides"]),
                       "evaluation_splits": tuple(checkpoint["config"]["evaluation_splits"])})
    network = build_network(cfg)
    centers = training_sites(load_splits(cfg.splits), cfg.fold)
    model = DANNWrapper(network, n_centers=len(centers)) if TECHNIQUES[cfg.technique].get("dann") else network
    model.load_state_dict(checkpoint["model"])
    return model.to(device), cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--technique", choices=list(TECHNIQUES), default="baseline")
    parser.add_argument("--fold", default="0", help="numéro de pli, ou 'final' pour tous les sujets d'entraînement")
    parser.add_argument("--data-root", default=str(DEFAULT_PROCESSED))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS))
    parser.add_argument("--output", default=str(REPO_ROOT / "adaptation" / "runs"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--time-budget-hours", type=float, default=10.0, help="garde-fou (session Kaggle de 12 h)")
    parser.add_argument("--max-iterations", type=int, default=40000, help="identique pour toutes les techniques")
    parser.add_argument("--validate-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=100, help="ligne de progression toutes les N itérations")
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patches-per-volume", type=int, default=2)
    parser.add_argument("--volumes-per-batch", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="fp16 (autocast + GradScaler) ; défaut fp32, voir RunConfig.amp")
    parser.add_argument("--resume", action="store_true", help="reprend depuis last.pt (session Kaggle interrompue)")
    parser.add_argument("--smoke", action="store_true", help="essai rapide : petit réseau, 4 sujets, 20 itérations")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--checkpoint", help="avec --evaluate-only : checkpoint à évaluer")
    parser.add_argument("--tent", action="store_true", help="avec --evaluate-only : adaptation TENT par centre")
    parser.add_argument("--evaluation-limit", type=int)
    args = parser.parse_args()

    if args.evaluate_only:
        model, cfg = load_model(args.checkpoint)
        cfg.data_root, cfg.splits = args.data_root, args.splits
        if args.smoke:
            cfg.evaluation_limit, cfg.tent_patches = 2, 2
        if args.evaluation_limit:
            cfg.evaluation_limit = args.evaluation_limit
        _, _, _, test_transform = technique_hooks(cfg, training_sites(load_splits(cfg.splits), cfg.fold))
        run_evaluation(model, cfg, Path(args.checkpoint).parent, test_transform, tent=args.tent)
        return
    cfg = RunConfig(technique=args.technique, fold=None if args.fold == "final" else int(args.fold),
                    data_root=args.data_root, splits=args.splits, output=args.output, seed=args.seed,
                    time_budget_hours=args.time_budget_hours, max_iterations=args.max_iterations,
                    validate_every=args.validate_every, log_every=args.log_every, validation_limit=args.validation_limit,
                    num_workers=args.num_workers, patches_per_volume=args.patches_per_volume,
                    volumes_per_batch=args.volumes_per_batch, amp=args.amp,
                    evaluation_limit=args.evaluation_limit)
    if args.smoke:
        cfg = smoke_config(cfg)
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
