"""Adaptation continue (version réduite) : centres OOD présentés un par un au modèle baseline.

    python -m adaptation.continual --checkpoint runs/unet3d_baseline_pli0/best.pt --data-root DATA --output OUT

Évaluation préquentielle : les sujets de chaque centre OOD arrivent par lots ; chaque lot est d'abord
noté (Dice, avec les étiquettes, uniquement pour la mesure) puis sert à l'adaptation, sans étiquettes.
Le modèle n'est jamais réinitialisé entre centres. Quatre stratégies, sur les mêmes lots et les mêmes patches :

- `fige` : modèle baseline inchangé (référence) ;
- `auto_apprentissage` : pseudo-étiquettes du modèle (seuil 0,5), Dice + BCE sur tous les poids ;
- `tent` : TENT continu (entropie, paramètres affines des normalisations seulement) ;
- `controlee` : TENT dans un challenger, promu champion seulement si (1) un décalage est détecté sur le lot
  (distance de Wasserstein entre histogrammes d'intensité du cerveau, `center_distance()` d'analysis/,
  au-dessus du seuil calibré sur la source), (2) l'entropie du challenger sur des patches neufs du lot
  baisse et (3) son Dice sur un jeu de garde étiqueté de la source ne baisse pas de plus de `guard_tolerance`.
  L'adaptation mélange le lot courant, un replay buffer des anciens centres OOD (entropie) et des patches
  étiquetés des centres source (Dice + BCE), pour limiter l'oubli.

Oubli : après chaque centre, Dice de chaque stratégie sur un sous-ensemble fixe du test ID (centres
source) et sur les premiers sujets de chaque centre OOD déjà vu.

Sorties dans `--output` : `prequentiel.csv` (un sujet par ligne et par stratégie), `oubli.csv`,
`decisions.csv` (un lot par ligne : distance, seuil, détection, promotion) et `resume.json`.
"""
import argparse
import copy
import importlib.util
import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generalization.calibration import load_splits  # noqa: E402
from generalization.data_processed import list_processed, load_processed_subject  # noqa: E402

from .tent import collect_params, configure_tent, voxel_entropy  # noqa: E402
from .train import (CsvWriter, load_model, predict, sample_patch, segmentation_network,  # noqa: E402
                    subject_arrays, technique_hooks)

STRATEGIES = ("fige", "auto_apprentissage", "tent", "controlee")


def _load_center_distance():
    """`center_distance` et `brain_histogram` d'analysis/center_distance.py (contrat C6)."""
    path = REPO_ROOT / "analysis" / "center_distance.py"
    spec = importlib.util.spec_from_file_location("center_distance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.center_distance, module.brain_histogram


center_distance, brain_histogram = _load_center_distance()


@dataclass
class ContinualConfig:
    subjects_per_center: int | None = 24   # version réduite (None : tous les sujets OOD du centre)
    lot_size: int = 4
    patches_per_subject: int = 4           # patches d'adaptation par sujet du lot
    learning_rate: float = 1e-4            # TENT et auto-apprentissage
    reference_per_site: int = 3            # sujets d'entraînement par centre source pour l'histogramme de référence
    detection_quantile: float = 0.95       # seuil = quantile des distances de lots source tirés au hasard
    guard_subjects: int = 8                # jeu de garde étiqueté (validation du pli, centres source)
    guard_tolerance: float = 0.01
    source_replay_subjects: int = 8        # sujets d'entraînement étiquetés dont les patches servent au replay
    replay_per_center: int = 8             # patches gardés par ancien centre OOD
    replay_weight: float = 1.0
    forgetting_id_subjects: int = 16       # sous-ensemble fixe du test ID pour mesurer l'oubli
    forgetting_per_center: int = 4         # premiers sujets de chaque centre OOD déjà vu
    seed: int = 2026


def dice(prediction, lesion):
    g, p = lesion[0] > 0, prediction[0] > 0
    total = int(g.sum()) + int(p.sum())
    return float("nan") if not g.any() else 2 * int((g & p).sum()) / total


class Volume:
    """Sujet chargé une fois (image float32, lésion uint8, cerveau bool) ; le Dice n'est défini que si lésion."""

    def __init__(self, record, test_transform=None):
        subject = load_processed_subject(record)
        if test_transform is not None:
            subject = test_transform(subject)
        image, lesion, brain = subject_arrays(subject)
        self.case_id, self.site = record["case_id"], record["site"]
        self.image, self.lesion, self.brain = image, (lesion > 0).to(torch.uint8), brain
        self.gt_ml = float(self.lesion.sum()) / 1000

    def histogram(self):
        return brain_histogram(self.image[0].numpy(), self.brain[0].numpy())

    def patches(self, n, size, generator, lesion_fraction=0.0):
        lesion = self.lesion.float() if lesion_fraction else torch.zeros_like(self.lesion, dtype=torch.float32)
        return [sample_patch(self.image, lesion, self.brain, size, lesion_fraction, generator) for _ in range(n)]


def stack(patches, device):
    images, lesions, brains = zip(*patches)
    return (torch.stack(images).to(device), torch.stack(lesions).to(device), torch.stack(brains).to(device))


# --- Stratégies ---

class Strategy:
    name = "fige"

    def __init__(self, model, cfg, ccfg):
        self.model, self.cfg, self.ccfg = model, cfg, ccfg
        self.log = {}

    @property
    def network(self):
        return self.model

    def score(self, volume):
        prediction, _ = predict(self.network, volume.image, volume.brain, self.cfg)
        return dice(prediction, volume.lesion)

    def adapt(self, patches, context):
        self.log = {}


class SelfTraining(Strategy):
    name = "auto_apprentissage"

    def __init__(self, model, cfg, ccfg):
        super().__init__(copy.deepcopy(segmentation_network(model)), cfg, ccfg)
        from monai.losses import DiceCELoss
        self.loss = DiceCELoss(sigmoid=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=ccfg.learning_rate)

    def adapt(self, patches, context):
        images, _, _ = patches
        self.model.train()
        with torch.no_grad():
            pseudo = (torch.sigmoid(self.model(images)) > 0.5).float()
        loss = self.loss(self.model(images), pseudo)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.log = dict(perte=float(loss.detach()), pseudo_lesion=float(pseudo.mean()))


class ContinualTent(Strategy):
    name = "tent"

    def __init__(self, model, cfg, ccfg):
        network = configure_tent(copy.deepcopy(segmentation_network(model)))
        super().__init__(network, cfg, ccfg)
        self.optimizer = torch.optim.Adam(collect_params(network), lr=ccfg.learning_rate)

    def adapt(self, patches, context):
        images, _, brains = patches
        loss = voxel_entropy(self.model(images), brains)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.log = dict(entropie=float(loss.detach()))


class Controlled(Strategy):
    """Champion déployé ; challenger adapté par TENT + replay, promu sous conditions (voir le module)."""
    name = "controlee"

    def __init__(self, model, cfg, ccfg, guard, source_replay):
        champion = configure_tent(copy.deepcopy(segmentation_network(model)))
        super().__init__(champion, cfg, ccfg)
        from monai.losses import DiceCELoss
        self.supervised = DiceCELoss(sigmoid=True)
        self.guard, self.source_replay = guard, source_replay
        self.replay = {}  # centre OOD -> patches (sans étiquettes)
        self.challenger = None
        self.champion_guard = self.guard_dice(self.model)

    def guard_dice(self, network):
        values = [dice(predict(network, v.image, v.brain, self.cfg)[0], v.lesion) for v in self.guard]
        values = [v for v in values if np.isfinite(v)]
        return float(np.mean(values)) if values else float("nan")

    def start_lot(self, detected):
        """Nouveau challenger (copie du champion) si un décalage est détecté sur le lot."""
        self.challenger = None
        if detected:
            self.challenger = copy.deepcopy(self.model)
            self.optimizer = torch.optim.Adam(collect_params(self.challenger), lr=self.ccfg.learning_rate)

    def adapt(self, patches, context):
        if self.challenger is None:
            return
        images, _, brains = patches
        loss = voxel_entropy(self.challenger(images), brains)
        old = [p for site, kept in self.replay.items() if site != context["site"] for p in kept]
        generator = context["generator"]
        if old:
            chosen = [old[i] for i in torch.randperm(len(old), generator=generator)[:len(images)].tolist()]
            old_images, _, old_brains = stack(chosen, images.device)
            loss = loss + self.ccfg.replay_weight * voxel_entropy(self.challenger(old_images), old_brains)
        if self.source_replay:
            chosen = [self.source_replay[i] for i in
                      torch.randperm(len(self.source_replay), generator=generator)[:len(images)].tolist()]
            source_images, source_lesions, _ = stack(chosen, images.device)
            loss = loss + self.ccfg.replay_weight * self.supervised(self.challenger(source_images), source_lesions)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.log.setdefault("perte", []).append(float(loss.detach()))

    @torch.no_grad()
    def end_lot(self, fresh, site, lot_patches):
        """Promotion éventuelle du challenger ; mise à jour du replay buffer du centre courant."""
        kept = self.replay.setdefault(site, [])
        for patch in lot_patches:
            if len(kept) < self.ccfg.replay_per_center:
                kept.append(tuple(t.cpu() for t in patch))
        if self.challenger is None:
            return dict(promu=False, raison="pas de décalage détecté")
        images, _, brains = fresh
        entropy_champion = float(voxel_entropy(self.model(images), brains))
        entropy_challenger = float(voxel_entropy(self.challenger(images), brains))
        guard = self.guard_dice(self.challenger)
        guard_ok = not np.isfinite(self.champion_guard) or guard >= self.champion_guard - self.ccfg.guard_tolerance
        promoted = entropy_challenger < entropy_champion and guard_ok
        decision = dict(promu=promoted, entropie_champion=entropy_champion, entropie_challenger=entropy_challenger,
                        garde_champion=self.champion_guard, garde_challenger=guard,
                        raison="promu" if promoted else ("garde en baisse" if not guard_ok else "entropie en hausse"))
        if promoted:
            self.model, self.champion_guard = self.challenger, guard
        self.challenger = None
        return decision


# --- Détection du décalage ---

def detection_threshold(reference, lot_size, quantile, seed):
    """Quantile des distances de lots source (tirés au hasard) au reste de la référence."""
    if len(reference) < 3:
        warnings.warn("Référence de moins de 3 sujets : seuil nul, décalage détecté sur chaque lot")
        return 0.0
    rng = np.random.default_rng(seed)
    distances = []
    for _ in range(200):
        chosen = rng.choice(len(reference), size=min(lot_size, len(reference) - 2), replace=False)
        rest = np.delete(reference, chosen, axis=0)
        distances.append(center_distance(reference[chosen], rest))
    return float(np.quantile(distances, quantile))


# --- Boucle ---

def sample_records(records, n, seed):
    if n is None or len(records) <= n:
        return records
    return records.sample(n=n, random_state=seed).sort_index()


def run(checkpoint, data_root, splits, output, ccfg=None, strategies=STRATEGIES, centers=None, device=None):
    ccfg = ccfg or ContinualConfig()
    started = time.time()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(checkpoint, device)
    model.eval()
    cfg.data_root, cfg.splits = str(data_root), str(splits)
    split_spec = load_splits(splits)
    _, _, _, test_transform = technique_hooks(cfg, [])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(dict(checkpoint=str(checkpoint), technique=cfg.technique,
                                                        fold=cfg.fold, **asdict(ccfg)), indent=2), encoding="utf-8")
    load = lambda record: Volume(record, test_transform)  # noqa: E731
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_records = list_processed(data_root, "train", cfg.fold, splits=split_spec, on_missing="warn")
        val_records = list_processed(data_root, "val", cfg.fold, splits=split_spec, on_missing="warn")
        id_records = list_processed(data_root, "id_test", cfg.fold, splits=split_spec, on_missing="warn")
        ood_records = list_processed(data_root, "ood_test", None, splits=split_spec, on_missing="warn")
    available = set(ood_records["site"])
    if centers:
        order = [c for c in centers if c in available]
    else:
        order = [c for c in split_spec.get("ood_centers", []) if c in available]
        order += sorted(available - set(order))
    print(f"Centres OOD dans l'ordre : {order} ; stratégies : {list(strategies)} ; appareil {device}", flush=True)

    # Référence d'intensité (centres source) et seuil de détection.
    reference_records = train_records.groupby("site").head(ccfg.reference_per_site)
    reference = np.stack([load(r).histogram() for _, r in reference_records.iterrows()])
    threshold = detection_threshold(reference, ccfg.lot_size, ccfg.detection_quantile, ccfg.seed)
    print(f"Référence : {len(reference)} sujets source ; seuil de détection {threshold:.4f}", flush=True)

    generator = torch.Generator().manual_seed(ccfg.seed)
    with_lesion = lambda frame: frame[~frame["empty_mask"]]  # noqa: E731
    guard = [load(r) for _, r in with_lesion(val_records).head(ccfg.guard_subjects).iterrows()]
    source_replay = [tuple(t for t in patch) for _, r in with_lesion(train_records).sample(
        n=min(ccfg.source_replay_subjects, len(with_lesion(train_records))), random_state=ccfg.seed).iterrows()
        for patch in load(r).patches(ccfg.patches_per_subject, cfg.patch, generator, lesion_fraction=0.5)]
    forgetting_id = [load(r) for _, r in sample_records(with_lesion(id_records), ccfg.forgetting_id_subjects,
                                                        ccfg.seed).iterrows()]
    builders = dict(fige=lambda: Strategy(model, cfg, ccfg),
                    auto_apprentissage=lambda: SelfTraining(model, cfg, ccfg),
                    tent=lambda: ContinualTent(model, cfg, ccfg),
                    controlee=lambda: Controlled(model, cfg, ccfg, guard, source_replay))
    runners = {name: builders[name]() for name in strategies}

    prequential = CsvWriter(output / "prequentiel.csv", ["strategie", "ordre", "centre", "lot", "case_id", "gt_ml",
                                                         "dice"])
    forgetting = CsvWriter(output / "oubli.csv", ["strategie", "apres_centre", "ensemble", "dice", "n"])
    decisions = CsvWriter(output / "decisions.csv", ["centre", "lot", "distance", "seuil", "decalage", "strategie",
                                                     "promu", "raison", "entropie_champion", "entropie_challenger",
                                                     "garde_champion", "garde_challenger", "perte"])
    seen = {}  # centre OOD -> volumes gardés pour mesurer l'oubli

    def evaluate_forgetting(after):
        sets = {"id_source": forgetting_id, **{f"ood_{site}": volumes for site, volumes in seen.items()}}
        for name, runner in runners.items():
            for label, volumes in sets.items():
                values = [v for v in (runner.score(volume) for volume in volumes) if np.isfinite(v)]
                forgetting(dict(strategie=name, apres_centre=after, ensemble=label,
                                dice=float(np.mean(values)) if values else None, n=len(values)))

    evaluate_forgetting("depart")
    lot_index = 0
    for position, site in enumerate(order, 1):
        records = sample_records(ood_records[ood_records["site"] == site], ccfg.subjects_per_center, ccfg.seed)
        center_start = time.time()
        for start in range(0, len(records), ccfg.lot_size):
            lot_index += 1
            volumes = [load(r) for _, r in records.iloc[start:start + ccfg.lot_size].iterrows()]
            # 1. Notation préquentielle avant toute adaptation sur ce lot.
            for name, runner in runners.items():
                for volume in volumes:
                    prequential(dict(strategie=name, ordre=position, centre=site, lot=lot_index,
                                     case_id=volume.case_id, gt_ml=volume.gt_ml, dice=runner.score(volume)))
            # 2. Détection du décalage, puis adaptation sans étiquettes sur des patches communs aux stratégies.
            distance = center_distance(np.stack([v.histogram() for v in volumes]), reference)
            detected = distance > threshold
            lot_patches = [p for v in volumes for p in v.patches(ccfg.patches_per_subject, cfg.patch, generator)]
            fresh = stack([p for v in volumes for p in v.patches(1, cfg.patch, generator)], device)
            context = dict(site=site, generator=generator)
            if "controlee" in runners:
                runners["controlee"].start_lot(detected)
            for step in range(ccfg.patches_per_subject):
                batch = stack(lot_patches[step::ccfg.patches_per_subject], device)
                for runner in runners.values():
                    runner.adapt(batch, context)
            for name, runner in runners.items():
                row = dict(centre=site, lot=lot_index, distance=round(distance, 5), seuil=round(threshold, 5),
                           decalage=detected, strategie=name)
                if isinstance(runner, Controlled):
                    losses = runner.log.pop("perte", [])
                    row.update(runner.end_lot(fresh, site, lot_patches), perte=np.mean(losses) if losses else None)
                elif runner.log:
                    row.update(perte=runner.log.get("perte", runner.log.get("entropie")))
                decisions(row)
            if len(seen.get(site, [])) < ccfg.forgetting_per_center:
                seen.setdefault(site, []).extend(
                    [v for v in volumes if v.gt_ml > 0][:ccfg.forgetting_per_center - len(seen.get(site, []))])
        print(f"[{position}/{len(order)}] {site} : {len(records)} sujets, {time.time() - center_start:.0f} s "
              f"(total {(time.time() - started) / 60:.1f} min)", flush=True)
        evaluate_forgetting(site)

    summary = summarize(output, order, threshold, time.time() - started)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def is_true(column):
    return column.astype(str).str.lower().eq("true")


def summarize(output, order, threshold, seconds):
    prequential = pd.read_csv(output / "prequentiel.csv")
    forgetting = pd.read_csv(output / "oubli.csv")
    decisions = pd.read_csv(output / "decisions.csv")
    strategies = {}
    for name, rows in prequential.groupby("strategie", sort=False):
        own = forgetting[forgetting["strategie"] == name]
        start = own[(own["apres_centre"] == "depart") & (own["ensemble"] == "id_source")]["dice"]
        end = own[(own["apres_centre"] == order[-1]) & (own["ensemble"] == "id_source")]["dice"] if order else start
        # Oubli des centres OOD : Dice en fin de flux - Dice juste après le passage sur le centre.
        backward = []
        for site in order[:-1]:
            label = f"ood_{site}"
            after = own[(own["apres_centre"] == site) & (own["ensemble"] == label)]["dice"]
            final = own[(own["apres_centre"] == order[-1]) & (own["ensemble"] == label)]["dice"]
            if len(after) and len(final) and np.isfinite(after.iloc[0]) and np.isfinite(final.iloc[0]):
                backward.append(final.iloc[0] - after.iloc[0])
        strategies[name] = dict(
            dice_prequentiel=float(rows["dice"].mean()),
            par_centre={site: float(group["dice"].mean()) for site, group in rows.groupby("centre", sort=False)},
            dice_id_depart=float(start.iloc[0]) if len(start) else None,
            dice_id_fin=float(end.iloc[0]) if len(end) else None,
            oubli_id=float(start.iloc[0] - end.iloc[0]) if len(start) and len(end) else None,
            oubli_ood_moyen=float(-np.mean(backward)) if backward else None)
    controlled = decisions[decisions["strategie"] == "controlee"]
    summary = dict(centres=order, seuil_detection=threshold, lots=int(decisions["lot"].nunique()),
                   lots_avec_decalage=int(is_true(decisions.drop_duplicates("lot")["decalage"]).sum()),
                   promotions=int(is_true(controlled["promu"]).sum()) if len(controlled) else None,
                   minutes=round(seconds / 60, 1), strategies=strategies)
    (output / "resume.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="best.pt de la baseline")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--splits", default=str(REPO_ROOT / "splits.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--centers", nargs="+", help="ordre des centres OOD (défaut : ood_centers de splits.json)")
    parser.add_argument("--subjects-per-center", type=int, default=24, help="0 : tous les sujets")
    parser.add_argument("--lot-size", type=int, default=4)
    parser.add_argument("--forgetting-id-subjects", type=int, default=16)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ccfg = ContinualConfig(subjects_per_center=args.subjects_per_center or None, lot_size=args.lot_size,
                           forgetting_id_subjects=args.forgetting_id_subjects)
    run(args.checkpoint, args.data_root, args.splits, args.output, ccfg, args.strategies, args.centers)


if __name__ == "__main__":
    main()
