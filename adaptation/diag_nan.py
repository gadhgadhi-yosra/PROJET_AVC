"""Diagnostic des NaN de la baseline (pli 0) : quatre variantes instrumentées, 1 000 itérations au plus chacune.

    python -u -m adaptation.diag_nan --data-root /kaggle/working/processed --output /kaggle/working/diag_nan
    .venv-dl\\Scripts\\python -m adaptation.diag_nan --iterations 20 --patch 64 --output adaptation\\runs\\diag_local

Même réseau, mêmes données, même pli, même graine et même ordre de tirage que `adaptation.train` (baseline,
sans augmentation) ; seule la variante change :
  A  baseline telle quelle : fp16 (autocast + GradScaler), DiceCELoss(sigmoid, smooth 1e-5, Dice par patch),
     sans clipping ni warmup ;
  B  A en fp32 (ni autocast ni scaler) ;
  C  A avec DiceCELoss(batch=True) : Dice sur tout le lot, comme nnU-Net ;
  D  A avec scaler.unscale_ + clip_grad_norm_(12) + warmup linéaire du LR sur 500 itérations.

Journal `<sortie>/<variante>/journal.csv`, une ligne par itération, écrite tout de suite : pertes (totale,
Dice, BCE), logits, gradients après unscale_ (norme globale, max), pas appliqué ou sauté, échelle du scaler,
LR, max|poids|, max|exp_avg|, max|exp_avg_sq|, sujets, centres, voxels de lésion par patch, et des sondes
(hooks forward) sur les premières couches : dtype d'entrée et de sortie, max|sortie|.

Rupture : au premier poids ou moment d'AdamW non fini après une mise à jour, la variante s'arrête et écrit
dans `<variante>/rupture/` le lot (`lot.pt`), les poids, l'état de l'optimiseur et du scaler AVANT ce pas
(`avant_pas.pt`), les gradients de ce pas par tenseur (`gradients.csv`) et l'état après le pas par tenseur
(`apres_pas.csv`). Résumé de toutes les variantes : `<sortie>/resume.json` (et à l'écran).
"""
import argparse
import copy
import csv
import json
import math
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from generalization.calibration import load_splits, training_sites

from .train import (PatchDataset, RunConfig, build_network, collate_patches, seed_everything, technique_hooks,
                    training_records, worker_seed)

VARIANTS = {
    "A": dict(description="baseline : fp16, Dice par patch, sans clipping ni warmup",
              amp=True, batch_dice=False, clip=None, warmup=0),
    "B": dict(description="A en fp32 (sans autocast ni scaler)", amp=False, batch_dice=False, clip=None, warmup=0),
    "C": dict(description="A avec DiceCELoss(batch=True)", amp=True, batch_dice=True, clip=None, warmup=0),
    "D": dict(description="A avec unscale_ + clip_grad_norm_(12) + warmup linéaire sur 500 itérations",
              amp=True, batch_dice=False, clip=12.0, warmup=500),
}
# Sondes : premières couches du UNet MONAI (premier bloc résiduel, puis premier bloc du niveau 2).
PROBES = {"conv1": "model.0.conv.unit0.conv", "norm1": "model.0.conv.unit0.adn.N", "act1": "model.0.conv.unit0.adn.A",
          "norm2": "model.0.conv.unit1.adn.N", "norm3": "model.1.submodule.0.conv.unit0.adn.N"}
KINDS = ("perte", "logits", "gradient", "poids", "exp_avg", "exp_avg_sq")
# Environnement de la run baseline rq3-baseline-pli0 : un écart est signalé dans le résumé, sans arrêter.
BASELINE_ENVIRONMENT = dict(torch="2.11.0+cu130", gpu="Tesla T4")


def environment_warnings(environment):
    warnings = []
    if environment["torch"] != BASELINE_ENVIRONMENT["torch"]:
        warnings.append(f"torch {environment['torch']} au lieu de {BASELINE_ENVIRONMENT['torch']} (baseline)")
    if BASELINE_ENVIRONMENT["gpu"] not in environment["gpu"]:
        warnings.append(f"GPU {environment['gpu']} au lieu de {BASELINE_ENVIRONMENT['gpu']} (baseline)")
    return warnings


class DiagDataset(PatchDataset):
    """PatchDataset qui renvoie aussi l'identifiant du sujet (mêmes tirages aléatoires que l'original)."""

    def __getitem__(self, index):
        patches, site = super().__getitem__(index)
        return patches, (site, self.records.iloc[index]["case_id"])


def collate_with_ids(items):
    batch = collate_patches([(patches, site) for patches, (site, _) in items])
    batch["case_id"] = [case for patches, (_, case) in items for _ in patches]
    return batch


def attach_probes(model):
    """Hooks forward : (dtype d'entrée, dtype de sortie, max|sortie| sur GPU) de chaque sonde, au dernier passage."""
    modules = dict(model.named_modules())
    values = {}

    def hook(label):
        def record(module, inputs, output):
            values[label] = (str(inputs[0].dtype).replace("torch.", ""), str(output.dtype).replace("torch.", ""),
                             output.detach().abs().amax().float())
        return record

    for label, path in PROBES.items():
        modules[path].register_forward_hook(hook(label))
    return values


def tensor_stats(tensors):
    """Par tenseur, sur GPU sans synchronisation : max|x| (NaN propagé), fini, nombre de NaN et d'Inf, norme."""
    return dict(abs_max=torch.stack([t.detach().abs().amax().float() for t in tensors]),
                finite=torch.stack([torch.isfinite(t).all() for t in tensors]),
                nan=torch.stack([torch.isnan(t).sum() for t in tensors]),
                inf=torch.stack([torch.isinf(t).sum() for t in tensors]),
                norm=torch.stack([torch.linalg.vector_norm(t.detach().float()) for t in tensors]))


def moments(optimizer, params, key):
    return [optimizer.state[p][key] for p in params if key in optimizer.state[p]]


def adam_steps(optimizer, params):
    steps = [float(optimizer.state[p]["step"]) for p in params if "step" in optimizer.state[p]]
    return int(max(steps)) if steps else 0


def value(tensor):
    return float(tensor.item())


def per_tensor_rows(names, shapes, prefix_stats):
    rows = []
    for index, (name, shape) in enumerate(zip(names, shapes)):
        row = dict(tenseur=name, forme="x".join(map(str, shape)))
        for prefix, stats in prefix_stats.items():
            if stats is None:
                continue
            for key, column in stats.items():
                row[f"{prefix}_{key}"] = column[index].item() if column.dim() else column.item()
        rows.append(row)
    return rows


def write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def run_variant(name, spec, cfg, output, iterations, time_limit):
    """Une variante : renvoie son résumé (statut, itération de rupture, pas sautés, premiers non finis…)."""
    folder = output / name
    folder.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    # Même ordre que adaptation.train.train : graines, points d'accroche, données, échantillonneur, réseau.
    seed_everything(cfg.seed)
    splits = load_splits(cfg.splits)
    centers = training_sites(splits, cfg.fold)
    augment = technique_hooks(cfg, centers)[0]
    train_records, _ = training_records(cfg)
    dataset = DiagDataset(train_records, augment, cfg, centers)
    sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=10**7,
                                             generator=torch.Generator().manual_seed(cfg.seed))
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.volumes_per_batch, sampler=sampler,
                                         num_workers=cfg.num_workers, collate_fn=collate_with_ids,
                                         worker_init_fn=worker_seed, persistent_workers=False, pin_memory=True)
    model = build_network(cfg).to(device)
    from monai.losses import DiceCELoss
    loss_fn = DiceCELoss(sigmoid=True, batch=spec["batch_dice"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=spec["amp"])
    names, params = zip(*model.named_parameters())
    shapes = [tuple(p.shape) for p in params]
    probes = attach_probes(model)
    columns = (["iteration", "lr", "echelle", "pas_applique", "pas_adam", "perte", "perte_dice", "perte_bce",
                "logits_finis", "logits_non_finis", "logits_max_abs", "grad_finis", "grad_norme", "grad_max_abs",
                "grad_norme_apres_clip", "poids_finis", "poids_max_abs", "exp_avg_finis", "exp_avg_max_abs",
                "exp_avg_sq_finis", "exp_avg_sq_max_abs", "sujets", "centres", "voxels_lesion", "secondes"]
               + [f"{label}_{key}" for label in PROBES for key in ("dtype_entree", "dtype_sortie", "max_abs")])
    handle = open(folder / "journal.csv", "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, columns)
    writer.writeheader()
    summary = dict(variante=name, description=spec["description"], statut=None, iterations=0, pas_appliques=0,
                   pas_sautes=0, rupture=None, premiers_non_finis={}, echelle_finale=None)
    budget = cfg.time_budget_hours * 3600
    started = time.time()
    model.train()
    iteration = 0
    try:
        for batch in loader:
            if iteration >= iterations:
                summary["statut"] = f"OK à {iterations}"
                break
            if time.time() - started > time_limit:
                summary["statut"] = f"temps écoulé à l'itération {iteration} (sans rupture)"
                break
            tick = time.time()
            # LR de la baseline (décroissance polynomiale sur l'avancement), warmup linéaire en D.
            progress = max((time.time() - started) / budget, iteration / cfg.max_iterations)
            lr = cfg.learning_rate * (1 - progress) ** 0.9
            if spec["warmup"]:
                lr *= min(1.0, (iteration + 1) / spec["warmup"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            # État AVANT ce pas (sur GPU, quelques centaines de Mo) : sauvegardé seulement en cas de rupture.
            before = dict(model={k: v.detach().clone() for k, v in model.state_dict().items()},
                          optimizer=copy.deepcopy(optimizer.state_dict()), scaler=copy.deepcopy(scaler.state_dict()))
            image, label = batch["image"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=spec["amp"]):
                logits = model(image)
                segmentation = logits.float()
                dice = loss_fn.dice(segmentation, label)
                bce = loss_fn.bce(segmentation, label)
                loss = loss_fn.lambda_dice * dice + loss_fn.lambda_ce * bce  # = DiceCELoss.forward
            optimizer.zero_grad(set_to_none=True)
            scale = scaler.get_scale() if spec["amp"] else 1.0
            steps_before = adam_steps(optimizer, params)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # sans effet en fp32 ; sinon gradients réels, et step ne divise plus
            grads = [p.grad for p in params]
            grad_stats = tensor_stats(grads)
            grad_norm = torch.linalg.vector_norm(grad_stats["norm"])
            clipped = None  # gradients.csv et grad_* : avant le clipping ; grad_norme_apres_clip : après (D)
            if spec["clip"]:
                torch.nn.utils.clip_grad_norm_(params, spec["clip"])
                clipped = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.float()) for g in grads]))
            scaler.step(optimizer)
            scaler.update()
            iteration += 1
            applied = adam_steps(optimizer, params) > steps_before
            weight_stats = tensor_stats(params)
            avg, avg_sq = moments(optimizer, params, "exp_avg"), moments(optimizer, params, "exp_avg_sq")
            avg_stats, sq_stats = (tensor_stats(avg) if avg else None), (tensor_stats(avg_sq) if avg_sq else None)
            flags = dict(perte=bool(torch.isfinite(loss)), logits=bool(torch.isfinite(logits).all()),
                         gradient=bool(grad_stats["finite"].all()), poids=bool(weight_stats["finite"].all()),
                         exp_avg=avg_stats is None or bool(avg_stats["finite"].all()),
                         exp_avg_sq=sq_stats is None or bool(sq_stats["finite"].all()))
            flags.update({f"sonde_{label_name}": math.isfinite(value(peak))
                          for label_name, (_, _, peak) in probes.items()})
            for kind, finite in flags.items():
                if not finite:
                    summary["premiers_non_finis"].setdefault(kind, iteration)
            lesion = label.sum(dim=(1, 2, 3, 4)).long().tolist()
            row = dict(iteration=iteration, lr=f"{lr:.3e}", echelle=scale, pas_applique=int(applied),
                       pas_adam=adam_steps(optimizer, params), perte=value(loss), perte_dice=value(dice),
                       perte_bce=value(bce), logits_finis=int(flags["logits"]),
                       logits_non_finis=int((~torch.isfinite(logits)).sum()),
                       logits_max_abs=value(logits.detach().abs().amax().float()),
                       grad_finis=int(flags["gradient"]), grad_norme=value(grad_norm),
                       grad_max_abs=value(grad_stats["abs_max"].max()),
                       grad_norme_apres_clip=None if clipped is None else value(clipped),
                       poids_finis=int(flags["poids"]), poids_max_abs=value(weight_stats["abs_max"].max()),
                       exp_avg_finis=int(flags["exp_avg"]),
                       exp_avg_max_abs=None if avg_stats is None else value(avg_stats["abs_max"].max()),
                       exp_avg_sq_finis=int(flags["exp_avg_sq"]),
                       exp_avg_sq_max_abs=None if sq_stats is None else value(sq_stats["abs_max"].max()),
                       sujets=";".join(batch["case_id"]), centres=";".join(batch["site"]),
                       voxels_lesion=";".join(map(str, lesion)), secondes=round(time.time() - tick, 3))
            for label_name, (dtype_in, dtype_out, peak) in probes.items():
                row.update({f"{label_name}_dtype_entree": dtype_in, f"{label_name}_dtype_sortie": dtype_out,
                            f"{label_name}_max_abs": value(peak)})
            writer.writerow(row)
            handle.flush()
            summary["iterations"] = iteration
            summary["pas_appliques" if applied else "pas_sautes"] += 1
            if iteration % 50 == 0 or iteration == 1:
                print(f"[{name}] it {iteration} perte {row['perte']:.4f} (dice {row['perte_dice']:.4f}, "
                      f"bce {row['perte_bce']:.4f}) |logits| {row['logits_max_abs']:.3g} grad {row['grad_norme']:.3g} "
                      f"échelle {scale:g} sautés {summary['pas_sautes']} {row['secondes']} s", flush=True)
            if not (flags["poids"] and flags["exp_avg"] and flags["exp_avg_sq"]):
                summary["statut"] = f"rupture à l'itération {iteration}"
                summary["rupture"] = save_rupture(folder / "rupture", iteration, lr, batch, before, names, shapes,
                                                  grad_stats, weight_stats, avg_stats, sq_stats, flags, applied, spec,
                                                  before_moments=(moments_before(before["optimizer"], len(params))))
                print(f"[{name}] RUPTURE à l'itération {iteration} : {json.dumps(summary['rupture'], ensure_ascii=False)}",
                      flush=True)
                break
        else:
            summary["statut"] = f"chargeur épuisé à l'itération {iteration}"
    except torch.cuda.OutOfMemoryError as error:
        summary["statut"] = f"mémoire GPU insuffisante à l'itération {iteration + 1} : {str(error).splitlines()[0]}"
    finally:
        handle.close()
        summary["echelle_finale"] = scaler.get_scale() if spec["amp"] else None
        summary["secondes"] = round(time.time() - started, 1)
        del loader, model, optimizer
        torch.cuda.empty_cache()
    return summary


def moments_before(optimizer_state, count):
    """Statistiques des moments AVANT le pas (dict d'état d'AdamW : indices 0..count-1)."""
    state = optimizer_state["state"]
    if not state:
        return None, None
    stats = {}
    for key in ("exp_avg", "exp_avg_sq"):
        tensors = [state[i][key] for i in range(count)]
        stats[key] = dict(abs_max=torch.stack([t.abs().amax().float() for t in tensors]),
                          min=torch.stack([t.amin().float() for t in tensors]))
    return stats["exp_avg"], stats["exp_avg_sq"]


def save_rupture(folder, iteration, lr, batch, before, names, shapes, grad_stats, weight_stats, avg_stats, sq_stats,
                 flags, applied, spec, before_moments):
    """Écrit le lot, l'état avant le pas, les gradients du pas et l'état après le pas ; renvoie un résumé."""
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({key: batch[key] for key in ("image", "label", "brain", "site", "case_id")}, folder / "lot.pt")
    torch.save(dict(before, iteration=iteration, lr=lr, variante=spec), folder / "avant_pas.pt")
    write_rows(folder / "gradients.csv", per_tensor_rows(names, shapes, {"grad": grad_stats}))
    avg_before, sq_before = before_moments
    write_rows(folder / "apres_pas.csv", per_tensor_rows(names, shapes, {
        "poids": weight_stats, "exp_avg": avg_stats, "exp_avg_sq": sq_stats,
        "exp_avg_avant": avg_before, "exp_avg_sq_avant": sq_before}))

    def broken(stats):
        if stats is None:
            return []
        return [dict(tenseur=names[i], nan=int(stats["nan"][i]), inf=int(stats["inf"][i]))
                for i in range(len(names)) if not bool(stats["finite"][i])]

    return dict(iteration=iteration, lr=lr, pas_applique=bool(applied), gradients_finis=flags["gradient"],
                grad_max_abs=float(grad_stats["abs_max"].max()), perte_finie=flags["perte"],
                logits_finis=flags["logits"],
                **{f"{kind}_non_finis": broken(stats)[:10] + ([f"… ({len(broken(stats))} tenseurs au total)"]
                                                              if len(broken(stats)) > 10 else [])
                   for kind, stats in (("poids", weight_stats), ("exp_avg", avg_stats), ("exp_avg_sq", sq_stats))})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=RunConfig.data_root)
    parser.add_argument("--splits", default=RunConfig.splits)
    parser.add_argument("--output", default=str(Path(RunConfig.output) / "diag_nan"))
    parser.add_argument("--variants", default="ABCD", help="sous-ensemble de ABCD, dans l'ordre voulu")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--minutes-per-variant", type=float, default=60.0, help="garde-fou de durée par variante")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patch", type=int, default=96, help="essai local seulement (96 = baseline)")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Pas de GPU CUDA : ce diagnostic porte sur le fp16 CUDA (l'autocast CPU utilise bf16)")
    cfg = RunConfig(technique="baseline", fold=0, data_root=args.data_root, splits=args.splits, seed=2026,
                    num_workers=args.num_workers, amp=True)
    if args.patch != cfg.patch:
        print(f"Attention : patch {args.patch}³ au lieu de {cfg.patch}³ (essai local, pas la config baseline)")
        cfg = replace(cfg, patch=args.patch)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    import monai
    environment = dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                       gpu=torch.cuda.get_device_name(0), monai=monai.__version__, python=platform.python_version())
    version_file = Path(__file__).resolve().parents[1] / "VERSION.json"
    code_version = json.loads(version_file.read_text(encoding="utf-8")) if version_file.is_file() else None
    warnings = environment_warnings(environment)
    print(environment, flush=True)
    print("Code :", code_version, flush=True)
    for warning in warnings:
        print("ATTENTION :", warning, flush=True)
    summaries = []
    for name in args.variants:
        print(f"=== Variante {name} : {VARIANTS[name]['description']}", flush=True)
        summaries.append(run_variant(name, VARIANTS[name], cfg, output, args.iterations, args.minutes_per_variant * 60))
        report = dict(environnement=environment, environnement_baseline=BASELINE_ENVIRONMENT,
                      avertissements=warnings, code=code_version, config=asdict(cfg), iterations_max=args.iterations,
                      sondes=PROBES, variantes=summaries)
        (output / "resume.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                                            encoding="utf-8")
    print("\n=== Résumé")
    print("Environnement :", "identique à la baseline (torch, GPU)" if not warnings else "; ".join(warnings))
    for summary in summaries:
        first = ", ".join(f"{kind} à {it}" for kind, it in sorted(summary["premiers_non_finis"].items(),
                                                                  key=lambda item: item[1])) or "aucun"
        print(f"{summary['variante']} ({summary['description']}) : {summary['statut']} ; "
              f"{summary['pas_appliques']} pas appliqués, {summary['pas_sautes']} sautés ; "
              f"échelle finale {summary['echelle_finale']} ; premiers non finis : {first}")
    print(json.dumps(summaries, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
