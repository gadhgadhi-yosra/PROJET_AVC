"""Vérification visuelle des transformations : l'anatomie et le masque doivent rester intacts.

Pour quelques sujets de centres différents, écrit une figure par famille de transformations
(coupe axiale, lésion en contour, original puis plusieurs tirages) et mesure le rapport d'intensité
lésion / tissu voisin avant et après chaque tirage (`contraste_lesion.csv`).

    .venv-dl\\Scripts\\python -m generalization.visual_check --synthetic
    .venv-dl\\Scripts\\python -m generalization.visual_check --sites R015 R029 R059 R014
    .venv-dl\\Scripts\\python -m generalization.visual_check --source processed   # données du contrat C1

Sur données réelles, seuls les sujets d'entraînement de splits.json sont utilisés : un centre de test
OOD ou exclu est refusé, parce que ces mesures servent à régler les transformations.
"""
import argparse
import tempfile
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .calibration import DEFAULT_SPLITS, check_training_sites, load_splits, restrict_to_training
from .contrast import subject_ratio
from .data_processed import DEFAULT_PROCESSED, list_processed, load_processed_subject
from .data_raw import DEFAULT_ROOT, list_subjects, load_subject
from .synthetic import write_processed_dataset, write_synthetic_dataset
from .transforms import FDA, RandomBrainGamma, build_transforms, load_config
from .transforms.base import unit_range

HERE = Path(__file__).resolve().parent
CONFIGS = HERE / "configs"
# Famille -> (configs, True si le preset est montré en entier plutôt que transformation par transformation).
FAMILIES = {
    "intensite": (["intensity.yaml"], False),
    "style": (["style_randconv.yaml", "style_gin.yaml"], False),
    "fourier": (["fda.yaml"], False),
    "harmonisation": (["histogram_matching.yaml"], False),
    "bigaug": (["bigaug.yaml"], True),
}
DETERMINISTIC = {"histogram_matching"}
SYNTHETIC_FOV_MM = (120.0, 144.0, 112.0)
SYNTHETIC_REFERENCE = ["R009"]  # seul centre de référence possible dans le jeu synthétique
BLACK_LEVEL = 0.05  # part de l'échelle [min, p99] du cerveau sous laquelle un voxel est « au noir »

# Même palette que exploration/02_figures.py.
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 8.5,
    "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK2, "axes.titlesize": 8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8,
})


def build_rows(family, data_root, reference_sites, synthetic=False, source="raw", synthetic_splits=None):
    """Liste (nom, transformation) : une ligne de figure par transformation, forcée à p = 1."""
    configs, whole_preset = FAMILIES[family]
    entries = [dict(entry) for name in configs for entry in load_config(CONFIGS / name).get("transforms") or []]
    for entry in entries:
        if synthetic and "reference_sites" in entry:
            # Les volumes synthétiques ne sont pas dans le splits.json du dépôt.
            entry.pop("splits", None)
            if synthetic_splits:
                entry["splits"] = str(synthetic_splits)
            entry["reference_sites"] = list(reference_sites or SYNTHETIC_REFERENCE)
        elif reference_sites and "reference_sites" in entry:
            entry["reference_sites"] = list(reference_sites)
    build = lambda config: build_transforms(config, data_root, source=source)  # noqa: E731
    if whole_preset:
        return [(f"{family} (preset complet)", build({"transforms": entries}))]
    rows = []
    for entry in entries:
        if entry.get("enabled", True):
            rows.append((entry["name"], build({"transforms": [dict(entry, p=1.0)]})))
    return rows


def axial_view(subject, window, box):
    """Coupe axiale où la lésion est la plus grande : image (fond en noir) et masque, recadrés sur le cerveau."""
    data, mask, brain = (subject[key].data[0].numpy() for key in ("t1", "mask", "brain"))
    area = mask.sum(axis=(0, 1))
    z = int(area.argmax()) if area.max() > 0 else data.shape[2] // 2
    image = np.where(brain[:, :, z] > 0, data[:, :, z], window[0])
    return image[box].T, mask[:, :, z][box].T  # RAS : lignes = antérieur, colonnes = droite


def brain_box(subject, margin=4):
    brain = subject["brain"].data[0].numpy() > 0
    slices = []
    for axis in (0, 1):
        index = np.nonzero(brain.any(axis=tuple(a for a in range(3) if a != axis)))[0]
        slices.append(slice(max(index[0] - margin, 0), index[-1] + margin + 1))
    return tuple(slices)


def show(ax, image, mask, window, aspect, title):
    ax.imshow(image, cmap="gray", origin="lower", vmin=window[0], vmax=window[1], aspect=aspect, interpolation="nearest")
    if mask.any():
        ax.contour(mask, levels=[0.5], colors=[ORANGE], linewidths=0.9)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def histogram_panel(ax, subject, transformed, reference_quantiles):
    """Histogrammes dans le cerveau : avant, après, et référence visée."""
    before = subject["t1"].data[subject["brain"].data.bool()].numpy()
    after = transformed["t1"].data[transformed["brain"].data.bool()].numpy()
    reference = reference_quantiles.numpy()
    bins = np.linspace(*np.percentile(np.concatenate([before, reference]), [0.5, 99.5]), 80)
    centers = (bins[:-1] + bins[1:]) / 2
    density = lambda values: np.histogram(values, bins=bins, density=True)[0]  # noqa: E731
    ax.plot(centers, density(reference), color=INK2, lw=1.2, ls=(0, (4, 2)), label="référence")
    ax.plot(centers, density(before), color=BLUE, lw=2, label="avant")
    ax.plot(centers, density(after), color=ORANGE, lw=2, label="après")
    ax.set_xlabel("intensité dans le cerveau (z-score)")
    ax.set_yticks([])
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="best")


def check_subject(family, rows, subject, out_dir, draws, stat_draws, seed):
    """Écrit la figure de la famille pour ce sujet et retourne une ligne de mesures par tirage."""
    ratio_before = subject_ratio(subject)
    window = tuple(np.percentile(subject["t1"].data[subject["brain"].data.bool()].numpy(), [0.5, 99.5]))
    box = brain_box(subject)
    spacing = subject["t1"].spacing
    aspect = spacing[1] / spacing[0]
    original = axial_view(subject, window, box)
    harmonisation = family == "harmonisation"
    n_cols = 3 if harmonisation else 1 + draws
    height = original[0].shape[0] * spacing[1] / (original[0].shape[1] * spacing[0])
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(2.3 * n_cols, (2.3 * height + 0.3) * len(rows) + 0.8),
                             squeeze=False, layout="constrained")
    records = []
    for row_axes, (name, transform) in zip(axes, rows):
        show(row_axes[0], *original, window, aspect, f"original · L/V {ratio_before:.2f}")
        row_axes[0].set_ylabel(name, fontsize=9.5, color=INK, fontweight="bold")
        n_draws = 1 if name in DETERMINISTIC else max(draws, stat_draws)
        for draw in range(n_draws):
            torch.manual_seed(seed + draw)
            start = time.perf_counter()
            out = transform(subject)
            seconds = time.perf_counter() - start
            ratio_after = subject_ratio(out)
            records.append(dict(site=subject["site"], subject=subject["subject"], family=family, transform=name,
                                draw=draw, ratio_before=ratio_before, ratio_after=ratio_after, seconds=seconds,
                                mask_changed=not torch.equal(out["mask"].data, subject["mask"].data)))
            if harmonisation:
                show(row_axes[1], *axial_view(out, window, box), window, aspect, f"après · L/V {ratio_after:.2f}")
                histogram_panel(row_axes[2], subject, out, transform[0].reference_quantiles)
            elif draw < draws:
                show(row_axes[1 + draw], *axial_view(out, window, box), window, aspect,
                     f"tirage {draw + 1} · L/V {ratio_after:.2f}")
    shape = "×".join(str(n) for n in subject["t1"].spatial_shape)
    fig.suptitle(f"{family} — {subject['site']} · {subject['subject']} ({shape} voxels)\n"
                 "lésion en orange · L/V = intensité lésion / tissu voisin", x=0.01, ha="left", fontsize=10)
    path = out_dir / f"{family}__{subject['site']}_{subject['subject']}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  {path}")
    return records


def summarize(table):
    """Par transformation : rapport L/V après, part du contraste conservée, temps médian.

    Contraste conservé = (1 - L/V après) / (1 - L/V avant) : 1 = inchangé, 0 = lésion invisible,
    négatif = lésion devenue plus claire que son voisinage.
    """
    table = table.assign(kept=(1 - table["ratio_after"]) / (1 - table["ratio_before"]))
    grouped = table.groupby(["family", "transform"], sort=False)
    span = lambda values: f"{values.min():.2f}-{values.max():.2f}"  # noqa: E731
    return pd.DataFrame({
        "tirages": grouped.size(),
        "L/V avant": grouped["ratio_before"].agg(span),
        "L/V après": grouped["ratio_after"].agg(span),
        "conservé min": grouped["kept"].min().round(2),
        "conservé médian": grouped["kept"].median().round(2),
        "< 50 %": grouped["kept"].agg(lambda v: int((v < 0.5).sum())),
        "inversés": grouped["kept"].agg(lambda v: int((v < 0).sum())),
        "masque modifié": grouped["mask_changed"].sum(),
        "s / volume": grouped["seconds"].median().round(2),
    })


def kept_by_site(table):
    """Contraste conservé minimal, par transformation (lignes) et par centre (colonnes)."""
    table = table.assign(kept=(1 - table["ratio_after"]) / (1 - table["ratio_before"]))
    return table.pivot_table(index=["family", "transform"], columns="site", values="kept", aggfunc="min",
                             sort=False).round(2)


def check_fda_betas(subject, bank, betas, n_references, out_dir):
    """Figure de calibrage de FDA : pour quelques images de référence, l'effet de chaque beta fixé.

    Sous chaque image, la carte de différence à l'original (en écarts-types du cerveau) montre où
    FDA agit. Retourne une ligne de mesures par (référence, beta).
    """
    data, brain = subject["t1"].data[0], subject["brain"].data[0].bool()
    ratio_before = subject_ratio(subject)
    window = tuple(np.percentile(data[brain].numpy(), [0.5, 99.5]))
    box = brain_box(subject)
    spacing = subject["t1"].spacing
    aspect = spacing[1] / spacing[0]
    original = axial_view(subject, window, box)
    area = subject["mask"].data[0].sum(dim=(0, 1))
    z = int(area.argmax()) if area.max() > 0 else data.shape[2] // 2
    indices = sorted({round(i * (len(bank) - 1) / max(n_references - 1, 1)) for i in range(n_references)})
    height = original[0].shape[0] * spacing[1] / (original[0].shape[1] * spacing[0])
    n_cols = 1 + len(betas)
    fig, axes = plt.subplots(2 * len(indices), n_cols, figsize=(2.3 * n_cols, (2.3 * height + 0.45) * 2 * len(indices) + 0.8),
                             squeeze=False, layout="constrained")
    difference_map = matplotlib.colors.LinearSegmentedColormap.from_list("difference", [BLUE, "#f0efec", ORANGE])
    difference_map.set_bad("#000000")
    records = []
    for row, index in enumerate(indices):
        image_axes, difference_axes = axes[2 * row], axes[2 * row + 1]
        reference = bank[index].float()
        source = bank.sources[index]
        name = f"{source.record['site']} · {source.record['subject']}" if hasattr(source, "record") else f"n° {index}"
        show(image_axes[0], *original, window, aspect, f"original · L/V {ratio_before:.2f}")
        image_axes[0].set_ylabel("image", fontsize=9.5, color=INK, fontweight="bold")
        resized = torch.nn.functional.interpolate(reference[None, None], size=data.shape, mode="trilinear")[0, 0]
        target = resized[:, :, z].numpy()[box].T
        show(difference_axes[0], np.where(target == 0, window[0], target), np.zeros((1, 1)), window, aspect,
             f"référence : {name}")
        difference_axes[0].set_ylabel("différence", fontsize=9.5, color=INK, fontweight="bold")
        for column, beta in enumerate(betas, start=1):
            torch.manual_seed(0)
            out = FDA(references=[reference], beta=(beta, beta), p=1.0)(subject)
            ratio_after = subject_ratio(out)
            delta = (out["t1"].data[0] - data)
            correlation = float(torch.corrcoef(torch.stack([out["t1"].data[0][brain], data[brain]]))[0, 1])
            record = dict(site=subject["site"], subject=subject["subject"], reference=name, beta=beta,
                          half_width=int(beta * min(data.shape)), ratio_before=ratio_before, ratio_after=ratio_after,
                          kept=(1 - ratio_after) / (1 - ratio_before), rms_change=float(delta[brain].pow(2).mean().sqrt()),
                          correlation=correlation)
            records.append(record)
            show(image_axes[column], *axial_view(out, window, box), window, aspect,
                 f"β = {beta:g} (b = {record['half_width']}) · L/V {ratio_after:.2f}")
            masked = np.ma.masked_where(~brain[:, :, z].numpy()[box].T, delta[:, :, z].numpy()[box].T)
            difference_axes[column].imshow(masked, cmap=difference_map, origin="lower", vmin=-1, vmax=1, aspect=aspect,
                                           interpolation="nearest")
            difference_axes[column].set_title(f"écart {record['rms_change']:.2f} σ · r = {correlation:.3f}")
            difference_axes[column].set_xticks([])
            difference_axes[column].set_yticks([])
            for spine in difference_axes[column].spines.values():
                spine.set_visible(False)
    fig.suptitle(f"calibrage de FDA — {subject['site']} · {subject['subject']}\nligne « différence » : après − avant, "
                 "de −1 σ (bleu) à +1 σ (orange) du cerveau ; b = demi-côté du cube de fréquences échangé",
                 x=0.01, ha="left", fontsize=10)
    path = out_dir / f"fda_beta__{subject['site']}_{subject['subject']}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  {path}")
    return records


def check_gammas(subject, gammas, out_dir):
    """Balayage du gamma : part du cerveau mise au noir et contraste conservé, pour des valeurs fixes.

    « Au noir » = sous 5 % de l'échelle [minimum, p99] du cerveau d'origine, celle du gamma.
    """
    data, brain = subject["t1"].data, subject["brain"].data.bool()
    lo, hi = unit_range(data, brain)
    ratio_before = subject_ratio(subject)
    window = tuple(np.percentile(data[brain].numpy(), [0.5, 99.5]))
    box = brain_box(subject)
    spacing = subject["t1"].spacing
    aspect = spacing[1] / spacing[0]
    original = axial_view(subject, window, box)
    height = original[0].shape[0] * spacing[1] / (original[0].shape[1] * spacing[0])
    fig, axes = plt.subplots(1, 1 + len(gammas), figsize=(2.3 * (1 + len(gammas)), 2.3 * height + 1.0),
                             squeeze=False, layout="constrained")
    show(axes[0, 0], *original, window, aspect, f"original · L/V {ratio_before:.2f}")
    records = []
    for column, gamma in enumerate(gammas, start=1):
        out = RandomBrainGamma(gamma=(gamma, gamma), p=1.0)(subject)
        unit = (out["t1"].data[brain] - lo) / (hi - lo)
        ratio_after = subject_ratio(out)
        record = dict(site=subject["site"], subject=subject["subject"], gamma=gamma,
                      black=float((unit < BLACK_LEVEL).float().mean()), ratio_before=ratio_before,
                      ratio_after=ratio_after, kept=(1 - ratio_after) / (1 - ratio_before))
        records.append(record)
        show(axes[0, column], *axial_view(out, window, box), window, aspect,
             f"γ = {gamma:g} · noir {record['black']:.0%} · L/V {ratio_after:.2f}")
    fig.suptitle(f"balayage du gamma — {subject['site']} · {subject['subject']}", x=0.01, ha="left", fontsize=10)
    path = out_dir / f"gamma__{subject['site']}_{subject['subject']}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  {path}")
    return records


def select_records(data_root, sites, subjects, splits=None, source="raw", splits_path=DEFAULT_SPLITS):
    """Un sujet par centre demandé : celui nommé dans `subjects`, sinon le premier avec une lésion.

    Avec `splits`, seuls les sujets d'entraînement sont candidats. Avec `source="processed"`, les
    sujets viennent de `list_processed` : entraînement seulement, sauf si `splits` vaut None
    (inspection des centres de test).
    """
    if source == "processed":
        parts = ["train"] if splits is not None else ["train", "id_test", "ood_test"]
        table = pd.concat([list_processed(data_root, part, splits=splits_path, on_missing="warn", sites=sites,
                                          warn_empty=False) for part in parts], ignore_index=True)
        table = table[~table["empty_mask"]]
    else:
        table = list_subjects(data_root)
        if splits is not None:
            table = restrict_to_training(table, splits)
        table = table[~table["known_empty_mask"]]
    chosen = []
    for site in sites:
        candidates = table[table["site"] == site]
        named = candidates[candidates["subject"].isin(subjects or [])]
        if candidates.empty:
            warnings.warn(f"Centre {site} absent de {data_root} : ignoré")
            continue
        chosen.append((named if len(named) else candidates).iloc[0])
    if not chosen:
        raise SystemExit(f"Aucun des centres {sites} n'est présent dans {data_root}")
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("raw", "processed"), default="raw",
                        help="données brutes (défaut) ou prétraitées du contrat C1")
    parser.add_argument("--data-root", type=Path,
                        help="racine des données (défaut : data/ATLAS3_Training_Raw ou data/processed selon --source)")
    parser.add_argument("--synthetic", action="store_true", help="volumes synthétiques au lieu des données réelles")
    parser.add_argument("--sites", nargs="+", default=["R015", "R029", "R059", "R014"], help="un sujet par centre")
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS,
                        help="splits.json : seuls ses sujets d'entraînement sont utilisés (données réelles)")
    parser.add_argument("--include-held-out", action="store_true",
                        help="autorise les centres de test : à réserver à l'inspection, jamais à un réglage")
    parser.add_argument("--subjects", nargs="*", help="identifiants à préférer (sinon premier sujet du centre)")
    parser.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    parser.add_argument("--reference-sites", nargs="+", help="remplace `reference_sites` des configs (FDA, histogrammes)")
    parser.add_argument("--draws", type=int, default=4, help="tirages affichés par transformation")
    parser.add_argument("--stat-draws", type=int, default=10, help="tirages utilisés pour mesurer le contraste")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="dossier de sortie (défaut : generalization/figures/<synthetic|raw>)")
    parser.add_argument("--fda-betas", nargs="+", type=float,
                        help="calibrage de FDA : une figure par sujet avec ces valeurs fixes de beta, à la place des familles")
    parser.add_argument("--fda-references", type=int, default=2, help="images de référence montrées par sujet")
    parser.add_argument("--gammas", nargs="+", type=float,
                        help="balayage du gamma : une figure par sujet avec ces valeurs fixes, à la place des familles")
    args = parser.parse_args()

    splits = None
    if not args.synthetic and not args.include_held_out:
        splits = load_splits(args.splits)
        check_training_sites(args.sites, splits, "la vérification visuelle")
        if args.reference_sites:
            check_training_sites(args.reference_sites, splits, "les images de référence")
    elif args.include_held_out:
        print("Attention : centres de test autorisés. Ces mesures ne doivent servir à aucun réglage.")

    processed = args.source == "processed"
    out_dir = args.out or HERE / "figures" / ("synthetic" if args.synthetic else args.source)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        data_root = args.data_root or (DEFAULT_PROCESSED if processed else DEFAULT_ROOT)
        splits_path, synthetic_splits = args.splits, None
        if args.synthetic:
            print("Génération des volumes synthétiques…")
            if processed:
                data_root, synthetic_splits = write_processed_dataset(Path(temporary) / "processed")
                splits_path, splits = synthetic_splits, load_splits(synthetic_splits)
            else:
                data_root = write_synthetic_dataset(Path(temporary) / "ATLAS3_Training_Raw", fov_mm=SYNTHETIC_FOV_MM)
        elif not data_root.is_dir():
            raise SystemExit(f"Données introuvables dans {data_root} : passer --data-root, prendre l'autre --source, "
                             "ou utiliser --synthetic")
        load = load_processed_subject if processed else load_subject
        rows_for = lambda family: build_rows(family, data_root, args.reference_sites, args.synthetic,  # noqa: E731
                                             args.source, synthetic_splits)
        subjects = [load(record) for record in
                    select_records(data_root, args.sites, args.subjects, splits, args.source, splits_path)]
        if args.gammas:
            table = pd.DataFrame([record for subject in subjects for record in check_gammas(subject, args.gammas, out_dir)])
            table.to_csv(out_dir / "gamma.csv", index=False)
            with pd.option_context("display.width", 200, "display.max_columns", None):
                print(table.pivot_table(index="gamma", columns="site", values="black").round(3))
                print(table.pivot_table(index="gamma", columns="site", values="kept").round(2))
            return
        if args.fda_betas:
            bank = rows_for("fourier")[0][1][0].references
            records = [record for subject in subjects
                       for record in check_fda_betas(subject, bank, args.fda_betas, args.fda_references, out_dir)]
            table = pd.DataFrame(records)
            table.to_csv(out_dir / "fda_beta.csv", index=False)
            with pd.option_context("display.width", 200, "display.max_columns", None):
                print(table.groupby("beta")[["half_width", "kept", "rms_change", "correlation"]]
                      .agg(["min", "median", "max"]).round(3))
            return
        records = []
        for family in args.families:
            print(f"Famille « {family} »")
            rows = rows_for(family)
            for subject in subjects:
                records += check_subject(family, rows, subject, out_dir, args.draws, args.stat_draws, args.seed)
    table = pd.DataFrame(records)
    table.to_csv(out_dir / "contraste_lesion.csv", index=False)
    summary = summarize(table)
    summary.to_csv(out_dir / "resume.csv")
    by_site = kept_by_site(table)
    by_site.to_csv(out_dir / "conserve_min_par_centre.csv")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(f"\nRésumé sur {table['subject'].nunique()} sujets (détail : {out_dir / 'contraste_lesion.csv'})\n{summary}")
        print(f"\nContraste conservé minimal par centre\n{by_site}")


if __name__ == "__main__":
    main()
