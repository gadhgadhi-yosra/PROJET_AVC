"""Agrégation du tableau d'ablation RQ3 à partir des sorties des runs (Kaggle ou locales).

    .venv-dl\\Scripts\\python -m adaptation.aggregate_rq3 data\\kaggle_sorties --output adaptation\\runs\\rq3_agregation

Entrées : un ou plusieurs dossiers (ou archives .zip téléchargées depuis l'onglet Output de Kaggle),
parcourus récursivement. Chaque `resultats_<split>[_tent].csv` (une ligne par sujet, colonnes
RESULT_COLUMNS de train.py) est rattaché à sa technique par la colonne `technique` ; peu importe
le compte Kaggle ou le nom du dossier. Un même sujet trouvé deux fois pour une technique (sortie
téléchargée en double) n'est gardé qu'une fois, depuis le fichier le plus récent.

Sorties :
- `tableau_rq3.csv` / `tableau_rq3.md` : Dice ID, Dice OOD, écart ID - OOD (IC 95 % par bootstrap),
  différence appariée à la baseline et test de Wilcoxon (sujets communs), p corrigé par Holm ;
- `par_centre.csv`, `par_taille.csv` : Dice moyen par centre et par classe de volume de lésion ;
- `entrainement.csv` : itérations, durée, meilleur Dice de validation (resume.json) ;
- figures PNG dans `--figures` (défaut : figures/rq3/).
Le Dice n'est défini que pour les sujets à lésion non vide (gt_ml > 0) : les masques vides sont exclus.
"""
import argparse
import io
import json
import math
import re
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE = "baseline"
ORDER = ["baseline", "intensity", "bigaug", "fda", "histogram_matching", "dann", "baseline+tent"]
LABELS = {"baseline": "Baseline", "intensity": "Intensité", "bigaug": "BigAug", "fda": "FDA",
          "histogram_matching": "Histogram matching", "dann": "DANN", "baseline+tent": "Baseline + TENT"}
SHORT_LABELS = {"histogram_matching": "HM", "baseline+tent": "Baseline\n+ TENT"}
# Comparaisons directes A - B entre techniques candidates au modèle final, hors correction de Holm.
COMPARISONS = [("bigaug", "intensity"), ("histogram_matching", "intensity")]
SIZE_BINS = [0, 1, 10, 50, math.inf]  # ml
SIZE_LABELS = ["< 1 ml", "1-10 ml", "10-50 ml", ">= 50 ml"]
# Étiquettes des figures : unité « mL » du rapport (les CSV gardent SIZE_LABELS).
FIGURE_SIZE_LABELS = [label.replace(" ml", " mL") for label in SIZE_LABELS]
RESULT_FILE = re.compile(r"(^|/)resultats_(id_test|ood_test)(_tent)?\.csv$")
SUMMARY_FILE = re.compile(r"(^|/)resume(_tent)?\.json$")
# Palette catégorielle (ordre fixe) et teinte séquentielle des figures.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# --- Lecture des sorties ---

def display_path(path):
    """Chemin relatif au dépôt quand c'est possible : aucun chemin absolu de la machine dans les sorties."""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return path.name if path.is_absolute() else str(path)


def iter_files(inputs, pattern):
    """(nom, date, contenu en octets) des fichiers dont le chemin correspond, dans les dossiers et les .zip."""
    for root in map(Path, inputs):
        candidates = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in candidates:
            if path.suffix == ".zip":
                with zipfile.ZipFile(path) as archive:
                    for info in archive.infolist():
                        if pattern.search(info.filename):
                            yield f"{display_path(path)}!{info.filename}", path.stat().st_mtime, archive.read(info)
            elif path.is_file() and pattern.search(path.as_posix()):
                yield display_path(path), path.stat().st_mtime, path.read_bytes()


def load_results(inputs):
    """Toutes les lignes sujet, dédoublonnées par (technique, split, case_id) ; colonne `source` = fichier."""
    frames = []
    for name, mtime, content in iter_files(inputs, RESULT_FILE):
        frame = pd.read_csv(io.BytesIO(content))
        if frame.empty:
            continue
        frames.append(frame.assign(source=name, mtime=mtime))
    if not frames:
        raise SystemExit(f"Aucun fichier resultats_*.csv sous {', '.join(map(str, inputs))}")
    rows = pd.concat(frames, ignore_index=True)
    rows = rows.sort_values("mtime", kind="stable").drop_duplicates(["technique", "split", "case_id"], keep="last")
    # Ordre fixe des sujets : les IC bootstrap ne dépendent ni des dates des fichiers ni des autres runs présentes.
    rows = rows.sort_values(["technique", "split", "case_id"], kind="stable")
    duplicates = sum(len(f) for f in frames) - len(rows)
    if duplicates:
        warnings.warn(f"{duplicates} lignes en double ignorées (sorties téléchargées plusieurs fois)")
    for column in ("dice", "gt_ml", "pred_ml", "hd95_mm", "lesion_f1", "abs_vol_diff_ml"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    return rows.drop(columns="mtime").reset_index(drop=True)


def load_training(inputs):
    records = []
    for name, _, content in iter_files(inputs, SUMMARY_FILE):
        summary = json.loads(content)
        if not summary.get("technique"):
            continue  # autre résumé (adaptation continue…), pas une run du tableau
        state = summary.get("entrainement") or {}
        records.append(dict(technique=summary.get("technique"), run=summary.get("run"),
                            iterations=state.get("iteration"),
                            heures=None if state.get("elapsed") is None else round(state["elapsed"] / 3600, 2),
                            best_dice_validation=state.get("best_dice"), best_iteration=state.get("best_iteration"),
                            source=name))
    return pd.DataFrame(records, columns=["technique", "run", "iterations", "heures", "best_dice_validation",
                                          "best_iteration", "source"])


def techniques_in(rows):
    present = list(dict.fromkeys(rows["technique"]))
    return [t for t in ORDER if t in present] + sorted(t for t in present if t not in ORDER)


# --- Statistiques ---

def scored(rows):
    """Sujets où le Dice est défini (lésion non vide, Dice fini)."""
    return rows[(rows["gt_ml"] > 0) & np.isfinite(rows["dice"])]


def bootstrap_mean(values, n=10000, seed=2026):
    """Moyenne et IC 95 % percentile ; (nan, nan, nan) sans valeur."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_gap(id_values, ood_values, n=10000, seed=2026):
    """Écart ID - OOD des moyennes ; IC 95 % en rééchantillonnant les deux groupes indépendamment."""
    a, b = np.asarray(id_values, dtype=float), np.asarray(ood_values, dtype=float)
    if not len(a) or not len(b):
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    gaps = (a[rng.integers(0, len(a), (n, len(a)))].mean(axis=1)
            - b[rng.integers(0, len(b), (n, len(b)))].mean(axis=1))
    return float(a.mean() - b.mean()), float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))


def paired(rows, technique, split, reference=BASELINE):
    """Dice appariés (technique, référence) sur les sujets communs du découpage."""
    subset = scored(rows[rows["split"] == split])
    a = subset[subset["technique"] == technique].set_index("case_id")["dice"]
    b = subset[subset["technique"] == reference].set_index("case_id")["dice"]
    common = a.index.intersection(b.index)
    return a.loc[common].to_numpy(), b.loc[common].to_numpy()


def wilcoxon_p(a, b):
    """p bilatéral du test des rangs signés ; nan si moins de 2 paires ou différences toutes nulles."""
    from scipy.stats import wilcoxon
    differences = np.asarray(a) - np.asarray(b)
    if len(differences) < 2 or not np.any(differences != 0):
        return math.nan
    return float(wilcoxon(a, b).pvalue)


def holm(pvalues):
    """Correction de Holm-Bonferroni (les nan restent nan)."""
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p[index]))
        adjusted[index] = running
    return adjusted.tolist()


def ablation_table(rows, n_boot=10000):
    table = []
    for technique in techniques_in(rows):
        own = scored(rows[rows["technique"] == technique])
        id_dice = own[own["split"] == "id_test"]["dice"]
        ood_dice = own[own["split"] == "ood_test"]["dice"]
        record = dict(technique=technique, n_id=len(id_dice), n_ood=len(ood_dice))
        for key, values in (("dice_id", id_dice), ("dice_ood", ood_dice)):
            record[key], record[key + "_ic_bas"], record[key + "_ic_haut"] = bootstrap_mean(values, n_boot)
        record["ecart"], record["ecart_ic_bas"], record["ecart_ic_haut"] = bootstrap_gap(id_dice, ood_dice, n_boot)
        for split, short in (("id_test", "id"), ("ood_test", "ood")):
            if technique == BASELINE:
                record.update({f"n_paires_{short}": None, f"delta_{short}": None, f"delta_{short}_ic_bas": None,
                               f"delta_{short}_ic_haut": None, f"p_wilcoxon_{short}": None})
                continue
            a, b = paired(rows, technique, split)
            delta = bootstrap_mean(a - b, n_boot) if len(a) else (math.nan,) * 3
            record.update({f"n_paires_{short}": len(a), f"delta_{short}": delta[0], f"delta_{short}_ic_bas": delta[1],
                           f"delta_{short}_ic_haut": delta[2], f"p_wilcoxon_{short}": wilcoxon_p(a, b)})
        for metric in ("hd95_mm", "lesion_f1", "abs_vol_diff_ml"):
            record[f"{metric}_ood"] = float(own[own["split"] == "ood_test"][metric].mean())
        table.append(record)
    table = pd.DataFrame(table)
    for short in ("id", "ood"):
        column = f"p_wilcoxon_{short}"
        if column in table:
            table[f"p_holm_{short}"] = holm(pd.to_numeric(table[column], errors="coerce"))
    return table


def per_center(rows):
    data = scored(rows)
    return (data.groupby(["technique", "split", "site"])["dice"].agg(dice="mean", n="count").reset_index()
            .sort_values(["split", "site", "technique"]))


def per_size(rows):
    data = scored(rows).copy()
    data["taille"] = pd.cut(data["gt_ml"], SIZE_BINS, labels=SIZE_LABELS, right=False)
    return (data.groupby(["technique", "split", "taille"], observed=True)["dice"].agg(dice="mean", n="count")
            .reset_index())


# --- Sorties ---

def fmt_ci(mean, low, high, digits=3):
    if mean is None or not np.isfinite(mean):
        return "-"
    return f"{mean:.{digits}f} [{low:.{digits}f} ; {high:.{digits}f}]"


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return "-"
    return f"{p:.2g}" if p >= 1e-4 else "< 1e-4"


def markdown_table(table):
    lines = ["| Technique | n ID / OOD | Dice ID [IC 95 %] | Dice OOD [IC 95 %] | Écart ID - OOD | "
             "Δ OOD vs baseline | p Wilcoxon OOD (Holm) | p Wilcoxon ID (Holm) |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        lines.append(f"| {LABELS.get(r.technique, r.technique)} | {r.n_id} / {r.n_ood} | "
                     f"{fmt_ci(r.dice_id, r.dice_id_ic_bas, r.dice_id_ic_haut)} | "
                     f"{fmt_ci(r.dice_ood, r.dice_ood_ic_bas, r.dice_ood_ic_haut)} | "
                     f"{fmt_ci(r.ecart, r.ecart_ic_bas, r.ecart_ic_haut)} | "
                     f"{fmt_ci(r.delta_ood, r.delta_ood_ic_bas, r.delta_ood_ic_haut)} | "
                     f"{fmt_p(r.p_wilcoxon_ood)} ({fmt_p(r.p_holm_ood)}) | "
                     f"{fmt_p(r.p_wilcoxon_id)} ({fmt_p(r.p_holm_id)}) |")
    return "\n".join(lines) + "\n"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figures(rows, table, centers, sizes, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder.mkdir(parents=True, exist_ok=True)
    order = list(table["technique"])
    labels = [LABELS.get(t, t) for t in order]
    paths = []

    # 1. Dice ID et OOD par technique, IC 95 % (deux séries).
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(order)), 4))
    x = np.arange(len(order))
    for offset, (key, name, color) in zip((-0.2, 0.2), (("dice_id", "Test ID", COLORS[0]),
                                                         ("dice_ood", "Test OOD", COLORS[1]))):
        mean = table[key].to_numpy(dtype=float)
        error = np.vstack([mean - table[key + "_ic_bas"], table[key + "_ic_haut"] - mean])
        ax.bar(x + offset, mean, 0.38, color=color, label=name, edgecolor="white", linewidth=1)
        ax.errorbar(x + offset, mean, yerr=np.nan_to_num(error), fmt="none", ecolor=INK, elinewidth=1, capsize=3)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Dice moyen (IC 95 %)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    style(ax)
    ax.set_title("RQ3 - Dice ID et OOD par technique (pli 0)", color=INK, loc="left")
    paths.append(save(fig, folder / "rq3_dice_id_ood.png"))

    # 2. Dice par centre : un centre par ligne (test ID puis test OOD), une technique par colonne.
    # Largeur 6,3 pouces (16 cm) : insérée à \linewidth dans le rapport, les textes gardent leur taille de 8 pt.
    from matplotlib.colors import LinearSegmentedColormap
    data = centers.assign(groupe=(centers["split"] == "ood_test").astype(int))
    lines = (data.groupby(["groupe", "split", "site"])["n"].max().reset_index().sort_values(["groupe", "site"]))
    grid = (data.pivot_table(index=["split", "site"], columns="technique", values="dice")
            .reindex(list(zip(lines["split"], lines["site"]))).reindex(columns=order).to_numpy(dtype=float))
    n_lines, n_id = len(lines), int((lines["groupe"] == 0).sum())
    fig, ax = plt.subplots(figsize=(6.3, min(9.0, max(3.5, 0.165 * n_lines + 1.4))))
    ramp = LinearSegmentedColormap.from_list("rq3", ["#f3f7fd", COLORS[0], "#0b2e63"])  # bleu de la palette
    image = ax.imshow(grid, cmap=ramp, vmin=0, vmax=1, aspect="auto")
    # Étiquettes courtes (comme dans les tableaux du rapport) : sept colonnes sur 16 cm sans chevauchement.
    ax.set_xticks(range(len(order)), [SHORT_LABELS.get(t, LABELS.get(t, t)) for t in order], fontsize=8)
    ax.xaxis.tick_top()
    ax.set_yticks(range(n_lines), [f"{site} (n = {n})" for site, n in zip(lines["site"], lines["n"])], fontsize=8)
    ax.tick_params(length=0, colors=MUTED, labelcolor=INK)
    for (i, j), value in np.ndenumerate(grid):
        if np.isfinite(value):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if value > 0.55 else INK)
    for start, stop, name in ((0, n_id, "Test ID"), (n_id, n_lines, "Test OOD")):
        if stop > start:
            ax.text(len(order) - 0.5 + 0.12, (start + stop - 1) / 2, name, rotation=270, ha="left", va="center",
                    fontsize=8, color=MUTED)
    if 0 < n_id < n_lines:
        ax.axhline(n_id - 0.5, color=INK, linewidth=1.5)
    ax.spines[:].set_visible(False)
    bar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.02, pad=0.015, aspect=40)
    bar.set_label("Dice moyen", fontsize=8, color=INK)
    bar.ax.tick_params(labelsize=8, colors=MUTED, labelcolor=INK)
    bar.outline.set_visible(False)
    paths.append(save(fig, folder / "rq3_dice_par_centre.png"))

    # 3. Dice OOD par classe de volume de lésion (une ligne par technique).
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, split in zip(axes, ("id_test", "ood_test")):
        for index, technique in enumerate(order):
            data = sizes[(sizes["technique"] == technique) & (sizes["split"] == split)]
            data = data.set_index("taille").reindex(SIZE_LABELS)
            ax.plot(FIGURE_SIZE_LABELS, data["dice"], marker="o", markersize=5, linewidth=2,
                    color=COLORS[(ORDER.index(technique) if technique in ORDER else index) % len(COLORS)], label=LABELS.get(technique, technique))
        ax.set_title("Test " + split.replace("_test", "").upper(), color=INK, loc="left")
        ax.set_xlabel("Volume de la lésion")
        ax.set_ylim(0, 1)
        style(ax)
    axes[0].set_ylabel("Dice moyen")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    paths.append(save(fig, folder / "rq3_dice_par_taille.png"))

    # 4. Différences appariées à la baseline sur le test OOD (une boîte par technique).
    others = [t for t in order if t != BASELINE]
    differences = [paired(rows, t, "ood_test")[0] - paired(rows, t, "ood_test")[1] for t in others]
    if others and any(len(d) for d in differences):
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(others)), 4))
        ax.boxplot([d if len(d) else [np.nan] for d in differences], showfliers=False,
                   medianprops=dict(color=COLORS[1], linewidth=2), boxprops=dict(color=MUTED),
                   whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
        ax.axhline(0, color=INK, linewidth=1)
        ax.set_xticks(range(1, len(others) + 1), [LABELS.get(t, t) for t in others], rotation=20, ha="right")
        ax.set_ylabel("Dice technique - Dice baseline")
        style(ax)
        ax.set_title("Test OOD - différences appariées à la baseline", color=INK, loc="left")
        paths.append(save(fig, folder / "rq3_delta_ood.png"))
    return paths


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def direct_comparisons(rows, pairs=COMPARISONS, n_boot=10000):
    """Comparaisons directes A - B (sujets appariés) : global ID et OOD, puis par centre OOD.

    Hors correction de Holm : comparaisons secondaires entre deux techniques candidates au modèle final.
    """
    present = set(rows["technique"])
    overall, by_center = [], []
    for a, b in pairs:
        if a not in present or b not in present:
            continue
        for split in ("id_test", "ood_test"):
            x, y = paired(rows, a, split, reference=b)
            d = x - y
            mean, low, high = bootstrap_mean(d, n_boot) if len(d) else (math.nan,) * 3
            overall.append(dict(a=a, b=b, split=split, n=len(d), dice_a=float(np.mean(x)) if len(d) else math.nan,
                                dice_b=float(np.mean(y)) if len(d) else math.nan, delta=mean, ic_bas=low,
                                ic_haut=high, mediane=float(np.median(d)) if len(d) else math.nan,
                                favorables_a=int((d > 0).sum()), favorables_b=int((d < 0).sum()),
                                p_wilcoxon=wilcoxon_p(x, y)))
        ood = scored(rows[(rows["split"] == "ood_test") & rows["technique"].isin([a, b])])
        grid = ood.pivot_table(index=["site", "case_id"], columns="technique", values="dice").dropna()
        for site, group in grid.groupby(level="site"):
            d = group[a] - group[b]
            by_center.append(dict(a=a, b=b, site=site, n=len(d), dice_a=float(group[a].mean()),
                                  dice_b=float(group[b].mean()), delta=float(d.mean()),
                                  favorables_a=int((d > 0).sum()), favorables_b=int((d < 0).sum()),
                                  p_wilcoxon=wilcoxon_p(group[a], group[b])))
    columns = ["a", "b", "split", "n", "dice_a", "dice_b", "delta", "ic_bas", "ic_haut", "mediane", "favorables_a",
               "favorables_b", "p_wilcoxon"]
    center_columns = ["a", "b", "site", "n", "dice_a", "dice_b", "delta", "favorables_a", "favorables_b",
                      "p_wilcoxon"]
    return pd.DataFrame(overall, columns=columns), pd.DataFrame(by_center, columns=center_columns)


def markdown_comparisons(overall, by_center):
    if overall.empty:
        return ""
    name = lambda t: LABELS.get(t, t)
    lines = ["## Comparaisons directes (hors correction de Holm)", "",
             "| A - B | Test | n | Dice A | Dice B | Δ moyen [IC 95 %] | Médiane | Sujets A > B / A < B | p Wilcoxon |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in overall.iterrows():
        lines.append(f"| {name(r.a)} - {name(r.b)} | {r.split.replace('_test', '').upper()} | {r.n} | {r.dice_a:.3f} | "
                     f"{r.dice_b:.3f} | {fmt_ci(r.delta, r.ic_bas, r.ic_haut)} | {r.mediane:+.4f} | "
                     f"{r.favorables_a} / {r.favorables_b} | {fmt_p(r.p_wilcoxon)} |")
    lines += ["", "Par centre OOD :", "",
              "| A - B | Centre | n | Dice A | Dice B | Δ moyen | Sujets A > B / A < B | p Wilcoxon |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in by_center.iterrows():
        lines.append(f"| {name(r.a)} - {name(r.b)} | {r.site} | {r.n} | {r.dice_a:.3f} | {r.dice_b:.3f} | "
                     f"{r.delta:+.3f} | {r.favorables_a} / {r.favorables_b} | {fmt_p(r.p_wilcoxon)} |")
    return "\n".join(lines) + "\n"


def aggregate(inputs, output, figures_dir=None, n_boot=10000, pairs=COMPARISONS):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = load_results(inputs)
    if BASELINE not in set(rows["technique"]):
        warnings.warn("Pas de résultats baseline : Δ et Wilcoxon non calculés")
    table = ablation_table(rows, n_boot)
    centers, sizes, training = per_center(rows), per_size(rows), load_training(inputs)
    overall, by_center = direct_comparisons(rows, pairs, n_boot)
    table.to_csv(output / "tableau_rq3.csv", index=False)
    centers.to_csv(output / "par_centre.csv", index=False)
    sizes.to_csv(output / "par_taille.csv", index=False)
    training.to_csv(output / "entrainement.csv", index=False)
    overall.to_csv(output / "comparaisons.csv", index=False)
    by_center.to_csv(output / "comparaisons_par_centre.csv", index=False)
    sources = rows.groupby("technique")["source"].unique()
    text = ["# RQ3 - tableau d'ablation (pli 0)", "",
            f"Bootstrap : {n_boot} tirages, IC 95 % percentile. Wilcoxon apparié sur les sujets communs avec la "
            "baseline, correction de Holm sur les techniques. Dice sur les sujets à lésion non vide.", "",
            markdown_table(table), markdown_comparisons(overall, by_center), "## Sources", ""]
    text += [f"- {LABELS.get(t, t)} : " + ", ".join(sorted(map(str, sources[t]))) for t in techniques_in(rows)]
    (output / "tableau_rq3.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    paths = figures(rows, table, centers, sizes, Path(figures_dir or REPO_ROOT / "figures" / "rq3"))
    return dict(table=table, centers=centers, sizes=sizes, training=training, comparisons=overall,
                comparisons_by_center=by_center, figures=paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="dossiers de sorties (un par run, ou parents) ou archives .zip")
    parser.add_argument("--output", default=str(REPO_ROOT / "adaptation" / "runs" / "rq3_agregation"))
    parser.add_argument("--figures", default=str(REPO_ROOT / "figures" / "rq3"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--comparer", action="append", metavar="A:B",
                        help="comparaison directe A - B (répétable ; défaut : " +
                             ", ".join(f"{a}:{b}" for a, b in COMPARISONS) + ")")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # console Windows (cp1252)
    pairs = [tuple(pair.split(":", 1)) for pair in args.comparer] if args.comparer else COMPARISONS
    result = aggregate(args.inputs, args.output, args.figures, args.bootstrap, pairs)
    print(markdown_table(result["table"]))
    print(markdown_comparisons(result["comparisons"], result["comparisons_by_center"]))
    print(result["training"].drop(columns="source").to_string(index=False))
    for path in result["figures"]:
        print(path)


if __name__ == "__main__":
    main()
