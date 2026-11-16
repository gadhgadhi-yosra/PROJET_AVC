"""Analyse des échecs RQ3 (pli 0) : centres OOD, taille des lésions, distance de domaine, cas R008 et R053.

    .venv-dl\\Scripts\\python -m adaptation.analyse_echecs data\\kaggle_sorties\\rq3-baseline-fp32-pli0 ^
        data\\kaggle_sorties\\rq3-fda-pli0 data\\kaggle_sorties\\rq3-intensity-pli0 ^
        data\\kaggle_sorties\\rq3-adaptation-continue-pli0

Entrées :
- `resultats_*.csv` des runs (lus par `aggregate_rq3.load_results`) ;
- `prequentiel.csv` de l'adaptation continue (Dice de chaque sujet avant adaptation sur son lot) ;
- `analysis/reports/center_distances_v1.csv` (distance de domaine C6 de Yosra : Wasserstein et MMD entre
  histogrammes d'intensité du cerveau, référence = tous les autres centres) ;
- les volumes prétraités (`data/processed`) pour le contraste des lésions des sujets OOD. Le résultat est mis
  en cache dans `contraste_lesions.csv` ; `--sans-volumes` saute ce calcul.

Contraste d'une lésion : z-score moyen dans la lésion moins z-score moyen d'un anneau de 1 à 4 mm autour
d'elle (dans le cerveau). Une lésion chronique en T1 est hypo-intense (contraste ≈ -1) ; un contraste
proche de 0 signifie une lésion invisible en T1.

Sorties (`--output`, défaut adaptation/outputs/rq3/echecs) : par_centre_ood.csv, par_taille.csv,
spearman.csv, zoom_R008_R053.csv, echecs.md ; figures dans figures/rq3/ (préfixe rq3_echecs_).
Le Dice n'est défini que pour les sujets à lésion non vide.
"""
import argparse
import io
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .aggregate_rq3 import (COLORS, GRID, INK, LABELS, MUTED, SIZE_BINS, SIZE_LABELS, iter_files, load_results,
                            save, scored, style)

REPO_ROOT = Path(__file__).resolve().parents[1]
DISTANCES = REPO_ROOT / "analysis" / "reports" / "center_distances_v1.csv"
PREQUENTIAL_FILE = re.compile(r"(^|/)prequentiel\.csv$")
CONTINUAL = {"fige": "continue_fige", "controlee": "continue_controlee", "tent": "continue_tent",
             "auto_apprentissage": "continue_auto_apprentissage"}
TECHNIQUE_ORDER = ["baseline", "intensity", "fda", "baseline+tent", "continue_fige", "continue_controlee",
                   "continue_tent", "continue_auto_apprentissage"]
TECHNIQUE_LABELS = {**LABELS, "continue_fige": "Continue (figé)", "continue_controlee": "Continue (contrôlée)",
                    "continue_tent": "Continue (TENT)", "continue_auto_apprentissage": "Continue (auto-appr.)"}
# Colonnes du tableau markdown (les autres stratégies continues restent dans les CSV).
SHOWN = ["baseline", "intensity", "fda", "baseline+tent", "continue_controlee"]
ZOOM = ("R008", "R053")
REFERENCE_TECHNIQUE = "intensity"


# --- Lecture ---

def load_prequential(inputs):
    """Lignes de prequentiel.csv au format des résultats (technique = continue_<stratégie>, split ood_test)."""
    frames = []
    for name, _, content in iter_files(inputs, PREQUENTIAL_FILE):
        frame = pd.read_csv(io.BytesIO(content))
        frames.append(frame.assign(source=name))
    if not frames:
        return pd.DataFrame(columns=["technique", "split", "case_id", "site", "dice", "gt_ml"])
    rows = pd.concat(frames, ignore_index=True).drop_duplicates(["strategie", "case_id"], keep="last")
    rows = rows.rename(columns={"centre": "site"})
    rows["technique"] = rows["strategie"].map(CONTINUAL).fillna("continue_" + rows["strategie"])
    rows["split"] = "ood_test"
    for column in ("dice", "gt_ml"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    return rows[["technique", "split", "case_id", "site", "dice", "gt_ml", "source"]]


def present(rows):
    found = set(rows["technique"])
    return [t for t in TECHNIQUE_ORDER if t in found] + sorted(found - set(TECHNIQUE_ORDER))


def lesion_contrast(t1, mask, brain, ring=(1, 4)):
    """z moyen de la lésion - z moyen d'un anneau de `ring` mm autour (dans le cerveau) ; nan si masque vide."""
    from scipy import ndimage
    lesion = mask > 0
    if not lesion.any():
        return dict(z_lesion=math.nan, z_anneau=math.nan, contraste=math.nan, n_composantes=0)
    inner = ndimage.binary_dilation(lesion, iterations=ring[0])
    outer = ndimage.binary_dilation(lesion, iterations=ring[1])
    around = outer & ~inner & (brain > 0)
    z_lesion = float(t1[lesion].mean())
    z_ring = float(t1[around].mean()) if around.any() else math.nan
    return dict(z_lesion=z_lesion, z_anneau=z_ring, contraste=z_lesion - z_ring,
                n_composantes=int(ndimage.label(lesion)[1]))


def brain_profile(t1, brain):
    """Percentiles du z-score dans le cerveau et part des voxels très brillants (z > 3)."""
    values = t1[brain > 0]
    p1, p50, p99 = np.percentile(values, [1, 50, 99])
    return dict(z_p1=float(p1), z_p50=float(p50), z_p99=float(p99), part_z_sup_3=float((values > 3).mean()))


def subject_volumes(record):
    import nibabel as nib
    return tuple(np.asarray(nib.load(record[f"{name}_path"]).get_fdata(dtype=np.float32))
                 for name in ("t1", "mask", "brain"))


def contrast_table(data_root, splits, cache, recompute=False):
    """Contraste des lésions et profil d'intensité de chaque sujet OOD (mis en cache dans `cache`)."""
    cache = Path(cache)
    if cache.is_file() and not recompute:
        return pd.read_csv(cache)
    from generalization.data_processed import list_processed
    records = list_processed(data_root, "ood_test", splits=splits, on_missing="warn", warn_empty=False)
    rows = []
    for index, record in records.iterrows():
        t1, mask, brain = subject_volumes(record)
        rows.append(dict(case_id=record["case_id"], site=record["site"], **lesion_contrast(t1, mask, brain),
                         **brain_profile(t1, brain)))
        if (index + 1) % 50 == 0:
            print(f"contraste : {index + 1}/{len(records)} sujets", flush=True)
    table = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(cache, index=False)
    return table


# --- Tableaux ---

def per_center(rows, distances, contrast=None):
    """Une ligne par centre OOD : effectifs, volume médian, distance C6, Dice moyen par technique."""
    ood = rows[rows["split"] == "ood_test"]
    reference = ood[ood["technique"] == REFERENCE_TECHNIQUE]
    if reference.empty:
        reference = ood.drop_duplicates("case_id")
    subjects = reference.groupby("site").agg(n_sujets=("case_id", "nunique"),
                                             n_lesion=("gt_ml", lambda s: int((s > 0).sum())),
                                             volume_median_ml=("gt_ml", lambda s: float(s[s > 0].median())),
                                             jours_post_avc=("days_post_stroke", "median")
                                             if "days_post_stroke" in reference else ("gt_ml", "size"))
    dice = scored(ood).pivot_table(index="site", columns="technique", values="dice", aggfunc="mean")
    dice = dice.reindex(columns=present(ood))
    table = subjects.join(distances.set_index("site")[["wasserstein", "mmd"]], how="left").join(dice, how="left")
    if contrast is not None and len(contrast):
        table = table.join(contrast.groupby("site")["contraste"].median().rename("contraste_median"), how="left")
        table = table.join(contrast.groupby("site")["part_z_sup_3"].median().rename("part_z_sup_3"), how="left")
    return table.sort_values("wasserstein", ascending=False).reset_index()


def per_size(rows):
    data = scored(rows).copy()
    data["taille"] = pd.cut(data["gt_ml"], SIZE_BINS, labels=SIZE_LABELS, right=False)
    table = (data.groupby(["split", "taille", "technique"], observed=True)["dice"].agg(dice="mean", n="count")
             .reset_index())
    return table


def spearman(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return math.nan, math.nan, int(keep.sum())
    result = spearmanr(x[keep], y[keep])
    return float(result.statistic), float(result.pvalue), int(keep.sum())


def correlations(rows, distances, centers, contrast=None):
    """Spearman distance-Dice (centres OOD, puis tous les centres de test), et facteurs au niveau sujet."""
    records = []
    techniques = [t for t in present(rows) if t in centers]
    for metric in ("wasserstein", "mmd"):
        for technique in techniques:
            rho, p, n = spearman(centers[metric], centers[technique])
            records.append(dict(niveau="centres OOD", x=metric, y=f"Dice {technique}", rho=rho, p=p, n=n))
        rho, p, n = spearman(centers[metric], centers["volume_median_ml"])
        records.append(dict(niveau="centres OOD", x=metric, y="volume médian des lésions", rho=rho, p=p, n=n))
    # Tous les centres de test (ID et OOD) : plus de points, mais les centres ID sont vus à l'entraînement.
    all_centers = scored(rows[rows["technique"].isin(["baseline", "intensity", "fda", "baseline+tent"])])
    all_centers = all_centers.groupby(["technique", "site"])["dice"].mean().reset_index()
    all_centers = all_centers.merge(distances[["site", "wasserstein"]], on="site")
    for technique, group in all_centers.groupby("technique"):
        rho, p, n = spearman(group["wasserstein"], group["dice"])
        records.append(dict(niveau="centres de test ID + OOD", x="wasserstein", y=f"Dice {technique}",
                            rho=rho, p=p, n=n))
    subjects = scored(rows[(rows["split"] == "ood_test") & (rows["technique"] == REFERENCE_TECHNIQUE)])
    rho, p, n = spearman(subjects["gt_ml"], subjects["dice"])
    records.append(dict(niveau="sujets OOD", x="volume de la lésion", y=f"Dice {REFERENCE_TECHNIQUE}", rho=rho, p=p,
                        n=n))
    if contrast is not None and len(contrast):
        merged = subjects.merge(contrast[["case_id", "contraste"]], on="case_id")
        rho, p, n = spearman(merged["contraste"], merged["dice"])
        records.append(dict(niveau="sujets OOD", x="contraste de la lésion", y=f"Dice {REFERENCE_TECHNIQUE}",
                            rho=rho, p=p, n=n))
    return pd.DataFrame(records)


def zoom(rows, contrast=None, sites=ZOOM):
    """Détail sujet par sujet des centres `sites` (technique de référence) avec contraste et intensités."""
    data = rows[(rows["split"] == "ood_test") & (rows["technique"] == REFERENCE_TECHNIQUE) & rows["site"].isin(sites)]
    columns = ["case_id", "site", "days_post_stroke", "gt_ml", "pred_ml", "dice", "hd95_mm", "lesion_count_diff"]
    table = data[[c for c in columns if c in data]].copy()
    others = scored(rows[rows["split"] == "ood_test"]).pivot_table(index="case_id", columns="technique",
                                                                   values="dice")
    table = table.join(others.add_prefix("dice_"), on="case_id")
    if contrast is not None and len(contrast):
        table = table.merge(contrast.drop(columns="site"), on="case_id", how="left")
    return table.sort_values(["site", "case_id"]).reset_index(drop=True)


# --- Markdown ---

def fmt(value, digits=2):
    return "-" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def markdown(centers, sizes, correlation, zoomed):
    shown = [t for t in SHOWN if t in centers]
    lines = ["# RQ3 - analyse des échecs (pli 0)", "",
             "Dice moyen sur les sujets à lésion non vide. Distance C6 : Wasserstein entre histogrammes "
             "d'intensité du cerveau (référence : tous les autres centres). Contraste : z lésion - z anneau "
             "1-4 mm (médiane du centre). « Continue » : Dice préquentiel, sur un sous-ensemble des sujets du centre.",
             "", "## Centres OOD", ""]
    header = ["Centre", "Sujets (lésion)", "Vol. médian (ml)", "J post-AVC", "Distance W1", "Contraste"]
    header += [TECHNIQUE_LABELS.get(t, t) for t in shown]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for _, r in centers.iterrows():
        cells = [r["site"], f"{r['n_sujets']} ({r['n_lesion']})", fmt(r["volume_median_ml"]),
                 fmt(r.get("jours_post_avc", math.nan), 0), fmt(r["wasserstein"], 3), fmt(r.get("contraste_median",
                                                                                               math.nan))]
        cells += [fmt(r[t]) for t in shown]
        lines.append("| " + " | ".join(map(str, cells)) + " |")
    lines += ["", "## Dice par volume de lésion (test OOD ; ID entre parenthèses)", ""]
    size_techniques = [t for t in shown if t in set(sizes["technique"])]
    lines += ["| Volume | " + " | ".join(TECHNIQUE_LABELS.get(t, t) for t in size_techniques) + " |",
              "|---|" + "---|" * len(size_techniques)]
    for label in SIZE_LABELS:
        cells = [label]
        for technique in size_techniques:
            values = {}
            for split in ("ood_test", "id_test"):
                match = sizes[(sizes["split"] == split) & (sizes["taille"] == label) & (sizes["technique"] == technique)]
                values[split] = (float(match["dice"].iloc[0]), int(match["n"].iloc[0])) if len(match) else None
            ood, id_ = values["ood_test"], values["id_test"]
            text = f"{ood[0]:.2f} (n={ood[1]})" if ood else "-"
            cells.append(text + (f" ({id_[0]:.2f})" if id_ else ""))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Corrélations de Spearman", "", "| Niveau | x | y | rho | p | n |", "|---|---|---|---|---|---|"]
    hidden = {f"Dice {t}" for t in TECHNIQUE_ORDER if t not in SHOWN}  # restent dans spearman.csv
    for _, r in correlation[~correlation["y"].isin(hidden)].iterrows():
        lines.append(f"| {r['niveau']} | {r['x']} | {r['y']} | {fmt(r['rho'])} | {fmt(r['p'], 3)} | {r['n']} |")
    lines += ["", "## R008 et R053, sujet par sujet (modèle intensité)", ""]
    zoom_columns = [c for c in ("case_id", "days_post_stroke", "gt_ml", "pred_ml", "dice", "lesion_count_diff",
                                "n_composantes", "z_lesion", "z_anneau", "contraste", "z_p99", "part_z_sup_3")
                    if c in zoomed]
    lines += ["| " + " | ".join(zoom_columns) + " |", "|" + "---|" * len(zoom_columns)]
    for _, r in zoomed.iterrows():
        cells = [r[c] if isinstance(r[c], str) else fmt(float(r[c]), 3 if c == "part_z_sup_3" else 2)
                 for c in zoom_columns]
        lines.append("| " + " | ".join(map(str, cells)) + " |")
    return "\n".join(lines) + "\n"


# --- Figures ---

def figures(rows, centers, sizes, contrast, folder, data_root=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    shown = [t for t in SHOWN if t in centers]
    color = {t: COLORS[i % len(COLORS)] for i, t in enumerate(shown)}

    # 1. Dice OOD moyen par centre, centres rangés par distance C6 croissante (techniques décalées).
    ordered = centers.sort_values("wasserstein").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(ordered))
    width = 0.7 / max(len(shown), 1)
    for index, technique in enumerate(shown):
        offset = (index - (len(shown) - 1) / 2) * width
        ax.plot(x + offset, ordered[technique], "o", markersize=8, color=color[technique],
                markeredgecolor="white", markeredgewidth=1.5, label=TECHNIQUE_LABELS.get(technique, technique))
    ax.set_xticks(x, [f"{r.site}\nW1 = {r.wasserstein:.3f}\nn = {r.n_lesion}" for r in ordered.itertuples()],
                  fontsize=8)
    ax.set_xlabel("Centre OOD, par distance croissante au reste des centres (Wasserstein)")
    ax.set_ylabel("Dice OOD moyen du centre")
    ax.set_ylim(-0.05, 1)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style(ax)
    ax.set_title("Distance de domaine et Dice par centre OOD", color=INK, loc="left")
    paths.append(save(fig, folder / "rq3_echecs_distance_dice.png"))

    # 2. Dice par volume de lésion, test OOD, avec l'adaptation continue.
    fig, ax = plt.subplots(figsize=(7, 4.2))
    data = sizes[sizes["split"] == "ood_test"]
    for technique in shown:
        line = data[data["technique"] == technique].set_index("taille").reindex(SIZE_LABELS)
        ax.plot(SIZE_LABELS, line["dice"], marker="o", markersize=6, linewidth=2, color=color[technique],
                label=TECHNIQUE_LABELS.get(technique, technique)
                + (", sous-ensemble" if technique.startswith("continue_") else ""))
    ax.set_xlabel("Volume de la lésion (vérité terrain)")
    ax.set_ylabel("Dice moyen")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax)
    ax.set_title("Test OOD - Dice par volume de lésion", color=INK, loc="left")
    paths.append(save(fig, folder / "rq3_echecs_dice_par_taille.png"))

    if contrast is None or not len(contrast):
        return paths

    # 3. Contraste de la lésion et Dice (sujets OOD, modèle intensité) : R008 et R053 en couleur.
    subjects = scored(rows[(rows["split"] == "ood_test") & (rows["technique"] == REFERENCE_TECHNIQUE)])
    merged = subjects.merge(contrast[["case_id", "contraste"]], on="case_id")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    rest = merged[~merged["site"].isin(ZOOM)]
    ax.plot(rest["contraste"], rest["dice"], "o", markersize=5, color=GRID, markeredgecolor=MUTED,
            markeredgewidth=0.5, label="Autres centres OOD")
    for index, site in enumerate(ZOOM):
        part = merged[merged["site"] == site]
        ax.plot(part["contraste"], part["dice"], "o", markersize=8, color=COLORS[index + 1], markeredgecolor="white",
                markeredgewidth=1.5, label=site)
    ax.axvline(0, color=MUTED, linewidth=1, linestyle="--")
    ax.set_xlabel("Contraste de la lésion en T1 (z lésion - z anneau 1-4 mm)")
    ax.set_ylabel("Dice (modèle intensité)")
    ax.set_ylim(-0.05, 1)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style(ax)
    ax.set_title("Contraste de la lésion en T1 et Dice (sujets OOD)", color=INK, loc="left")
    paths.append(save(fig, folder / "rq3_echecs_contraste_dice.png"))

    # 4. Coupes : lésion de R008 (aiguë) à côté d'une lésion OOD typique, contour de la vérité terrain.
    if data_root is not None:
        path = r008_slices(rows, contrast, folder, data_root, plt)
        if path:
            paths.append(path)
    return paths


def r008_slices(rows, contrast, folder, data_root, plt):
    from generalization.data_processed import list_processed
    records = list_processed(data_root, "ood_test", on_missing="warn", warn_empty=False).set_index("case_id")
    lesions = contrast[contrast["contraste"].notna()]
    picks = []
    r008 = lesions[lesions["site"] == "R008"].merge(rows[["case_id", "gt_ml"]].drop_duplicates("case_id"), on="case_id")
    if len(r008):
        picks.append(("R008 (J1)", r008.sort_values("gt_ml").iloc[-1]["case_id"]))
    typical = lesions[~lesions["site"].isin(ZOOM)]
    typical = typical.iloc[(typical["contraste"] - typical["contraste"].median()).abs().argsort()]
    if len(typical):
        picks.append((f"{typical.iloc[0]['site']} (contraste médian)", typical.iloc[0]["case_id"]))
    if not picks:
        return None
    fig, axes = plt.subplots(1, len(picks), figsize=(4.2 * len(picks), 4.4))
    for ax, (title, case) in zip(np.atleast_1d(axes), picks):
        t1, mask, brain = subject_volumes(records.loc[case])
        z = int(np.argmax(mask.sum(axis=(0, 1))))
        ax.imshow(np.rot90(t1[:, :, z]), cmap="gray", vmin=-3, vmax=3)
        ax.contour(np.rot90(mask[:, :, z]), levels=[0.5], colors=[COLORS[1]], linewidths=1.2)
        ax.set_title(f"{title}\n{case}", fontsize=8, color=INK, loc="left")
        ax.axis("off")
    fig.suptitle("Contour orange : lésion annotée (coupe axiale où elle est la plus grande). "
                 "z-score affiché de -3 à 3.", fontsize=8, color=MUTED)
    return save(fig, folder / "rq3_echecs_coupes_R008.png")


# --- Programme ---

def analyse(inputs, output, figures_dir, data_root=None, splits=None, distances_path=DISTANCES, recompute=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = load_results(inputs)
    rows = pd.concat([results, load_prequential(inputs)], ignore_index=True)
    distances = pd.read_csv(distances_path)
    contrast = None
    if data_root is not None:
        from generalization.calibration import DEFAULT_SPLITS
        contrast = contrast_table(data_root, splits or DEFAULT_SPLITS, output / "contraste_lesions.csv", recompute)
    centers = per_center(rows, distances, contrast)
    sizes = per_size(rows)
    correlation = correlations(rows, distances, centers, contrast)
    zoomed = zoom(rows, contrast)
    centers.to_csv(output / "par_centre_ood.csv", index=False)
    sizes.to_csv(output / "par_taille.csv", index=False)
    correlation.to_csv(output / "spearman.csv", index=False)
    zoomed.to_csv(output / "zoom_R008_R053.csv", index=False)
    text = markdown(centers, sizes, correlation, zoomed)
    (output / "echecs.md").write_text(text, encoding="utf-8")
    paths = figures(rows, centers, sizes, contrast, Path(figures_dir), data_root)
    return dict(text=text, centers=centers, sizes=sizes, correlation=correlation, zoom=zoomed, figures=paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="dossiers de sorties des runs (ou archives .zip)")
    parser.add_argument("--output", default=str(REPO_ROOT / "adaptation" / "outputs" / "rq3" / "echecs"))
    parser.add_argument("--figures", default=str(REPO_ROOT / "figures" / "rq3"))
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data" / "processed"))
    parser.add_argument("--distances", default=str(DISTANCES))
    parser.add_argument("--sans-volumes", action="store_true", help="ne pas lire les volumes (pas de contraste)")
    parser.add_argument("--recalculer", action="store_true", help="recalculer le contraste même s'il est en cache")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    result = analyse(args.inputs, args.output, args.figures, None if args.sans_volumes else args.data_root,
                     distances_path=args.distances, recompute=args.recalculer)
    print(result["text"])
    for path in result["figures"]:
        print(path)


if __name__ == "__main__":
    main()
