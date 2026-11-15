"""Génère les notebooks Kaggle du tableau d'ablation RQ3 (un par technique, mêmes pli et graine).

    .venv-dl\\Scripts\\python adaptation\\kaggle\\make_notebooks.py
    .venv-dl\\Scripts\\python adaptation\\kaggle\\make_notebooks.py --kernels-dir data\\kaggle_kernels   # + kaggle kernels push

Le code vient du dataset Kaggle privé `projet-avc-code` (voir build_code_dataset.py), pas de GitHub :
le dépôt devient privé. Les données viennent du dataset privé `atlas-c1-processed`.
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
USERNAME = "fayfahachicha"
DATASET = "atlas-c1-processed"
CODE_DATASET = "projet-avc-code"
FOLD, SEED = 0, 2026
RUNS = [  # fichier, technique, titre, durée estimée sur une T4 (entraînement + validation + évaluation)
    ("00_baseline", "baseline", "Baseline (aucune technique de généralisation), puis TENT au test", "6 h 30 à 7 h 30"),
    ("01_intensite", "intensity", "Augmentation d'intensité (gamma, contraste, bruit, flou, biais, basse résolution)", "6 h 30 à 7 h"),
    ("02_bigaug", "bigaug", "Preset BigAug", "6 h 30 à 7 h"),
    ("03_fda", "fda", "FDA (échange d'amplitude basse fréquence)", "7 à 9 h"),
    ("04_histogram_matching", "histogram_matching", "Histogram matching (aussi appliqué à la validation et au test)", "7 à 7 h 30"),
    ("05_dann", "dann", "DANN (classifieur de centre et inversion de gradient)", "6 h 30 à 7 h"),
]


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def notebook(name, technique, title, duration):
    run = f"unet3d_{technique}_pli{FOLD}"
    cells = [
        markdown(f"""
# RQ3 — {title}

Run `{run}` du tableau d'ablation : UNet MONAI 3D en fp32 (clipping des gradients à 12), patches de 96³, **pli {FOLD}** de `splits.json`, **graine {SEED}**,
40 000 itérations (lots de 4 patches), identiques pour toutes les techniques. Durée estimée sur une T4 : {duration}
(garde-fou à 10 h d'entraînement, session limitée à 12 h).

**Avant de lancer** : accélérateur *GPU T4*, Internet activé (installation de MONAI et TorchIO), datasets privés
`{DATASET}` (données C1) et `{CODE_DATASET}` (code) ajoutés en entrée.
Notebook à garder **privé** (accord d'utilisation ATLAS). *Save Version → Save & Run All* pour une exécution en arrière-plan.
"""),
        code(f"""
# Code du projet (dataset privé {CODE_DATASET}) et dépendances absentes de l'image Kaggle.
import hashlib, json, shutil, subprocess, sys, zipfile
from pathlib import Path

def find_input(slug):  # dossier d'un dataset attaché, quel que soit le chemin de montage de Kaggle
    for pattern in (slug, f"*/{{slug}}", f"*/*/{{slug}}", f"*/*/*/{{slug}}"):
        found = sorted(Path("/kaggle/input").glob(pattern))
        if found:
            return found[0]
    raise FileNotFoundError(f"Dataset {{slug}} non attaché : " + str(sorted(Path("/kaggle/input").glob("*"))))

CODE = find_input("{CODE_DATASET}")
REPO = Path("/kaggle/working/PROJET_AVC")
if not REPO.exists():
    archives = sorted(CODE.rglob("projet_avc_code.zip"))
    if archives:
        zipfile.ZipFile(archives[0]).extractall(REPO)
    else:  # Kaggle a décompressé l'archive : copie de l'arborescence
        shutil.copytree(next(CODE.rglob("adaptation/train.py")).parents[1], REPO)
print(json.loads((REPO / "VERSION.json").read_text()))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "monai==1.6.1", "torchio==1.2.1", "nibabel==5.4.2",
                "pyarrow"], check=True)
import torch; print(torch.__version__, torch.cuda.get_device_name(0))
"""),
        code(f"""
# Données C1 : liens vers les dossiers de sujets, quelle que soit l'archive d'origine.
INPUT = find_input("{DATASET}")
DATA = Path("/kaggle/working/processed")
DATA.mkdir(exist_ok=True)
sources = [INPUT]
if not any(INPUT.rglob("qc.json")):  # archives restées compressées : extraction sur le disque de travail
    extracted = Path("/kaggle/working/c1_extrait")
    for archive in sorted(INPUT.rglob("processed_partie*.zip")):
        zipfile.ZipFile(archive).extractall(extracted)
    sources = [extracted]
for qc in (path for source in sources for path in source.rglob("qc.json")):
    link = DATA / qc.parent.name
    if not link.exists():
        link.symlink_to(qc.parent, target_is_directory=True)
table = DATA / "subjects.parquet"
if not table.exists():
    table.symlink_to(next(INPUT.rglob("subjects.parquet")))
print(sum(1 for _ in DATA.glob("*/qc.json")), "sujets prétraités")
# Même découpage que le code : splits.json du dataset identique à celui du dépôt.
digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert digest(next(INPUT.rglob("splits.json"))) == digest(REPO / "splits.json"), "splits.json différent du dépôt"
"""),
        code(f"""
# Entraînement, validation sur le pli, puis évaluation sur le test ID du pli et le test OOD.
# Session interrompue : rattacher la sortie précédente, copier last.pt dans /kaggle/working/runs/{run}/ et ajouter --resume.
!cd {{REPO}} && python -u -m adaptation.train --technique {technique} --fold {FOLD} --seed {SEED} --data-root {{DATA}} --output /kaggle/working/runs --num-workers 4 --validate-every 2000 --log-every 100
"""),
    ]
    if technique == "baseline":
        cells.append(code(f"""
# TENT au test, par centre, sur le modèle baseline (stratégie « TENT seul »).
!cd {{REPO}} && python -u -m adaptation.train --evaluate-only --checkpoint /kaggle/working/runs/{run}/best.pt --tent --data-root {{DATA}}
"""))
    cells.append(code(f"""
# Résumé et fichiers à récupérer (onglet Output) ; aucune image de patient n'est écrite.
for path in sorted(Path("/kaggle/working/runs/{run}").glob("resume*.json")):
    print(path.name, json.dumps(json.loads(path.read_text()), indent=2, ensure_ascii=False))
!ls -la /kaggle/working/runs/{run}
"""))
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


DIAG = ("06_diag_nan", "rq3-diag-nan")


def diag_notebook(commit=None):
    """Diagnostic des NaN de la baseline : adaptation/diag_nan.py, variantes A à D, 1 000 itérations au plus.

    commit : hash attendu dans VERSION.json du dataset de code ; la cellule de contrôle arrête le notebook
    avant les variantes s'il diffère (copie poussée sur Kaggle uniquement).
    """
    setup = notebook(*RUNS[0])["cells"][1:3]  # code du dataset, dépendances, liens vers les données C1
    check = f'assert version["commit"] == "{commit}", f"Code {{version[\'commit\']}} au lieu de {commit}"' if commit else ""
    cells = [
        markdown(f"""
# RQ3 — Diagnostic des NaN de la baseline (pli {FOLD}, graine {SEED})

`adaptation/diag_nan.py` : même réseau, mêmes données, même pli et même graine que la baseline, 1 000 itérations
au plus par variante, journal par itération. A = baseline (fp16, Dice par patch, sans clipping ni warmup),
B = fp32, C = Dice sur le lot, D = unscale_ + clip_grad_norm_(12) + warmup sur 500 itérations.
Au premier poids ou moment non fini : lot, état avant le pas et gradients du pas dans `diag_nan/<variante>/rupture/`.
"""),
        code("""
# Environnement (baseline : torch 2.11.0+cu130 sur Tesla T4) ; un écart est signalé dans resume.json sans arrêter.
import torch
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| GPU",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "aucun")
"""),
        *setup,
        code(f"""
# Version du code (VERSION.json du dataset projet-avc-code), contrôlée avant de lancer les variantes.
version = json.loads((REPO / "VERSION.json").read_text())
print("Commit du code :", version["commit"], "-", version["message"])
{check}
"""),
        code("""
# Diagnostic : les quatre variantes l'une après l'autre ; sortie complète aussi dans diag_nan/diag_nan.log.
!mkdir -p /kaggle/working/diag_nan && cd {REPO} && python -u -m adaptation.diag_nan --data-root {DATA} --output /kaggle/working/diag_nan --num-workers 4 2>&1 | tee /kaggle/working/diag_nan/diag_nan.log
"""),
        code("""
# Fichiers à récupérer (onglet Output).
print((Path("/kaggle/working/diag_nan") / "resume.json").read_text())
!find /kaggle/working/diag_nan -type f -exec ls -la {} +
"""),
    ]
    return {**notebook(*RUNS[0]), "cells": cells}


def diag_metadata():
    name, slug = DIAG
    return {**kernel_metadata(name, "baseline"), "id": f"{USERNAME}/{slug}", "title": slug,
            "code_file": f"{name}.ipynb"}


def kernel_metadata(name, technique):
    """Métadonnées de `kaggle kernels push` : notebook privé, GPU, Internet, les deux datasets privés."""
    slug = f"rq3-{technique.replace('_', '-')}-pli{FOLD}"
    return {"id": f"{USERNAME}/{slug}", "title": slug, "code_file": f"{name}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [f"{USERNAME}/{DATASET}", f"{USERNAME}/{CODE_DATASET}"],
            "competition_sources": [], "kernel_sources": []}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kernels-dir", type=Path,
                        help="écrit aussi un dossier par notebook avec kernel-metadata.json (kaggle kernels push)")
    args = parser.parse_args()
    jobs = [(name, notebook(name, technique, title, duration), kernel_metadata(name, technique))
            for name, technique, title, duration in RUNS]
    jobs.append((DIAG[0], diag_notebook(), diag_metadata()))
    for name, content, metadata in jobs:
        path = HERE / f"{name}.ipynb"
        write_notebook(path, content)
        print(path)
        if args.kernels_dir:
            folder = args.kernels_dir / name
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, folder / path.name)
            (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if args.kernels_dir:  # copie poussée sur Kaggle : contrôle du commit attendu (HEAD, celui du dataset de code)
        commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], check=True,
                                capture_output=True, text=True).stdout.strip()
        write_notebook(args.kernels_dir / DIAG[0] / f"{DIAG[0]}.ipynb", diag_notebook(commit))
        print(f"{DIAG[0]} (Kaggle) : commit attendu {commit}")


def write_notebook(path, content):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(content, handle, ensure_ascii=False, indent=1)
        handle.write("\n")


if __name__ == "__main__":
    main()
