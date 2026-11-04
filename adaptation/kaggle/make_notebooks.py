"""Génère les notebooks Kaggle du tableau d'ablation RQ3 (un par technique, mêmes pli et graine).

    .venv-dl\\Scripts\\python adaptation\\kaggle\\make_notebooks.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = "https://github.com/gadhgadhi-yosra/PROJET_AVC.git"
BRANCH = "adaptation-entrainement"
DATASET = "atlas-c1-processed"
FOLD, SEED = 0, 2026
RUNS = [  # fichier, technique, titre, durée estimée sur une T4 (entraînement + validation + évaluation)
    ("00_baseline", "baseline", "Baseline (aucune technique de généralisation), puis TENT au test", "2 à 3 h"),
    ("01_intensite", "intensity", "Augmentation d'intensité (gamma, contraste, bruit, flou, biais, basse résolution)", "3 à 4 h"),
    ("02_bigaug", "bigaug", "Preset BigAug", "4 à 6 h"),
    ("03_fda", "fda", "FDA (échange d'amplitude basse fréquence)", "4 à 6 h"),
    ("04_histogram_matching", "histogram_matching", "Histogram matching (aussi appliqué à la validation et au test)", "3 à 5 h"),
    ("05_dann", "dann", "DANN (classifieur de centre et inversion de gradient)", "2 à 3 h"),
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

Run `{run}` du tableau d'ablation : UNet MONAI 3D, patches de 96³, **pli {FOLD}** de `splits.json`, **graine {SEED}**,
40 000 itérations (lots de 4 patches), identiques pour toutes les techniques. Durée estimée sur une T4 : {duration}
(garde-fou à 10 h d'entraînement, session limitée à 12 h).

**Avant de lancer** : accélérateur *GPU T4*, Internet activé, dataset privé `{DATASET}` ajouté en entrée.
Notebook à garder **privé** (accord d'utilisation ATLAS). *Save Version → Save & Run All* pour une exécution en arrière-plan.
"""),
        code(f"""
# Code du projet (dépôt public) et dépendances absentes de l'image Kaggle.
import hashlib, json, subprocess, sys
from pathlib import Path

REPO = Path("/kaggle/working/PROJET_AVC")
if not REPO.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", "{BRANCH}", "{REPOSITORY}", str(REPO)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "monai==1.6.1", "torchio==1.2.1", "nibabel==5.4.2",
                "pyarrow"], check=True)
print(subprocess.run(["git", "-C", str(REPO), "log", "-1", "--oneline"], capture_output=True, text=True).stdout)
import torch; print(torch.__version__, torch.cuda.get_device_name(0))
"""),
        code(f"""
# Données C1 : liens vers les dossiers de sujets, quelle que soit l'archive d'origine.
import zipfile
INPUT = Path("/kaggle/input/{DATASET}")
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
!cd {{REPO}} && python -m adaptation.train --technique {technique} --fold {FOLD} --seed {SEED} --data-root {{DATA}} --output /kaggle/working/runs --num-workers 4
"""),
    ]
    if technique == "baseline":
        cells.append(code(f"""
# TENT au test, par centre, sur le modèle baseline (stratégie « TENT seul »).
!cd {{REPO}} && python -m adaptation.train --evaluate-only --checkpoint /kaggle/working/runs/{run}/best.pt --tent --data-root {{DATA}}
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


def main():
    for name, technique, title, duration in RUNS:
        path = HERE / f"{name}.ipynb"
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(notebook(name, technique, title, duration), handle, ensure_ascii=False, indent=1)
            handle.write("\n")
        print(path)


if __name__ == "__main__":
    main()
