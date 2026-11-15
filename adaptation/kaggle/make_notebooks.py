"""Génère les notebooks Kaggle du tableau d'ablation RQ3 (un par technique, mêmes pli et graine).

    .venv-dl\\Scripts\\python adaptation\\kaggle\\make_notebooks.py
    .venv-dl\\Scripts\\python adaptation\\kaggle\\make_notebooks.py --kernels-dir data\\kaggle_kernels --handoff-dir data\\kaggle_a_envoyer

- `adaptation/kaggle/*.ipynb` : copies de référence, sans contrôle de commit ;
- `--kernels-dir` : un dossier par notebook du compte propriétaire, avec kernel-metadata.json (`kaggle kernels push`) ;
- `--handoff-dir` : `<prénom>/<slug>.ipynb` pour les runs importées à la main dans l'interface Kaggle
  par les autres membres de l'équipe (répartition : ASSIGNMENT), avec LISEZMOI.md.
Les copies Kaggle contrôlent que VERSION.json du dataset de code porte le commit HEAD.

Le code vient du dataset Kaggle privé `projet-avc-code` (voir build_code_dataset.py), pas de GitHub ;
les données viennent du dataset privé `atlas-c1-processed`. Les deux sont cherchés récursivement sous
/kaggle/input (un dataset partagé peut être monté sous un autre chemin que celui du propriétaire).
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
SLOW_SECONDS = 0.75  # s/it à l'itération 100 au-delà desquelles 40 000 itérations + validations dépassent 10 h
RUNS = [  # fichier, technique, titre, durée estimée sur une T4 (entraînement + validation + évaluation)
    ("00_baseline", "baseline", "Baseline (aucune technique de généralisation), puis TENT au test", "6 h 30 à 7 h 30"),
    ("01_intensite", "intensity", "Augmentation d'intensité (gamma, contraste, bruit, flou, biais, basse résolution)", "6 h 30 à 7 h"),
    ("02_bigaug", "bigaug", "Preset BigAug", "6 h 30 à 7 h"),
    ("03_fda", "fda", "FDA (échange d'amplitude basse fréquence)", "7 à 9 h"),
    ("04_histogram_matching", "histogram_matching", "Histogram matching (aussi appliqué à la validation et au test)", "7 à 7 h 30"),
    ("05_dann", "dann", "DANN (classifieur de centre et inversion de gradient)", "6 h 30 à 7 h"),
]
# Répartition des runs : compte propriétaire (push en ligne de commande) et membres de l'équipe (import manuel).
ASSIGNMENT = {USERNAME: ["baseline", "fda"], "yosra": ["intensity", "histogram_matching"], "anas": ["bigaug", "dann"]}
SLUGS = {"baseline": f"rq3-baseline-fp32-pli{FOLD}"}
CONTINUAL = ("07_adaptation_continue", f"rq3-adaptation-continue-pli{FOLD}")


def kernel_slug(technique):
    """Slug Kaggle du run ; la baseline fp32 a son propre slug (rq3-baseline-pli0 = ancienne run fp16 avec NaN)."""
    return SLUGS.get(technique, f"rq3-{technique.replace('_', '-')}-pli{FOLD}")


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def fill(template, **values):
    for key, value in values.items():
        template = template.replace(f"@@{key}@@", str(value))
    return template


ENVIRONMENT = '''
# Environnement : GPU obligatoire (accélérateur « GPU T4 »), sinon arrêt immédiat avec un message clair.
import hashlib, json, os, re, shutil, subprocess, sys, time, zipfile
from pathlib import Path
import torch
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| GPU",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "aucun")
if not torch.cuda.is_available() and os.environ.get("RQ3_GPU", "1") == "1":
    raise RuntimeError("AUCUN GPU DISPONIBLE : dans le panneau de droite, Settings > Accelerator > « GPU T4 x2 », "
                       "puis relancer (Save Version > Save & Run All). Rien n'a été lancé.")
INPUT = Path(os.environ.get("RQ3_INPUT", "/kaggle/input"))      # datasets attachés
WORKING = Path(os.environ.get("RQ3_WORKING", "/kaggle/working"))  # toutes les sorties
SCRATCH = Path(os.environ.get("RQ3_TMP", "/tmp/rq3"))            # liens vers les données, jamais dans les sorties
EXPECTED_COMMIT = @@COMMIT@@  # commit attendu dans VERSION.json du dataset de code (None : pas de contrôle)
'''

CODE_SETUP = '''
# Code du projet : dataset privé projet-avc-code, cherché récursivement sous /kaggle/input.
def code_version(candidate):
    if candidate.suffix == ".zip":
        with zipfile.ZipFile(candidate) as archive:
            return json.loads(archive.read("VERSION.json")) if "VERSION.json" in archive.namelist() else None
    return json.loads(candidate.read_text())

candidates = sorted(INPUT.rglob("projet_avc_code.zip")) + sorted(
    path for path in INPUT.rglob("VERSION.json") if (path.parent / "adaptation" / "train.py").is_file())
found = [(path, code_version(path)) for path in candidates]
found = [(path, version) for path, version in found if version]
if not found:
    raise FileNotFoundError("Dataset projet-avc-code introuvable sous /kaggle/input : Add Input > Datasets > "
                            "« projet-avc-code » (onglet Shared with you), puis relancer.")
matching = [(path, version) for path, version in found if EXPECTED_COMMIT in (None, version["commit"])]
if not matching:
    raise RuntimeError(f"VERSION.json : commit {sorted({v['commit'] for _, v in found})} au lieu de {EXPECTED_COMMIT}. "
                       "Mettre l'entrée projet-avc-code à sa dernière version (menu ⋮ de l'entrée), puis relancer.")
source, VERSION = matching[0]
print("Code :", source, VERSION)
REPO = WORKING / "PROJET_AVC"
if not REPO.exists():
    if source.suffix == ".zip":
        zipfile.ZipFile(source).extractall(REPO)
    else:  # Kaggle a décompressé l'archive : copie de l'arborescence
        shutil.copytree(source.parent, REPO)
if os.environ.get("RQ3_PIP", "1") == "1":
    pip = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "monai==1.6.1", "torchio==1.2.1",
                          "nibabel==5.4.2", "pyarrow"])
    if pip.returncode:
        raise RuntimeError("Installation de MONAI / TorchIO impossible : activer Internet "
                           "(Settings > Internet on, compte vérifié par téléphone), puis relancer.")
'''

DATA_SETUP = '''
# Données C1 : dataset privé atlas-c1-processed, cherché récursivement ; liens vers les dossiers de sujets.
tables = sorted(INPUT.rglob("subjects.parquet"))
if len(tables) != 1:
    raise FileNotFoundError(f"subjects.parquet trouvé {len(tables)} fois sous /kaggle/input (1 attendu) : "
                            "ajouter le dataset atlas-c1-processed (Add Input > Datasets > Shared with you).")
TABLE = tables[0]
DATA = SCRATCH / "processed"
DATA.mkdir(parents=True, exist_ok=True)
subjects = sorted(INPUT.rglob("qc.json"))
if not subjects:  # archives restées compressées : extraction hors des sorties
    for archive in sorted(INPUT.rglob("processed_partie*.zip")):
        zipfile.ZipFile(archive).extractall(SCRATCH / "c1_extrait")
    subjects = sorted((SCRATCH / "c1_extrait").rglob("qc.json"))
for qc in subjects:
    link = DATA / qc.parent.name
    if not link.exists():
        link.symlink_to(qc.parent, target_is_directory=True)
if not (DATA / "subjects.parquet").exists():
    (DATA / "subjects.parquet").symlink_to(TABLE)
print(len(subjects), "sujets prétraités ; table", TABLE)
if not subjects:
    raise FileNotFoundError("Aucun dossier de sujet (qc.json) dans atlas-c1-processed")
# Même découpage que le code : splits.json du dataset (à côté de subjects.parquet) identique à celui du dépôt.
data_splits = next((folder / "splits.json" for folder in [TABLE.parent, *TABLE.parents]
                    if folder != INPUT and INPUT in folder.parents and (folder / "splits.json").is_file()), None)
if data_splits is None:
    raise FileNotFoundError("splits.json absent du dataset atlas-c1-processed")
digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert digest(data_splits) == digest(REPO / "splits.json"), "splits.json du dataset différent de celui du code"
'''

TRAIN = '''
# Entraînement, validation sur le pli, puis évaluation sur le test ID du pli et le test OOD.
# Journal complet aussi dans /kaggle/working/runs/@@RUN@@/entrainement.log.
# Session interrompue : rattacher la sortie précédente, copier last.pt dans /kaggle/working/runs/@@RUN@@/ et ajouter "--resume".
def run_logged(command, log_path, slow_seconds=None, check_at=100):
    """Lance `command` depuis REPO, recopie sa sortie dans le notebook et dans `log_path`.

    Si la durée par itération dépasse `slow_seconds` à l'itération `check_at`, l'écrit clairement dans le
    journal, sans arrêter la run (le garde-fou de 10 h de train.py arrête l'entraînement plus tôt).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        def emit(text):
            print(text, end="", flush=True)
            log.write(text)
            log.flush()
        process = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        for line in process.stdout:
            emit(line)
            match = re.search(r"'progression': (\\d+),.*'s_par_iteration': ([\\d.]+)", line)
            if slow_seconds and match and int(match[1]) == check_at and float(match[2]) > slow_seconds:
                seconds = float(match[2])
                reached = int((10 * 3600 - 20 * 270) / seconds)  # budget de 10 h moins ~20 validations de 4 min 30
                emit(f"\\n{'!' * 100}\\nATTENTION : {seconds:.3f} s/it à l'itération {check_at}, au-dessus de "
                     f"{slow_seconds} s/it.\\nÀ ce rythme, 40 000 itérations et les validations ne tiennent pas dans "
                     f"le budget de 10 h : le garde-fou arrêtera l'entraînement vers l'itération {min(reached, 40000)}."
                     f"\\nLa run continue (aucun arrêt) ; à signaler dans le tableau RQ3.\\n{'!' * 100}\\n\\n")
        process.wait()
    if process.returncode:
        raise RuntimeError(f"{command[3]} terminé avec le code {process.returncode} : voir {log_path}")

RUN = WORKING / "runs" / "@@RUN@@"
run_logged([sys.executable, "-u", "-m", "adaptation.train", "--technique", "@@TECHNIQUE@@", "--fold", "@@FOLD@@",
            "--seed", "@@SEED@@", "--data-root", str(DATA), "--output", str(WORKING / "runs"), "--num-workers", "4",
            "--validate-every", "2000", "--log-every", "100"], RUN / "entrainement.log", slow_seconds=@@SLOW@@)
'''

TENT = '''
# TENT au test, par centre, sur le modèle baseline (stratégie « TENT seul »).
run_logged([sys.executable, "-u", "-m", "adaptation.train", "--evaluate-only", "--checkpoint", str(RUN / "best.pt"),
            "--tent", "--data-root", str(DATA)], RUN / "tent.log")
'''

SUMMARY = '''
# Résumé et fichiers à récupérer (onglet Output) ; aucune image de patient n'est écrite.
for path in sorted(RUN.glob("resume*.json")):
    print(path.name, json.dumps(json.loads(path.read_text()), indent=2, ensure_ascii=False))
for path in sorted(RUN.iterdir()):
    print(f"{path.stat().st_size / 1e6:10.1f} Mo  {path.name}")
'''


def setup_cells(commit=None):
    return [code(fill(ENVIRONMENT, COMMIT=repr(commit))), code(CODE_SETUP), code(DATA_SETUP)]


def notebook(name, technique, title, duration, commit=None):
    run = f"unet3d_{technique}_pli{FOLD}"
    cells = [
        markdown(f"""
# RQ3 — {title}

Run `{run}` (notebook Kaggle `{kernel_slug(technique)}`) du tableau d'ablation : UNet MONAI 3D en fp32 (clipping des
gradients à 12), patches de 96³, **pli {FOLD}** de `splits.json`, **graine {SEED}**, 40 000 itérations (lots de 4 patches),
identiques pour toutes les techniques. Durée estimée sur une T4 : {duration} (garde-fou à 10 h d'entraînement,
session limitée à 12 h). Si l'itération 100 dépasse {SLOW_SECONDS} s/it, un avertissement est écrit dans le journal.

**Avant de lancer** : accélérateur *GPU T4*, Internet activé (installation de MONAI et TorchIO), datasets privés
`{DATASET}` (données C1) et `{CODE_DATASET}` (code) ajoutés en entrée.
Notebook à garder **privé** (accord d'utilisation ATLAS). *Save Version → Save & Run All* pour une exécution en arrière-plan.
Toutes les sorties sont écrites dans `/kaggle/working/runs/{run}/`.
"""),
        *setup_cells(commit),
        code(fill(TRAIN, RUN=run, TECHNIQUE=technique, FOLD=FOLD, SEED=SEED, SLOW=SLOW_SECONDS)),
    ]
    if technique == "baseline":
        cells.append(code(TENT))
    cells.append(code(SUMMARY))
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


CONTINUAL_RUN = '''
# Checkpoint best.pt de la baseline fp32 (sortie du notebook @@BASELINE@@, attachée en entrée).
checkpoints = [path for path in INPUT.rglob("best.pt") if path.parent.name == "unet3d_baseline_pli@@FOLD@@"]
if not checkpoints:
    raise FileNotFoundError("best.pt de la baseline introuvable : le notebook @@BASELINE@@ doit être terminé et "
                            "attaché en entrée (Add Input > Notebooks).")
CHECKPOINT = checkpoints[0]
state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
assert all(torch.isfinite(value).all() for value in state["model"].values() if value.is_floating_point()), \\
    "best.pt contient des poids non finis"
print("Checkpoint :", CHECKPOINT, state["state"])
OUTPUT = WORKING / "adaptation_continue"
'''

CONTINUAL_MAIN = '''
# Flux des centres OOD, évaluation préquentielle, 4 stratégies, oubli (journal aussi dans adaptation_continue/).
started = time.time()
command = [sys.executable, "-u", "-m", "adaptation.continual", "--checkpoint", str(CHECKPOINT), "--data-root", str(DATA),
           "--output", str(OUTPUT), "--subjects-per-center", "24", "--lot-size", "4"]
OUTPUT.mkdir(parents=True, exist_ok=True)
with open(OUTPUT / "adaptation_continue.log", "a", encoding="utf-8") as log:
    process = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
    process.wait()
print(f"{(time.time() - started) / 60:.1f} min")
if process.returncode:
    raise RuntimeError(f"adaptation.continual terminé avec le code {process.returncode}")
for path in sorted(OUTPUT.iterdir()):
    print(f"{path.stat().st_size / 1e3:10.1f} ko  {path.name}")
'''


def continual_notebook(commit=None):
    baseline = kernel_slug("baseline")
    cells = [
        markdown(f"""
# RQ3 — Adaptation continue (version réduite), pli {FOLD}

`adaptation/continual.py` sur le `best.pt` de la baseline fp32 (sortie du notebook `{baseline}`, attachée en entrée) :
les 6 centres OOD présentés un par un (24 sujets par centre, lots de 4), évaluation préquentielle (chaque lot est noté
avant de servir à l'adaptation), 4 stratégies (modèle figé, auto-apprentissage naïf, TENT seul, adaptation contrôlée =
TENT + champion/challenger + replay buffer), Dice sur les anciens centres pour mesurer l'oubli, détection du décalage par
distance de Wasserstein entre histogrammes d'intensité (`center_distance()`, analysis/). Durée estimée sur une T4 : 30 à 50 min.

**Avant de lancer** : *GPU T4*, Internet activé, datasets `{DATASET}` et `{CODE_DATASET}`, notebook `{baseline}` terminé.
Sorties dans `/kaggle/working/adaptation_continue/`.
"""),
        *setup_cells(commit),
        code(fill(CONTINUAL_RUN, BASELINE=baseline, FOLD=FOLD)),
        code(CONTINUAL_MAIN),
    ]
    return {**notebook(*RUNS[0]), "cells": cells}


DIAG = ("06_diag_nan", "rq3-diag-nan")


def diag_notebook(commit=None):
    """Diagnostic des NaN de la baseline (run terminée) : adaptation/diag_nan.py, variantes A à D."""
    cells = [
        markdown(f"""
# RQ3 — Diagnostic des NaN de la baseline (pli {FOLD}, graine {SEED})

`adaptation/diag_nan.py` : même réseau, mêmes données, même pli et même graine que la baseline, 1 000 itérations
au plus par variante, journal par itération. A = baseline (fp16, Dice par patch, sans clipping ni warmup),
B = fp32, C = Dice sur le lot, D = unscale_ + clip_grad_norm_(12) + warmup sur 500 itérations.
Au premier poids ou moment non fini : lot, état avant le pas et gradients du pas dans `diag_nan/<variante>/rupture/`.
"""),
        *setup_cells(commit),
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


def kernel_metadata(name, slug, kernel_sources=()):
    """Métadonnées de `kaggle kernels push` : notebook privé, GPU, Internet, les deux datasets privés."""
    return {"id": f"{USERNAME}/{slug}", "title": slug, "code_file": f"{name}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [f"{USERNAME}/{DATASET}", f"{USERNAME}/{CODE_DATASET}"],
            "competition_sources": [], "kernel_sources": [f"{USERNAME}/{slug}" for slug in kernel_sources]}


HANDOFF_README = """# Runs RQ3 à lancer depuis ton compte Kaggle (@@NAME@@)

Notebooks : @@FILES@@. Chacun est autonome ; compter 6 h 30 à 9 h par run, toutes les sorties vont dans /kaggle/working/.
Tu peux lancer les deux en même temps (2 sessions GPU en arrière-plan par compte).

Pour chaque notebook :
1. kaggle.com > Code > New Notebook, puis File > Import Notebook > choisir le fichier .ipynb.
2. Titre du notebook = nom du fichier sans .ipynb (ex. rq3-fda-pli0). Garder le notebook **privé**.
3. Panneau de droite > Input > Add Input > Datasets > onglet « Shared with you » : ajouter
   `atlas-c1-processed` puis `projet-avc-code` (fayfahachicha).
4. Settings : Accelerator = **GPU T4 x2** ; Internet = **On** (il faut un compte vérifié par téléphone).
5. Save Version > **Save & Run All (Commit)** > Save. La run tourne en arrière-plan ; on peut fermer l'onglet.
6. Dans les 5 premières minutes, ouvrir la version en cours (Logs) : on doit voir le GPU (Tesla T4), le commit du code
   et « sujets prétraités ». En cas d'erreur, le message dit quoi corriger (GPU, Internet, dataset, version du code).
7. À la fin (onglet Output de la version), tout télécharger (Download) et envoyer le .zip tel quel.
"""


def write_notebook(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(content, handle, ensure_ascii=False, indent=1)
        handle.write("\n")


def write_kernel(folder, name, content, metadata):
    write_notebook(folder / f"{name}.ipynb", content)
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kernels-dir", type=Path, help="dossiers kaggle kernels push (runs du compte propriétaire)")
    parser.add_argument("--handoff-dir", type=Path, help="notebooks à importer à la main, un dossier par membre")
    args = parser.parse_args()
    for name, technique, title, duration in RUNS:
        write_notebook(HERE / f"{name}.ipynb", notebook(name, technique, title, duration))
    write_notebook(HERE / f"{CONTINUAL[0]}.ipynb", continual_notebook())
    write_notebook(HERE / f"{DIAG[0]}.ipynb", diag_notebook())
    print("Notebooks de référence :", HERE)
    if not (args.kernels_dir or args.handoff_dir):
        return
    commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    by_technique = {technique: (name, title, duration) for name, technique, title, duration in RUNS}
    if args.kernels_dir:
        for technique in ASSIGNMENT[USERNAME]:
            if technique == "baseline":
                continue  # déjà poussée (rq3-baseline-fp32-pli0, version 1) ; dossier 00_baseline inchangé
            name, title, duration = by_technique[technique]
            write_kernel(args.kernels_dir / name, name, notebook(name, technique, title, duration, commit),
                         kernel_metadata(name, kernel_slug(technique)))
            print(args.kernels_dir / name, kernel_slug(technique))
        name, slug = CONTINUAL
        write_kernel(args.kernels_dir / name, name, continual_notebook(commit),
                     kernel_metadata(name, slug, kernel_sources=[kernel_slug("baseline")]))
        print(args.kernels_dir / name, slug)
    if args.handoff_dir:
        for member, techniques in ASSIGNMENT.items():
            if member == USERNAME:
                continue
            files = []
            for technique in techniques:
                name, title, duration = by_technique[technique]
                path = args.handoff_dir / member / f"{kernel_slug(technique)}.ipynb"
                write_notebook(path, notebook(name, technique, title, duration, commit))
                files.append(path.name)
                print(path)
            (args.handoff_dir / member / "LISEZMOI.md").write_text(
                fill(HANDOFF_README, NAME=member.capitalize(), FILES=", ".join(files)), encoding="utf-8")
    print("Commit attendu dans VERSION.json :", commit)


if __name__ == "__main__":
    main()
