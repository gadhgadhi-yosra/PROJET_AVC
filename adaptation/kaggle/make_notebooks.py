"""Génère les notebooks Kaggle et Google Colab du tableau d'ablation RQ3 (un par technique, mêmes pli et graine).

    .venv-dl\\Scripts\\python adaptation\\kaggle\\make_notebooks.py
    .venv-dl\\Scripts\\python adaptation\\kaggle\\make_notebooks.py --kernels-dir data\\kaggle_kernels --colab-dir data\\colab

- `adaptation/kaggle/*.ipynb` : copies de référence (Kaggle), sans contrôle de commit ;
- `--kernels-dir` : un dossier par notebook Kaggle (runs de ASSIGNMENT["kaggle"] et adaptation continue), avec
  kernel-metadata.json pour `kaggle kernels push` ;
- `--colab-dir` : `<slug>.ipynb` pour Google Colab (runs de ASSIGNMENT["colab"] et adaptation continue) :
  datasets téléchargés par l'API Kaggle (jeton dans les secrets Colab), sorties et checkpoints sur Google Drive
  (MyDrive/PROJET_AVC/<slug>/), reprise automatique depuis last.pt (`--resume`, même trajectoire).
Ces copies contrôlent que VERSION.json du dataset de code porte le commit HEAD.

Même réglage partout : code du dataset privé `projet-avc-code` (voir build_code_dataset.py), données du dataset
privé `atlas-c1-processed`, cherchés récursivement sous le dossier d'entrée ; fp32, pli 0, graine 2026, 4 workers.
"""
import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
USERNAME = "fayfahachicha"
DATASET = "atlas-c1-processed"
CODE_DATASET = "projet-avc-code"
FOLD, SEED = 0, 2026
SLOW_SECONDS = 0.75  # s/it à l'itération 100 au-delà desquelles 40 000 itérations + validations dépassent 10 h
REFERENCE = dict(torch="2.11.0+cu130", gpu="Tesla T4")  # environnement Kaggle de la baseline
RUNS = [  # fichier, technique, titre, durée estimée sur une T4 Kaggle (entraînement + validation + évaluation)
    ("00_baseline", "baseline", "Baseline (aucune technique de généralisation), puis TENT au test", "6 h 30 à 7 h 30"),
    ("01_intensite", "intensity", "Augmentation d'intensité (gamma, contraste, bruit, flou, biais, basse résolution)", "6 h 30 à 7 h"),
    ("02_bigaug", "bigaug", "Preset BigAug", "6 h 30 à 7 h"),
    ("03_fda", "fda", "FDA (échange d'amplitude basse fréquence)", "7 à 9 h"),
    ("04_histogram_matching", "histogram_matching", "Histogram matching (aussi appliqué à la validation et au test)", "7 à 7 h 30"),
    ("05_dann", "dann", "DANN (classifieur de centre et inversion de gradient)", "6 h 30 à 7 h"),
]
COLAB_DURATIONS = {"dann": "6 h 30 à 8 h (0,40 s/it mesuré sur Colab)", "intensity": "10 h 30 à 11 h 30 (au-delà du garde-fou de 10 h)",
                   "bigaug": "plus de 11 h (au-delà du garde-fou de 10 h)", "histogram_matching": "10 à 11 h"}
# Répartition : Kaggle (compte du propriétaire, 2 sessions GPU) et Google Colab (1 session, reprise sur Drive).
# Colab (2 processeurs) mesuré à 0,80-0,87 s/it pour l'intensité : DANN, la moins coûteuse à préparer, y passe.
# BigAug passée sur Kaggle (plus de GPU Colab) : 6 h 30 à 7 h 30 estimées, dans le quota restant.
ASSIGNMENT = {"kaggle": ["baseline", "fda", "intensity", "bigaug"], "colab": ["dann", "histogram_matching"]}
# Baseline fp32 et DANN v2 (classifieur normalisé, poids 0,1) : slugs propres, jamais les sorties des anciennes runs.
SLUGS = {"baseline": f"rq3-baseline-fp32-pli{FOLD}", "dann": f"rq3-dann-v2-pli{FOLD}"}
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


# --- Cellules propres à Google Colab ---

COLAB_KEEPALIVE = '''
# Session active : écran maintenu allumé (Screen Wake Lock du navigateur) tant que cet onglet est visible.
# Rien n'est cliqué automatiquement dans Colab. Laisser l'onglet ouvert, l'ordinateur branché et sans mise en veille.
from google.colab.output import eval_js
status = eval_js("""
(async () => {
  if (!("wakeLock" in navigator)) return "API Wake Lock absente de ce navigateur";
  const lock = async () => { window.rq3WakeLock = await navigator.wakeLock.request("screen"); };
  try { await lock(); } catch (error) { return "refusé : " + error.message; }
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") lock().catch(() => {}); });
  return "actif";
})()
""")
print("Maintien de l'écran :", status)
if status != "actif":
    print("Régler la mise en veille de l'ordinateur sur « Jamais » (secteur) pendant la run.")
'''

COLAB_SETUP = '''
# Google Drive (sorties et checkpoints) et jeton Kaggle lu dans les secrets Colab (jamais écrit dans le notebook).
import os, shutil, subprocess, sys
from pathlib import Path
from google.colab import drive, userdata
DRIVE = Path(os.environ.get("RQ3_DRIVE", "/content/drive"))
drive.mount(str(DRIVE))
SLUG = "@@SLUG@@"
DRIVE_RUN = DRIVE / "MyDrive" / "PROJET_AVC" / SLUG
DRIVE_RUN.mkdir(parents=True, exist_ok=True)

def secret(name):
    try:
        return userdata.get(name)
    except Exception:  # secret absent ou accès non accordé à ce notebook
        return None

if secret("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = secret("KAGGLE_API_TOKEN")
elif secret("KAGGLE_USERNAME") and secret("KAGGLE_KEY"):
    os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = secret("KAGGLE_USERNAME"), secret("KAGGLE_KEY")
else:
    raise RuntimeError("Jeton Kaggle absent : panneau Secrets (icône clé) > ajouter KAGGLE_API_TOKEN, ou "
                       "KAGGLE_USERNAME et KAGGLE_KEY, avec « Accès au notebook » activé, puis relancer.")
for key, value in dict(RQ3_INPUT="/content/entrees", RQ3_WORKING="/content/travail", RQ3_TMP="/content/tmp").items():
    os.environ.setdefault(key, value)  # disque local de la machine Colab
os.environ.update(RQ3_PLATFORM="colab", RQ3_RUNS=str(DRIVE_RUN / "runs"))
print("Sorties :", DRIVE_RUN)
'''

COLAB_DOWNLOAD = '''
# Datasets privés téléchargés par l'API Kaggle sur le disque de la machine Colab (pas sur Drive).
if os.environ.get("RQ3_PIP", "1") == "1":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle==2.2.5", "six"], check=True)
for dataset in ("@@USER@@/@@CODE_DATASET@@", "@@USER@@/@@DATASET@@"):
    target = Path(os.environ["RQ3_INPUT"]) / dataset.split("/")[1]
    if (target / ".complet").exists():
        continue
    shutil.rmtree(target, ignore_errors=True)
    download = subprocess.run(["kaggle", "datasets", "download", dataset, "-p", str(target), "--unzip"])
    if download.returncode:
        raise RuntimeError(f"Téléchargement de {dataset} impossible : vérifier le jeton Kaggle (secrets Colab).")
    (target / ".complet").touch()
print(f"{shutil.disk_usage(os.environ['RQ3_INPUT']).free / 1e9:.0f} Go libres sur le disque local")
'''

COLAB_BASELINE = '''
# best.pt de la baseline : sortie du notebook Kaggle @@BASELINE@@ (terminé), téléchargée par l'API Kaggle.
target = Path(os.environ["RQ3_INPUT"]) / "baseline"
if not any(target.rglob("best.pt")):
    download = subprocess.run(["kaggle", "kernels", "output", "@@USER@@/@@BASELINE@@", "-p", str(target),
                               "--file-pattern", r"unet3d_baseline_pli@@FOLD@@/(best\\.pt|resume\\.json)$"])
    if download.returncode:
        raise RuntimeError("Sortie de @@BASELINE@@ indisponible : la baseline doit être terminée sur Kaggle.")
'''

# --- Cellules communes (Kaggle et Colab) ---

ENVIRONMENT = '''
# Environnement : GPU obligatoire, sinon arrêt immédiat avec un message clair ; écarts à Kaggle signalés.
import hashlib, json, os, re, shutil, subprocess, sys, time, zipfile
from pathlib import Path
import torch
PLATFORM = os.environ.get("RQ3_PLATFORM", "kaggle")
GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "aucun"
print("Plateforme", PLATFORM, "| torch", torch.__version__, "| CUDA", torch.version.cuda, "| GPU", GPU,
      "| Python", sys.version.split()[0])
if not torch.cuda.is_available() and os.environ.get("RQ3_GPU", "1") == "1":
    where = ("Exécution > Modifier le type d'exécution > GPU T4" if PLATFORM == "colab"
             else "Settings > Accelerator > « GPU T4 x2 »")
    raise RuntimeError(f"AUCUN GPU DISPONIBLE : {where}, puis relancer. Rien n'a été lancé.")
INPUT = Path(os.environ.get("RQ3_INPUT", "/kaggle/input"))      # datasets
WORKING = Path(os.environ.get("RQ3_WORKING", "/kaggle/working"))  # code
RUNS_DIR = Path(os.environ.get("RQ3_RUNS", str(WORKING / "runs")))  # toutes les sorties des runs
SCRATCH = Path(os.environ.get("RQ3_TMP", "/tmp/rq3"))            # liens vers les données, jamais dans les sorties
EXPECTED_COMMIT = @@COMMIT@@  # commit(s) accepté(s) dans VERSION.json, par ordre de préférence (None : pas de contrôle)
REFERENCE = @@REFERENCE@@  # environnement Kaggle de la baseline
differences = []
if REFERENCE["gpu"] not in GPU:
    differences.append(f"GPU {GPU} au lieu de {REFERENCE['gpu']} : durées et arrondis différents de Kaggle")
if torch.__version__ != REFERENCE["torch"]:
    differences.append(f"torch {torch.__version__} au lieu de {REFERENCE['torch']} (Kaggle)")
for difference in differences:
    print("AVERTISSEMENT (la run continue) :", difference)
RUNS_DIR.mkdir(parents=True, exist_ok=True)
(RUNS_DIR / f"environnement_{PLATFORM}_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(dict(
    plateforme=PLATFORM, torch=torch.__version__, cuda=torch.version.cuda, gpu=GPU, python=sys.version.split()[0],
    reference=REFERENCE, avertissements=differences), indent=2, ensure_ascii=False))
'''

CODE_SETUP = '''
# Code du projet : dataset privé projet-avc-code, cherché récursivement dans le dossier d'entrée.
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
    raise FileNotFoundError("Dataset projet-avc-code introuvable : l'ajouter en entrée, puis relancer.")
accepted = None if EXPECTED_COMMIT is None else (
    (EXPECTED_COMMIT,) if isinstance(EXPECTED_COMMIT, str) else tuple(EXPECTED_COMMIT))
matching = sorted(((path, version) for path, version in found if accepted is None or version["commit"] in accepted),
                  key=lambda item: accepted.index(item[1]["commit"]) if accepted else 0)
if not matching:
    raise RuntimeError(f"VERSION.json : commit {sorted({v['commit'] for _, v in found})} au lieu de {EXPECTED_COMMIT}. "
                       "Mettre l'entrée projet-avc-code à sa dernière version, puis relancer.")
source, VERSION = matching[0]
print("Code :", source, VERSION)
REPO = WORKING / "PROJET_AVC"
if not REPO.exists():
    if source.suffix == ".zip":
        zipfile.ZipFile(source).extractall(REPO)
    else:  # archive décompressée : copie de l'arborescence
        shutil.copytree(source.parent, REPO)
if os.environ.get("RQ3_PIP", "1") == "1":
    pip = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "monai==1.6.1", "torchio==1.2.1",
                          "nibabel==5.4.2", "pyarrow"])
    if pip.returncode:
        raise RuntimeError("Installation de MONAI / TorchIO impossible : activer Internet, puis relancer.")
'''

DATA_SETUP = '''
# Données C1 : dataset privé atlas-c1-processed, cherché récursivement ; liens vers les dossiers de sujets.
tables = sorted(INPUT.rglob("subjects.parquet"))
if len(tables) != 1:
    raise FileNotFoundError(f"subjects.parquet trouvé {len(tables)} fois (1 attendu) : "
                            "ajouter le dataset atlas-c1-processed en entrée.")
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
# Journal complet aussi dans runs/@@RUN@@/entrainement.log.@@RESUME_NOTE@@
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

RUN = RUNS_DIR / "@@RUN@@"
run_logged([sys.executable, "-u", "-m", "adaptation.train", "--technique", "@@TECHNIQUE@@", "--fold", "@@FOLD@@",
            "--seed", "@@SEED@@", "--data-root", str(DATA), "--output", str(RUNS_DIR), "--num-workers", "4",
            "--validate-every", "2000", "--log-every", "100"@@EXTRA@@], RUN / "entrainement.log", slow_seconds=@@SLOW@@)
'''

TENT = '''
# TENT au test, par centre, sur le modèle baseline (stratégie « TENT seul »).
run_logged([sys.executable, "-u", "-m", "adaptation.train", "--evaluate-only", "--checkpoint", str(RUN / "best.pt"),
            "--tent", "--data-root", str(DATA)], RUN / "tent.log")
'''

SUMMARY = '''
# Résumé ; résultats légers (CSV, JSON, journaux, sans checkpoint) aussi dans resultats_@@RUN@@.zip.
for path in sorted(RUN.glob("resume*.json")):
    print(path.name, json.dumps(json.loads(path.read_text()), indent=2, ensure_ascii=False))
with zipfile.ZipFile(RUNS_DIR / "resultats_@@RUN@@.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(RUN.iterdir()):
        if path.suffix in (".csv", ".json", ".log"):
            archive.write(path, f"runs/@@RUN@@/{path.name}")
for path in sorted(RUN.iterdir()):
    print(f"{path.stat().st_size / 1e6:10.1f} Mo  {path.name}")
print(f"Total : {sum(p.stat().st_size for p in RUNS_DIR.rglob('*') if p.is_file()) / 1e6:.0f} Mo dans {RUNS_DIR}")
'''


def setup_cells(commit=None):
    return [code(fill(ENVIRONMENT, COMMIT=repr(commit), REFERENCE=repr(REFERENCE))), code(CODE_SETUP), code(DATA_SETUP)]


def colab_cells(slug):
    return [code(COLAB_KEEPALIVE), code(fill(COLAB_SETUP, SLUG=slug)),
            code(fill(COLAB_DOWNLOAD, USER=USERNAME, CODE_DATASET=CODE_DATASET, DATASET=DATASET))]


def notebook_metadata(colab=False):
    metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}}
    if colab:
        metadata["accelerator"] = "GPU"
        metadata["colab"] = {"gpuType": "T4", "provenance": []}
    return metadata


def notebook(name, technique, title, duration, commit=None, colab=False):
    run = f"unet3d_{technique}_pli{FOLD}"
    slug = kernel_slug(technique)
    if colab:
        platform = f"""**Google Colab** (exécution > type d'exécution : *GPU T4*). Jeton Kaggle dans les secrets Colab
(`KAGGLE_API_TOKEN`, ou `KAGGLE_USERNAME` et `KAGGLE_KEY`). Sorties et checkpoints dans `MyDrive/PROJET_AVC/{slug}/` ;
`last.pt` toutes les 20 min. **Après une coupure, relancer « Tout exécuter »** : l'entraînement reprend au dernier
checkpoint, avec les mêmes lots et les mêmes tirages aléatoires que sans coupure. Durée estimée (Colab, 2 processeurs) :
{COLAB_DURATIONS.get(technique, duration)}, en une ou plusieurs sessions."""
    else:
        platform = f"""**Avant de lancer** : accélérateur *GPU T4*, Internet activé (installation de MONAI et TorchIO), datasets privés
`{DATASET}` (données C1) et `{CODE_DATASET}` (code) ajoutés en entrée. Notebook à garder **privé** (accord d'utilisation
ATLAS). *Save Version → Save & Run All*. Durée estimée sur une T4 Kaggle : {duration}. Sorties dans `/kaggle/working/runs/`."""
    cells = [
        markdown(f"""
# RQ3 — {title}

Run `{run}` (`{slug}`) du tableau d'ablation : UNet MONAI 3D en fp32 (clipping des gradients à 12), patches de 96³,
**pli {FOLD}** de `splits.json`, **graine {SEED}**, 40 000 itérations (lots de 4 patches), 4 workers, identiques pour toutes
les techniques et sur les deux plateformes (garde-fou à 10 h d'entraînement). Si l'itération 100 dépasse {SLOW_SECONDS} s/it,
un avertissement est écrit dans le journal, sans arrêter la run.

{platform}
"""),
        *(colab_cells(slug) if colab else []),
        *setup_cells(commit),
        code(fill(TRAIN, RUN=run, TECHNIQUE=technique, FOLD=FOLD, SEED=SEED, SLOW=SLOW_SECONDS,
                  EXTRA=', "--resume"' if colab else "",
                  RESUME_NOTE=("\n# Reprise automatique : --resume repart de last.pt sur Drive s'il existe." if colab else
                               "\n# Session interrompue : rattacher la sortie précédente, copier last.pt dans "
                               f"/kaggle/working/runs/{run}/ et ajouter \"--resume\"."))),
    ]
    if technique == "baseline":
        cells.append(code(TENT))
    cells.append(code(fill(SUMMARY, RUN=run)))
    return {"cells": cells, "metadata": notebook_metadata(colab), "nbformat": 4, "nbformat_minor": 5}


CONTINUAL_RUN = '''
# Checkpoint best.pt de la baseline fp32 (sortie du notebook @@BASELINE@@).
checkpoints = [path for path in INPUT.rglob("best.pt") if path.parent.name == "unet3d_baseline_pli@@FOLD@@"]
if not checkpoints:
    raise FileNotFoundError("best.pt de la baseline introuvable : le notebook @@BASELINE@@ doit être terminé et "
                            "attaché en entrée (Kaggle : Add Input > Notebooks).")
CHECKPOINT = checkpoints[0]
state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
assert all(torch.isfinite(value).all() for value in state["model"].values() if value.is_floating_point()), \\
    "best.pt contient des poids non finis"
print("Checkpoint :", CHECKPOINT, state["state"])
OUTPUT = RUNS_DIR.parent / "adaptation_continue"
'''

CONTINUAL_MAIN = '''
# Flux des centres OOD, évaluation préquentielle, 4 stratégies, oubli (journal aussi dans adaptation_continue/).
# Courte (30 à 50 min) : après une coupure, elle recommence du début (sorties réécrites).
started = time.time()
command = [sys.executable, "-u", "-m", "adaptation.continual", "--checkpoint", str(CHECKPOINT), "--data-root", str(DATA),
           "--output", str(OUTPUT), "--subjects-per-center", "24", "--lot-size", "4"]
OUTPUT.mkdir(parents=True, exist_ok=True)
with open(OUTPUT / "adaptation_continue.log", "w", encoding="utf-8") as log:
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


def continual_notebook(commit=None, colab=False):
    baseline = kernel_slug("baseline")
    name, slug = CONTINUAL
    where = (f"**Google Colab** (GPU T4, jeton Kaggle dans les secrets) : datasets et `best.pt` téléchargés par l'API "
             f"Kaggle, sorties dans `MyDrive/PROJET_AVC/{slug}/adaptation_continue/`." if colab else
             f"**Kaggle** : *GPU T4*, Internet activé, datasets `{DATASET}` et `{CODE_DATASET}`, sortie du notebook "
             f"`{baseline}` en entrée. Sorties dans `/kaggle/working/adaptation_continue/`.")
    cells = [
        markdown(f"""
# RQ3 — Adaptation continue (version réduite), pli {FOLD}

`adaptation/continual.py` sur le `best.pt` de la baseline fp32 (notebook `{baseline}`, terminé) : les 6 centres OOD
présentés un par un (24 sujets par centre, lots de 4), évaluation préquentielle (chaque lot est noté avant de servir à
l'adaptation), 4 stratégies (modèle figé, auto-apprentissage naïf, TENT seul, adaptation contrôlée = TENT +
champion/challenger + replay buffer), Dice sur les anciens centres pour mesurer l'oubli, détection du décalage par distance
de Wasserstein entre histogrammes d'intensité (`center_distance()`, analysis/). Durée estimée sur une T4 : 30 à 50 min.

{where}
"""),
        *(colab_cells(slug) if colab else []),
        *([code(fill(COLAB_BASELINE, USER=USERNAME, BASELINE=baseline, FOLD=FOLD))] if colab else []),
        *setup_cells(commit),
        code(fill(CONTINUAL_RUN, BASELINE=baseline, FOLD=FOLD)),
        code(CONTINUAL_MAIN),
    ]
    return {"cells": cells, "metadata": notebook_metadata(colab), "nbformat": 4, "nbformat_minor": 5}


# --- Reprise sur Kaggle d'une run interrompue ailleurs (DANN v2 coupée sur Colab) ---

RESUME = dict(name="08_dann_reprise", slug=f"rq3-dann-v2-reprise-pli{FOLD}", technique="dann",
              checkpoints=f"rq3-dann-v2-checkpoints-pli{FOLD}", commit="7449c49", code_version=8)

RESUME_SETUP = '''
# Checkpoints de la run interrompue (dataset privé @@CHECKPOINTS@@) copiés dans le dossier de la run ; contrôles
# avant de relancer : fichiers intacts, configuration identique, LR et lambda déterminés par l'itération.
import ast, math
sys.path.insert(0, str(REPO))
from adaptation.dann import ganin_lambda
RUN = RUNS_DIR / "@@RUN@@"
found = [path.parent for path in INPUT.rglob("last.pt") if (path.parent / "config.json").is_file()
         and json.loads((path.parent / "config.json").read_text()).get("technique") == "@@TECHNIQUE@@"]
if len(found) != 1:
    raise FileNotFoundError(f"last.pt de la run @@TECHNIQUE@@ trouvé {len(found)} fois (1 attendu) : "
                            "ajouter le dataset @@CHECKPOINTS@@ en entrée.")
SOURCE = found[0]
sums = SOURCE / "sha256.txt"
if sums.is_file():
    for line in sums.read_text().split("\\n"):
        if line.strip():
            expected, name = line.split()
            assert digest(SOURCE / name) == expected, f"{name} altéré pendant le transfert (sha256 différent)"
    print("sha256 vérifiés :", sums.read_text().split())
RUN.mkdir(parents=True, exist_ok=True)
for name in ("last.pt", "best.pt"):
    if not (RUN / name).exists():
        shutil.copy2(SOURCE / name, RUN / name)
last = torch.load(RUN / "last.pt", map_location="cpu", weights_only=False)
best = torch.load(RUN / "best.pt", map_location="cpu", weights_only=False)
for label, checkpoint in (("last.pt", last), ("best.pt", best)):
    assert all(torch.isfinite(v).all() for v in checkpoint["model"].values() if v.is_floating_point()), \\
        f"{label} contient des poids non finis"
STATE, SAVED = last["state"], last["reprise"] or {}
CONFIG = json.loads(json.dumps(last["config"], default=str))  # tuples en listes, comme config.json
IT, MAX = STATE["iteration"], CONFIG["max_iterations"]
BUDGET = CONFIG["time_budget_hours"] * 3600
print("last.pt :", {k: STATE[k] for k in ("iteration", "elapsed", "best_dice", "best_iteration")})
print("best.pt :", {k: best["state"][k] for k in ("iteration", "best_dice", "best_iteration")})
print(f"Reste {MAX - IT} itérations ({IT} faites sur {MAX}) ; workers sauvegardés {SAVED.get('num_workers')}, "
      f"{len(SAVED.get('rng_lots', []))} états aléatoires, {SAVED.get('volumes_consommes')} volumes consommés")
# Configuration de la run : celle de last.pt, à l'identique (seuls les chemins changent).
WANTED = dict(technique="@@TECHNIQUE@@", fold=@@FOLD@@, seed=@@SEED@@, num_workers=4, max_iterations=40000,
              validate_every=2000, log_every=100, amp=False, grad_clip=12.0, dann_weight=0.1, dann_gamma=5.0,
              dann_start=0.05, volumes_per_batch=2, patches_per_volume=2, learning_rate=0.001)
mismatch = {k: (CONFIG.get(k), v) for k, v in WANTED.items() if CONFIG.get(k) != v}
assert not mismatch, f"last.pt ne vient pas de la run DANN v2 attendue : {mismatch}"
assert IT < MAX, "last.pt est déjà à la dernière itération : rien à reprendre"
assert SAVED.get("num_workers") == 4 and len(SAVED.get("rng_lots", [])) == 4, "états aléatoires des workers absents"
assert SAVED.get("volumes_consommes") == IT * CONFIG["volumes_per_batch"], "position du sampler incohérente"
# Meilleur modèle sur toute la run : best.pt d'avant la coupure, cohérent avec l'état de last.pt.
if best["state"]["iteration"] <= IT:
    assert best["state"]["best_iteration"] == STATE["best_iteration"], "best.pt ne correspond pas à last.pt"
    assert math.isclose(best["state"]["best_dice"], STATE["best_dice"]), "best.pt ne correspond pas à last.pt"
# Avancement : max(temps / budget, itération / max) ; il doit rester fixé par l'itération jusqu'à la fin.
lr_at = lambda done: CONFIG["learning_rate"] * (1 - done / MAX) ** 0.9  # noqa: E731
assert math.isclose(SAVED["lr"], lr_at(IT - 1), rel_tol=1e-9), \\
    f"LR sauvegardé {SAVED['lr']:.6e} au lieu de {lr_at(IT - 1):.6e} : avancement fixé par le temps, reprise inexacte"
hours_left = 2.0  # majorant large : ~4 000 itérations et 2 validations sur une T4
assert (STATE["elapsed"] + hours_left * 3600) / BUDGET < IT / MAX, "le budget de temps fixerait le LR à la reprise"
FIRST = (IT // 100 + 1) * 100  # première ligne de progression après la reprise
EXPECTED = dict(iteration=IT, premiere_ligne=FIRST, lr=f"{lr_at(FIRST - 1):.2e}",
                **{"lambda": round(ganin_lambda((FIRST - 1) / MAX, CONFIG["dann_gamma"], CONFIG["dann_start"]), 4)},
                sampler_start=IT * CONFIG["volumes_per_batch"])
print("Attendu à la reprise :", EXPECTED)
# Journal des validations : lignes postérieures à last.pt retirées (elles seront refaites) ; original conservé.
journal = SOURCE / "journal_entrainement.csv"
if journal.is_file() and not (RUN / "journal_entrainement_avant_reprise.csv").exists():
    shutil.copy2(journal, RUN / "journal_entrainement_avant_reprise.csv")
    lines = journal.read_text().splitlines()
    kept = [lines[0]] + [line for line in lines[1:] if line and int(line.split(",")[0]) <= IT]
    (RUN / "journal_entrainement.csv").write_text("\\n".join(kept) + "\\n")
    print(f"Journal : {len(kept) - 1} validations conservées, {len(lines) - len(kept)} refaites")
'''

RESUME_TRAIN = '''
# Reprise avec --resume jusqu'à 40 000 itérations, validations toutes les 2 000, puis évaluation ID et OOD.
# Arrêt immédiat si la reprise n'est pas exacte (itération, états aléatoires, configuration, LR ou lambda).
# Journal complet dans runs/@@RUN@@/entrainement.log.
checks = []
def stop(process, message):
    process.terminate()
    raise RuntimeError("REPRISE INEXACTE : " + message)

log_path = RUN / "entrainement.log"
command = [sys.executable, "-u", "-m", "adaptation.train", "--technique", "@@TECHNIQUE@@", "--fold", "@@FOLD@@",
           "--seed", "@@SEED@@", "--data-root", str(DATA), "--output", str(RUNS_DIR), "--num-workers", "4",
           "--validate-every", "2000", "--log-every", "100", "--resume"]
with open(log_path, "a", encoding="utf-8") as log:
    process = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               bufsize=1, encoding="utf-8", errors="replace",
                               env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
        if line.startswith("Reprise "):
            if f"itération {IT} " not in line or "états aléatoires restaurés" not in line:
                stop(process, line.strip())
            config = json.loads((RUN / "config.json").read_text())
            paths = ("data_root", "splits", "output")
            diff = {k: (CONFIG.get(k), v) for k, v in config.items() if k not in paths and CONFIG.get(k) != v}
            if diff:
                stop(process, f"configuration différente de last.pt : {diff}")
            checks.append(f"itération {IT}, sampler au volume {IT * 2}, 4 états de workers, configuration identique")
        elif line.startswith("{'progression'") and len(checks) == 1:
            first = ast.literal_eval(line.strip())
            got = dict(premiere_ligne=first["progression"], lr=first["lr"], **{"lambda": first["lambda"]})
            want = {k: EXPECTED[k] for k in got}
            if got != want:
                stop(process, f"première ligne {got} au lieu de {want}")
            checks.append(f"première ligne {got} = attendu")
            log.write(f"CONTRÔLE DE REPRISE : OK ({' ; '.join(checks)})\\n")
            print("CONTRÔLE DE REPRISE : OK", checks, flush=True)
    process.wait()
if process.returncode:
    raise RuntimeError(f"adaptation.train terminé avec le code {process.returncode} : voir {log_path}")
assert len(checks) == 2, "contrôle de reprise incomplet : voir le journal"
'''


def resume_notebook(commit):
    spec = RESUME
    run = f"unet3d_{spec['technique']}_pli{FOLD}"
    cells = [
        markdown(f"""
# RQ3 — DANN v2 : fin de l'entraînement après la coupure sur Colab

Run `{run}` commencée sur Google Colab (code `{spec['commit']}`), coupée par la limite GPU de Colab avant 40 000
itérations. Ce notebook reprend **exactement** depuis `last.pt` (`--resume` : itération, poids, optimiseur, position du
sampler et états aléatoires des 4 workers), avec la même configuration, jusqu'à 40 000 itérations et les validations
toutes les 2 000, puis évalue le meilleur modèle de **toute** la run (`best.pt` d'avant la coupure inclus) sur le test ID
du pli {FOLD} et le test OOD, comme les autres runs.

**Avant de lancer** : *GPU T4*, Internet activé, datasets privés `{DATASET}`, `{CODE_DATASET}` (version
{spec['code_version']} = `{spec['commit']}`) et `{spec['checkpoints']}` (last.pt, best.pt, journal, config) en entrée.
Durée estimée : 1 à 1 h 30. Avant l'entraînement, la reprise est contrôlée (fichiers, configuration, LR et lambda) ;
pendant, la run s'arrête aussitôt si l'itération, les états aléatoires, le LR ou lambda de la première ligne diffèrent
des valeurs attendues.
"""),
        *setup_cells(commit),
        code(fill(RESUME_SETUP, RUN=run, TECHNIQUE=spec["technique"], CHECKPOINTS=spec["checkpoints"], FOLD=FOLD,
                  SEED=SEED)),
        code(fill(RESUME_TRAIN, RUN=run, TECHNIQUE=spec["technique"], FOLD=FOLD, SEED=SEED)),
        code(fill(SUMMARY, RUN=run)),
    ]
    return {"cells": cells, "metadata": notebook_metadata(), "nbformat": 4, "nbformat_minor": 5}


def resume_metadata():
    metadata = kernel_metadata(RESUME["name"], RESUME["slug"], code_version=RESUME["code_version"])
    metadata["dataset_sources"].append(f"{USERNAME}/{RESUME['checkpoints']}")
    return metadata


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
    return {"cells": cells, "metadata": notebook_metadata(), "nbformat": 4, "nbformat_minor": 5}


def kernel_metadata(name, slug, kernel_sources=(), code_version=None):
    """Métadonnées de `kaggle kernels push` : notebook privé, GPU, Internet, les deux datasets privés.

    code_version : numéro de version du dataset de code à attacher (défaut : la dernière).
    """
    code_source = f"{USERNAME}/{CODE_DATASET}" + (f"/{code_version}" if code_version else "")
    return {"id": f"{USERNAME}/{slug}", "title": slug, "code_file": f"{name}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [f"{USERNAME}/{DATASET}", code_source],
            "competition_sources": [], "kernel_sources": [f"{USERNAME}/{slug}" for slug in kernel_sources]}


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
    parser.add_argument("--kernels-dir", type=Path, help="dossiers kaggle kernels push (runs Kaggle)")
    parser.add_argument("--colab-dir", type=Path, help="notebooks Google Colab (<slug>.ipynb)")
    parser.add_argument("--commit", help="commit(s) accepté(s) dans VERSION.json, séparés par des virgules, par ordre "
                                          "de préférence (défaut : HEAD)")
    parser.add_argument("--code-version", type=int, help="kernels Kaggle : version du dataset de code à attacher")
    parser.add_argument("--dann-resume", type=Path, help="dossier du kernel de reprise de DANN v2 (seul généré)")
    args = parser.parse_args()
    if args.dann_resume:
        write_kernel(args.dann_resume, RESUME["name"], resume_notebook(RESUME["commit"]), resume_metadata())
        print(args.dann_resume, RESUME["slug"], "commit attendu", RESUME["commit"])
        return
    for name, technique, title, duration in RUNS:
        write_notebook(HERE / f"{name}.ipynb", notebook(name, technique, title, duration))
    write_notebook(HERE / f"{CONTINUAL[0]}.ipynb", continual_notebook())
    write_notebook(HERE / f"{DIAG[0]}.ipynb", diag_notebook())
    print("Notebooks de référence :", HERE)
    if not (args.kernels_dir or args.colab_dir):
        return
    commit = args.commit or subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short",
                                            "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    commit = tuple(commit.split(",")) if "," in commit else commit
    by_technique = {technique: (name, title, duration) for name, technique, title, duration in RUNS}
    if args.kernels_dir:
        for technique in ASSIGNMENT["kaggle"]:
            if technique == "baseline":
                continue  # déjà poussée (rq3-baseline-fp32-pli0, version 1) ; dossier 00_baseline inchangé
            name, title, duration = by_technique[technique]
            write_kernel(args.kernels_dir / name, name, notebook(name, technique, title, duration, commit),
                         kernel_metadata(name, kernel_slug(technique), code_version=args.code_version))
            print(args.kernels_dir / name, kernel_slug(technique))
        name, slug = CONTINUAL
        write_kernel(args.kernels_dir / name, name, continual_notebook(commit),
                     kernel_metadata(name, slug, kernel_sources=[kernel_slug("baseline")], code_version=args.code_version))
        print(args.kernels_dir / name, slug)
    if args.colab_dir:
        for technique in ASSIGNMENT["colab"]:
            name, title, duration = by_technique[technique]
            path = args.colab_dir / f"{kernel_slug(technique)}.ipynb"
            write_notebook(path, notebook(name, technique, title, duration, commit, colab=True))
            print(path)
        path = args.colab_dir / f"{CONTINUAL[1]}.ipynb"
        write_notebook(path, continual_notebook(commit, colab=True))
        print(path)
    print("Commit attendu dans VERSION.json :", commit)


if __name__ == "__main__":
    main()
