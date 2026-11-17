"""Rapatrie les sorties légères d'un kernel Kaggle terminé, puis refait l'agrégation RQ3 et l'analyse des échecs.

    .venv-dl\\Scripts\\python -m adaptation.kaggle.rapatrier fayfahachicha/rq3-bigaug-pli0

1. Vérifie que le kernel est COMPLETE (sinon : arrêt, rien n'est téléchargé ni recalculé).
2. Télécharge les seuls .csv, .json, .log et .txt (jamais les .pt) dans data/kaggle_sorties/<slug>/.
3. Retire la copie du code (dossier PROJET_AVC/), garde son VERSION.json sous version_code.json.
4. Vérifie la présence de resultats_id_test.csv et resultats_ood_test.csv.
5. Relance `aggregate_rq3` (sortie adaptation/outputs/rq3, figures figures/rq3) et `analyse_echecs`
   sur toutes les runs RQ3 présentes dans data/kaggle_sorties (liste RUNS ci-dessous).
Une seule requête de statut, pas d'attente : à lancer quand le kernel est terminé.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SORTIES = REPO_ROOT / "data" / "kaggle_sorties"
# Runs du tableau RQ3, dans l'ordre (dossiers de data/kaggle_sorties ; les absents sont ignorés).
RUNS = ["rq3-baseline-fp32-pli0", "rq3-intensity-pli0", "rq3-bigaug-pli0", "rq3-fda-pli0",
        "rq3-histogram-matching-pli0", "rq3-dann-v2-reprise-pli0", "rq3-adaptation-continue-pli0"]
LIGHT = r"\.(csv|json|log|txt)$"


def kaggle_executable():
    found = shutil.which("kaggle") or shutil.which("kaggle.exe")
    fallback = Path.home() / ".local" / "bin" / "kaggle.exe"
    if found:
        return found
    if fallback.is_file():
        return str(fallback)
    raise SystemExit("CLI kaggle introuvable (pip install kaggle)")


def kaggle(*args):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run([kaggle_executable(), *args], capture_output=True, text=True, encoding="utf-8",
                            errors="replace", env=env)
    if result.returncode:
        raise SystemExit(f"kaggle {' '.join(args)} a échoué :\n{result.stdout}{result.stderr}")
    return result.stdout


def fetch(kernel, sorties=SORTIES):
    """Statut puis téléchargement des sorties légères ; renvoie le dossier local."""
    status = kaggle("kernels", "status", kernel)
    if "COMPLETE" not in status:
        raise SystemExit(f"{kernel} n'est pas terminé : {status.strip()} ; rien n'est téléchargé")
    destination = Path(sorties) / kernel.split("/")[-1]
    destination.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", kernel, "-p", str(destination), "--file-pattern", LIGHT)
    clean(destination)
    return destination


def clean(destination):
    """Copie du code retirée (seul son VERSION.json est gardé) ; aucun .pt ne doit rester."""
    code = destination / "PROJET_AVC"
    if (code / "VERSION.json").is_file():
        (code / "VERSION.json").replace(destination / "version_code.json")
    if code.is_dir():
        shutil.rmtree(code)
    heavy = [p for p in destination.rglob("*.pt")]
    if heavy:
        raise SystemExit(f"Fichiers .pt inattendus dans {destination} : {heavy}")
    results = {p.name for p in destination.rglob("resultats_*_test.csv")}
    missing = {"resultats_id_test.csv", "resultats_ood_test.csv"} - results
    if missing:
        raise SystemExit(f"Sorties incomplètes dans {destination} : {sorted(missing)} absents")
    version = destination / "version_code.json"
    return json.loads(version.read_text(encoding="utf-8")) if version.is_file() else None


def refresh(sorties=SORTIES, output=None, figures=None):
    """Agrégation RQ3 et analyse des échecs sur toutes les runs présentes."""
    from adaptation.aggregate_rq3 import aggregate, markdown_table
    from adaptation.analyse_echecs import analyse
    inputs = [Path(sorties) / name for name in RUNS if (Path(sorties) / name).is_dir()]
    output = Path(output or REPO_ROOT / "adaptation" / "outputs" / "rq3")
    figures = Path(figures or REPO_ROOT / "figures" / "rq3")
    result = aggregate(inputs, output, figures)
    failures = analyse(inputs, output / "echecs", figures, REPO_ROOT / "data" / "processed")
    return inputs, markdown_table(result["table"]), failures


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kernel", help="propriétaire/slug du kernel Kaggle terminé")
    parser.add_argument("--sans-telechargement", action="store_true", help="seulement l'agrégation et l'analyse")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not args.sans_telechargement:
        destination = fetch(args.kernel)
        print(f"Sorties légères dans {destination} (code : {clean(destination)})")
    inputs, table, _ = refresh()
    print("Runs agrégées : " + ", ".join(p.name for p in inputs))
    print(table)
    print("Tableaux : adaptation/outputs/rq3/ ; analyse des échecs : adaptation/outputs/rq3/echecs/echecs.md")


if __name__ == "__main__":
    main()
