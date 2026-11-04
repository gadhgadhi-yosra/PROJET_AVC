"""Prépare le dataset Kaggle privé `projet-avc-code` : le code commité, sans données ni sorties.

    .venv-dl\\Scripts\\python adaptation\\kaggle\\build_code_dataset.py
    cd data\\kaggle_projet_avc_code ; kaggle datasets create -p . -r skip          # première fois
    cd data\\kaggle_projet_avc_code ; kaggle datasets version -p . -r skip -m "…"  # mises à jour

`projet_avc_code.zip` contient, depuis `git archive HEAD` (donc seulement des fichiers suivis par Git,
jamais `generalization/figures/`, `adaptation/runs/` ni `data/`) :
- `generalization/` et `adaptation/` ;
- `baselines/common.py` (métriques ISLES importées par adaptation/train.py) ;
- `splits.json` (découpages, exigés par les configurations) ;
- `VERSION.json` : branche, commit et date, affichés par les notebooks.
Refuse de construire si `generalization/` ou `adaptation/` ont des modifications non commitées.
"""
import argparse
import io
import json
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PATHS = ["generalization", "adaptation", "baselines/common.py", "splits.json"]
FORBIDDEN = (".nii", ".nii.gz", ".pt", ".pth", ".ckpt", ".npy", ".npz", ".png", ".jpg")


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=REPO / "data" / "kaggle_projet_avc_code")
    parser.add_argument("--username", default="fayfahachicha")
    args = parser.parse_args()

    if git("status", "--porcelain", "--", "generalization", "adaptation").strip():
        raise SystemExit("Modifications non commitées dans generalization/ ou adaptation/ : commiter d'abord")
    archive = zipfile.ZipFile(io.BytesIO(git("archive", "--format=zip", "HEAD", *PATHS)))
    names = [name for name in archive.namelist() if not name.endswith("/")]
    forbidden = [name for name in names if name.lower().endswith(FORBIDDEN)]
    if forbidden:
        raise SystemExit(f"Fichiers interdits dans l'archive : {forbidden[:5]}")
    version = dict(branche=git("rev-parse", "--abbrev-ref", "HEAD").decode().strip(),
                   commit=git("rev-parse", "--short", "HEAD").decode().strip(),
                   message=git("log", "-1", "--format=%s").decode().strip(),
                   date=datetime.now().isoformat(timespec="seconds"), fichiers=len(names))
    args.output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output / "projet_avc_code.zip", "w", zipfile.ZIP_DEFLATED) as out:
        for name in names:
            out.writestr(name, archive.read(name))
        out.writestr("VERSION.json", json.dumps(version, indent=2, ensure_ascii=False) + "\n")
    metadata = {"title": "Projet AVC - code (privé)", "id": f"{args.username}/projet-avc-code",
                "subtitle": "generalization/ et adaptation/ du projet AVC, sans données ni sorties",
                "description": "Code du projet, pour les notebooks Kaggle privés de l'équipe. Aucune donnée patient.",
                "licenses": [{"name": "other"}]}
    (args.output / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                                                      encoding="utf-8")
    size = (args.output / "projet_avc_code.zip").stat().st_size
    print(json.dumps(version, ensure_ascii=False), f"{size / 1e3:.0f} ko")


if __name__ == "__main__":
    main()
