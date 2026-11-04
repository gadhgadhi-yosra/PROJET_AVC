"""Prépare le dossier à téléverser comme dataset Kaggle PRIVÉ (données C1), sans rien envoyer.

    .venv-dl\\Scripts\\python adaptation\\kaggle\\build_dataset.py --username <identifiant-kaggle>

Écrit dans `data/kaggle_atlas_c1/` (exclu de Git) :
- `processed_partieN.zip` : dossiers de sujets `processed/<sujet>/{t1,mask,brain}.nii.gz, qc.json`,
  stockés sans recompression (les .nii.gz le sont déjà), chaque partie sous `--max-gb` ;
- `subjects.parquet`, `splits.json` ;
- `dataset-metadata.json` (dataset privé : ne jamais passer --public à kaggle).
Le téléversement se fait à la main, avec les commandes affichées à la fin.
"""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATASET = "atlas-c1-processed"


def subject_folders(processed):
    return sorted(path.parent for path in processed.glob("*/qc.json"))


def folder_size(folder):
    return sum(path.stat().st_size for path in folder.iterdir() if path.is_file())


def plan_parts(folders, max_bytes):
    """Répartit les dossiers de sujets en parties de moins de `max_bytes`, dans l'ordre alphabétique."""
    parts, current, size = [], [], 0
    for folder in folders:
        weight = folder_size(folder)
        if current and size + weight > max_bytes:
            parts.append(current)
            current, size = [], 0
        current.append(folder)
        size += weight
    if current:
        parts.append(current)
    return parts


def write_part(path, folders):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for folder in folders:
            for file in sorted(folder.iterdir()):
                if file.is_file():
                    archive.write(file, f"processed/{folder.name}/{file.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed", type=Path, default=REPO / "data" / "processed")
    parser.add_argument("--splits", type=Path, default=REPO / "splits.json")
    parser.add_argument("--output", type=Path, default=REPO / "data" / "kaggle_atlas_c1")
    parser.add_argument("--max-gb", type=float, default=2.0, help="taille maximale d'une partie (Go)")
    parser.add_argument("--username", default="<identifiant-kaggle>")
    args = parser.parse_args()

    folders = subject_folders(args.processed)
    if not folders or not (args.processed / "subjects.parquet").is_file():
        raise SystemExit(f"Données C1 introuvables dans {args.processed}")
    args.output.mkdir(parents=True, exist_ok=True)
    parts = plan_parts(folders, int(args.max_gb * 1e9))
    manifest = []
    for index, part in enumerate(parts, start=1):
        path = args.output / f"processed_partie{index}.zip"
        write_part(path, part)
        manifest.append(dict(fichier=path.name, sujets=len(part), premier=part[0].name, dernier=part[-1].name,
                             octets=path.stat().st_size))
        print(f"{path.name} : {len(part)} sujets, {path.stat().st_size / 1e9:.2f} Go")
    for name, source in [("subjects.parquet", args.processed / "subjects.parquet"), ("splits.json", args.splits)]:
        shutil.copy2(source, args.output / name)
    metadata = {
        "title": "ATLAS v3 - données prétraitées C1 (privé)",
        "id": f"{args.username}/{DATASET}",
        "subtitle": "Projet AVC : IRM T1 RAS 1 mm, z-score, recadrées (pipeline de Yosra)",
        "description": ("Données dérivées d'ATLAS v3, soumises à son accord d'utilisation : dataset privé, partagé "
                        "uniquement avec l'équipe du projet. Ne pas rendre public."),
        "licenses": [{"name": "other"}],
    }
    (args.output / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                                                      encoding="utf-8")
    digest = hashlib.sha256(args.splits.read_bytes()).hexdigest()[:16]
    (args.output / "manifeste.json").write_text(json.dumps(dict(parties=manifest, sujets=len(folders),
                                                               splits_sha256_16=digest), indent=2) + "\n",
                                               encoding="utf-8")
    total = sum(item["octets"] for item in manifest)
    print(f"\n{len(folders)} sujets, {len(parts)} parties, {total / 1e9:.2f} Go dans {args.output}")
    print("Téléversement (dataset privé, après accord) :")
    print(f"  kaggle datasets create -p {args.output} --dir-mode skip")


if __name__ == "__main__":
    main()
