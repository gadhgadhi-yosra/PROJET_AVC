"""Temps CPU de chaque transformation sur un volume synthétique de taille réaliste (1 mm).

    .venv-dl\\Scripts\\python -m generalization.benchmark
    .venv-dl\\Scripts\\python -m generalization.benchmark --source processed   # format du contrat C1

Avec `--source processed`, le volume est écrit au format prétraité (z-score int16, recadré) et lu par
`data_processed.load_processed_subject`, comme à l'entraînement sur les données C1. Avec en plus
`--data-root data\\processed`, les mesures portent sur deux vrais sujets d'entraînement C1 (centres
différents) au lieu du volume synthétique.
"""
import argparse
import statistics
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch

from .data_processed import list_processed, load_processed_subject
from .data_raw import list_subjects, load_subject
from .synthetic import FOV_REALISTIC_MM, synthetic_volume, write_processed_subject, write_subject
from .transforms import INTENSITY_NAMES, REGISTRY, build_transforms, fit_reference_quantiles

CONFIGS = Path(__file__).resolve().parent / "configs"
SITES = ("R009", "R001")


def seconds(transform, subject, seeds):
    times = []
    for seed in seeds:
        torch.manual_seed(seed)
        start = time.perf_counter()
        transform(subject)
        times.append(time.perf_counter() - start)
    return times


def synthetic_subjects(folder, source):
    """Deux sujets synthétiques de taille réaliste : celui qu'on transforme et une référence."""
    volumes = [synthetic_volume(fov_mm=FOV_REALISTIC_MM, seed=index) for index in range(len(SITES))]
    if source == "processed":
        subjects = []
        for site, volume in zip(SITES, volumes):
            out = write_processed_subject(folder, site, f"sub-{site.lower()}s001", *volume)
            record = dict(case_id=f"{site}_{out.name}_ses-1", subject=out.name, site=site, session="ses-1",
                          empty_mask=False, norm_mean=0.0, norm_std=1.0,
                          **{f"{name}_path": str(out / f"{name}.nii.gz") for name in ("t1", "mask", "brain")})
            subject = load_processed_subject(pd.Series(record))
            subject.load()
            subjects.append(subject)
        return subjects
    for site, volume in zip(SITES, volumes):
        write_subject(folder, site, f"sub-{site.lower()}s001", *volume, metadata={})
    return [load_subject(record) for _, record in list_subjects(folder).iterrows()]


def real_subjects(root):
    """Deux sujets d'entraînement C1 de centres différents, avec lésion : celui qu'on transforme et une référence."""
    records = list_processed(root, "train", on_missing="warn", warn_empty=False)
    records = records[~records["empty_mask"]].drop_duplicates("site").head(2)
    subjects = [load_processed_subject(record) for _, record in records.iterrows()]
    for subject in subjects:
        subject.load()
    return subjects


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--source", choices=("raw", "processed"), default="raw",
                        help="format du volume synthétique : brut (défaut) ou prétraité (contrat C1)")
    parser.add_argument("--data-root", type=Path,
                        help="avec --source processed : vrais volumes C1 (sujets d'entraînement) au lieu du synthétique")
    args = parser.parse_args()
    if args.data_root and args.source != "processed":
        parser.error("--data-root s'utilise avec --source processed")

    with tempfile.TemporaryDirectory() as temporary:
        if args.data_root:
            subject, reference = real_subjects(args.data_root)
        else:
            subject, reference = synthetic_subjects(temporary, args.source)
        shape = "×".join(str(n) for n in subject["t1"].spatial_shape)
        origin = f"vrai sujet {subject['case_id']}" if args.data_root else "synthétique"
        print(f"Source {args.source} ({origin}), volume {shape} ({subject['t1'].data.numel() / 1e6:.1f} M voxels), "
              f"{torch.get_num_threads()} threads CPU, médiane de {args.repeats} essais\n")
        print("| Transformation | Temps (s) |\n|---|---|")
        extra = {"fda": dict(references=[reference["t1"].data[0]]),
                 "histogram_matching": dict(reference_quantiles=fit_reference_quantiles([reference]))}
        for name in INTENSITY_NAMES:
            transform = REGISTRY[name](p=1.0, **extra.get(name, {}))
            print(f"| `{name}` | {statistics.median(seconds(transform, subject, range(args.repeats))):.2f} |")
        # Configurations complètes, avec leurs probabilités : coût moyen par volume à l'entraînement.
        for config in ("intensity.yaml", "bigaug.yaml"):
            transform = build_transforms(CONFIGS / config)
            print(f"| `{config}` (moyenne de 10 tirages) | {statistics.mean(seconds(transform, subject, range(10))):.2f} |")


if __name__ == "__main__":
    main()
