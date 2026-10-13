"""Temps CPU de chaque transformation sur un volume synthétique de taille réaliste (1 mm).

    .venv-dl\\Scripts\\python -m generalization.benchmark
"""
import argparse
import statistics
import tempfile
import time
from pathlib import Path

import torch

from .data_raw import list_subjects, load_subject
from .synthetic import FOV_REALISTIC_MM, synthetic_volume, write_subject
from .transforms import INTENSITY_NAMES, REGISTRY, build_transforms, fit_reference_quantiles

CONFIGS = Path(__file__).resolve().parent / "configs"


def seconds(transform, subject, seeds):
    times = []
    for seed in seeds:
        torch.manual_seed(seed)
        start = time.perf_counter()
        transform(subject)
        times.append(time.perf_counter() - start)
    return times


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as temporary:
        for index, site in enumerate(("R009", "R001")):
            volume = synthetic_volume(fov_mm=FOV_REALISTIC_MM, seed=index)
            write_subject(temporary, site, f"sub-{site.lower()}s001", *volume, metadata={})
        subject, reference = (load_subject(record) for _, record in list_subjects(temporary).iterrows())
        shape = "×".join(str(n) for n in subject["t1"].spatial_shape)
        print(f"Volume {shape} ({subject['t1'].data.numel() / 1e6:.1f} M voxels), "
              f"{torch.get_num_threads()} threads CPU, médiane de {args.repeats} essais\n")
        print("| Transformation | Temps (s) |\n|---|---|")
        extra = {"fda": dict(references=[reference["t1"].data[0]]),
                 "histogram_matching": dict(reference_quantiles=fit_reference_quantiles([reference]))}
        for name in INTENSITY_NAMES:
            transform = REGISTRY[name](p=1.0, **extra.get(name, {}))
            print(f"| `{name}` | {statistics.median(seconds(transform, subject, range(args.repeats))):.2f} |")
        # Configurations complètes, avec leurs probabilités : coût moyen par volume à l'entraînement.
        for config in ("intensity.yaml", "bigaug.yaml"):
            transform = build_transforms(CONFIGS / config, data_root=temporary)
            print(f"| `{config}` (moyenne de 10 tirages) | {statistics.mean(seconds(transform, subject, range(10))):.2f} |")


if __name__ == "__main__":
    main()
