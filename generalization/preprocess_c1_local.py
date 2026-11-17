"""Pilote local du prétraitement C1 sans Spark, pour Windows (Spark y exige winutils.exe).

Appelle tels quels `pipeline.inventory` et `week4.work` de preprocessing/ (code de Yosra, non
modifié) ; seule la répartition change : des processus Python remplacent `spark_run`, avec la même
règle (volumes de plus de 40 M voxels traités un par un, à la fin). Écrit `<output>/subjects.parquet`
et `<output>/summary.json` comme `week4.run`. Reprise automatique des sujets déjà traités.
À lancer dans l'environnement du pipeline (.venv-spark, requirements.txt racine) :

    .venv-spark\\Scripts\\python generalization\\preprocess_c1_local.py --workers 2
    .venv-spark\\Scripts\\python generalization\\preprocess_c1_local.py --cases selection.json   # sous-ensemble
"""
import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
    os.environ[_key] = "1"

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "preprocessing"))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pipeline import inventory  # noqa: E402
import week4  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO / "data" / "ATLAS3_Training_Raw")
    parser.add_argument("--output", type=Path, default=REPO / "data" / "processed")
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    rows = inventory(args.root)
    if args.cases:
        selected = set(json.loads(args.cases.read_text()))
        rows = [row for row in rows if week4.case_id(row) in selected]
        if {week4.case_id(row) for row in rows} != selected:
            raise ValueError("Sélection contenant des cas inconnus")
    if len({row["subject"] for row in rows}) != len(rows):
        raise ValueError("Plusieurs sessions par sujet")
    args.output.mkdir(parents=True, exist_ok=True)
    output = str(args.output.resolve())
    # Même règle que week4.spark_run : les volumes de plus de 40 M voxels passent un par un, à la fin.
    small, large = [], []
    for row in rows:
        try:
            big = row["site"] != "R032" and row["t1_path"] and np.prod(nib.load(row["t1_path"]).shape) > 40_000_000
        except Exception:
            big = False  # le worker consignera le fichier illisible comme erreur de QC
        (large if big else small).append(row)
    print(f"{len(small)} sujets en parallèle ({args.workers} processus), {len(large)} gros volumes un par un", flush=True)
    started = time.perf_counter()
    work = partial(week4.work, output=output)
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(work, small, chunksize=1))
    print(f"Parallèle terminé en {time.perf_counter() - started:.0f} s", flush=True)
    with ProcessPoolExecutor(1) as pool:
        results += list(pool.map(work, large, chunksize=1))
    seconds = time.perf_counter() - started
    frame = pd.DataFrame(results)
    frame.to_parquet(args.output / "subjects.parquet", index=False)
    summary = dict(version=week4.VERSION, subjects=len(rows), sites=int(frame.site.nunique()),
                   statuses=frame.status.value_counts().to_dict(), seconds=seconds, workers=args.workers,
                   orchestration="ProcessPoolExecutor (sans Spark)")
    week4.write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    errors = frame[frame.status == "error"]
    if len(errors):
        print(errors[["case_id", "error"]].to_string())


if __name__ == "__main__":
    main()
