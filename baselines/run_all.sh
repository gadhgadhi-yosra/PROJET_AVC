#!/usr/bin/env bash
# Full pipeline, from nnunet/:  bash run_all.sh
#   preprocessing -> ML models (CPU) in parallel with the 4 CNN variants (GPU) -> results table + figures
set -eu
cd "$(dirname "$0")"
PY=.venv/Scripts/python
MIN=${MIN:-8}            # training minutes per CNN variant
mkdir -p outputs/logs

$PY preprocessing/01_build_dataset.py   # splits + nnU-Net raw dataset (fast: images are hard links)
$PY preprocessing/02_prepare_arrays.py  # RAS, 1 mm, crop, z-score -> data_fast/

OMP_NUM_THREADS=2 $PY training/train_ml.py > outputs/logs/ml.log 2>&1 &
ML=$!
$PY training/train_cnn.py --freeze none    --minutes "$MIN" > outputs/logs/cnn_freeze-none.log 2>&1
$PY training/train_cnn.py --freeze encoder --minutes "$MIN" > outputs/logs/cnn_freeze-encoder.log 2>&1
$PY training/train_cnn.py --freeze partial --minutes "$MIN" > outputs/logs/cnn_freeze-partial.log 2>&1
$PY training/train_cnn.py --random         --minutes "$MIN" > outputs/logs/cnn_random.log 2>&1
wait $ML

$PY results/make_results.py
