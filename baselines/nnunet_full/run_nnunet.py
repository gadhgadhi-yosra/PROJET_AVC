"""Run nnU-Net v2 (3d_fullres baseline, PDF section 3.4) on Dataset601_ATLAS3. Use the project venv:

    nnunet/.venv/Scripts/python nnunet/nnunet_full/run_nnunet.py plan      # fingerprint + plans + preprocessing
    nnunet/.venv/Scripts/python nnunet/nnunet_full/run_nnunet.py train     # fold 0 (ID val = other patients, same centres)
    nnunet/.venv/Scripts/python nnunet/nnunet_full/run_nnunet.py predict   # OOD centres (imagesTs), same checkpoint
    nnunet/.venv/Scripts/python nnunet/nnunet_full/run_nnunet.py all

The laptop GPU has 4 GB (RTX 2050), so plans are generated for a 3.5 GB target under a separate plans
name (nnUNetPlans_4GB) and the trainer is shortened (TRAINER below); the default nnU-Net schedule
(1000 epochs) would take days on this GPU. torch.compile is off (no Triton on Windows).
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]  # nnunet/ root
DATASET_ID = "601"
DATASET = "Dataset601_ATLAS3"
CONFIG = "3d_fullres"
PLANS = "nnUNetPlans_4GB"
GPU_GB = "3.5"
TRAINER = os.environ.get("TRAINER", "nnUNetTrainer_100epochs")
FOLD = "0"
BIN = Path(sys.executable).parent

ENV = dict(os.environ,
           nnUNet_raw=str(HERE / "nnUNet_raw"),
           nnUNet_preprocessed=str(HERE / "nnUNet_preprocessed"),
           nnUNet_results=str(HERE / "nnUNet_results"),
           nnUNet_compile="f",
           nnUNet_n_proc_DA="6",
           PYTHONUNBUFFERED="1")


def run(*cmd):
    print(">>", " ".join(cmd), flush=True)
    subprocess.run([str(BIN / cmd[0]), *cmd[1:]], env=ENV, check=True)


def plan():
    run("nnUNetv2_plan_and_preprocess", "-d", DATASET_ID, "-c", CONFIG, "-np", "3", "-npfp", "3",
        "-gpu_memory_target", GPU_GB, "-overwrite_plans_name", PLANS, "--verify_dataset_integrity")


def train():
    args = ["nnUNetv2_train", DATASET_ID, CONFIG, FOLD, "-tr", TRAINER, "-p", PLANS]
    fold_dir = HERE / "nnUNet_results" / DATASET / f"{TRAINER}__{PLANS}__{CONFIG}" / f"fold_{FOLD}"
    if (fold_dir / "checkpoint_latest.pth").exists() and not (fold_dir / "checkpoint_final.pth").exists():
        args.append("--c")  # resume an interrupted run
    run(*args)


def predict():
    out = HERE / "predictions_ood" / f"{TRAINER}__{PLANS}__{CONFIG}_fold{FOLD}"
    run("nnUNetv2_predict", "-i", str(HERE / "nnUNet_raw" / DATASET / "imagesTs"), "-o", str(out),
        "-d", DATASET_ID, "-c", CONFIG, "-tr", TRAINER, "-p", PLANS, "-f", FOLD,
        "-chk", "checkpoint_final.pth", "-npp", "2", "-nps", "2")


if __name__ == "__main__":
    steps = sys.argv[1:] or ["all"]
    if steps == ["all"]:
        steps = ["plan", "train", "predict"]
    for s in steps:
        {"plan": plan, "train": train, "predict": predict}[s]()
