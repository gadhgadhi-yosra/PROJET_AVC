"""Aucun réglage ne doit utiliser un centre de test OOD ni un centre exclu de splits.json."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from generalization.calibration import (CALIBRATION_FILE, DEFAULT_SPLITS, check_training_sites, gamma_range,
                                        held_out_sites, load_splits, restrict_to_training, site_table,
                                        splits_fingerprint, training_sites)
from generalization.transforms import build_transforms, load_config
from generalization.transforms.build import reference_records
from generalization.visual_check import select_records

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
pytestmark = pytest.mark.skipif(not DEFAULT_SPLITS.is_file(), reason="splits.json absent")


@pytest.fixture(scope="module")
def splits():
    return load_splits()


@pytest.fixture(scope="module")
def calibration():
    return json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))


def config_entry(name, transform):
    return next(entry for entry in load_config(CONFIGS / name)["transforms"] if entry["name"] == transform)


# --- Ce qui est versionné ---

@pytest.mark.parametrize("config", sorted(path.name for path in CONFIGS.glob("*.yaml")))
def test_les_centres_de_reference_sont_des_centres_d_entrainement(splits, config):
    for entry in load_config(CONFIGS / config).get("transforms") or []:
        if "reference_sites" in entry:
            check_training_sites(entry["reference_sites"], splits, f"{config} / {entry['name']}")
            # Le garde-fou s'applique aussi à l'exécution : références prises parmi les sujets d'entraînement.
            assert entry.get("splits") == "splits.json", (config, entry["name"])


def test_le_calibrage_n_utilise_aucun_centre_de_test(splits, calibration):
    assert calibration["held_out_sites"] == held_out_sites(splits)
    assert calibration["training_sites"] == training_sites(splits)
    assert calibration["splits"]["sha256_16"] == splits_fingerprint(DEFAULT_SPLITS), \
        "splits.json a changé depuis le calibrage : relancer python -m generalization.calibration et les balayages"
    used = {
        "gamma": calibration["gamma"]["sites"],
        "contraste": calibration["contrast"]["sites"],
        "référence": [calibration["reference"]["site"]] + [row["site"] for row in calibration["reference"]["ranking"]],
        "balayage de FDA": calibration["fda_beta"]["sweep_sites"] + [calibration["fda_beta"]["reference_site"]],
        "balayage du gamma de BigAug": calibration["bigaug_gamma"]["sweep_sites"],
    }
    for purpose, sites in used.items():
        check_training_sites(sites, splits, purpose)


def test_les_configs_reprennent_le_calibrage(calibration):
    assert config_entry("intensity.yaml", "gamma")["gamma"] == calibration["gamma"]["range"]
    assert config_entry("intensity.yaml", "contrast")["factor"] == calibration["contrast"]["range"]
    assert config_entry("bigaug.yaml", "gamma")["gamma"] == calibration["bigaug_gamma"]["range"]
    assert config_entry("fda.yaml", "fda")["beta"] == calibration["fda_beta"]["range"]
    for name, transform in [("fda.yaml", "fda"), ("histogram_matching.yaml", "histogram_matching")]:
        assert config_entry(name, transform)["reference_sites"] == [calibration["reference"]["site"]]


# --- Le garde-fou lui-même ---

def test_le_garde_fou_refuse_les_centres_ood_exclus_et_inconnus(splits):
    first_training_site = training_sites(splits)[0]
    assert check_training_sites(first_training_site, splits) == [first_training_site]
    for site in held_out_sites(splits):
        with pytest.raises(ValueError, match="interdits"):
            check_training_sites([first_training_site, site], splits)
    with pytest.raises(ValueError, match="absents"):
        check_training_sites(["R999"], splits)


def test_le_calibrage_refuse_des_mesures_d_un_centre_ood(splits):
    ood = splits["ood_centers"][0]
    stats = pd.DataFrame({"site": [training_sites(splits)[0], ood], "p50": [1.0, 1.0], "p99": [2.0, 2.0],
                          "cv": [0.4, 0.4], "levels": [500, 500], "voxel_min": [1.0, 1.0], "voxel_max": [1.0, 1.0]})
    with pytest.raises(ValueError, match="interdits"):
        site_table(stats, splits)


@pytest.mark.parametrize("transform", ["fda", "histogram_matching"])
def test_build_transforms_refuse_une_reference_ood(dataset_root, transform):
    # Le jeu synthétique contient SOOP, centre de test dans splits.json.
    entry = {"name": transform, "reference_sites": ["SOOP"], "splits": "splits.json", "n_references": 2}
    with pytest.raises(ValueError, match="interdits"):
        build_transforms({"transforms": [entry]}, data_root=dataset_root)


def test_visual_check_refuse_un_centre_ood(splits):
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-m", "generalization.visual_check", "--sites", "SOOP"],
                            capture_output=True, text=True, cwd=CONFIGS.parents[1])
    assert result.returncode != 0 and "interdits" in result.stderr


# --- Seuls les sujets d'entraînement servent, jamais le test ID ---

@pytest.fixture
def toy_splits(records, tmp_path):
    """splits.json synthétique : sub-r009s001 en entraînement, sub-r009s002 en test ID, SOOP en OOD."""
    rows = records[records["site"].isin(["R009", "SOOP"])]
    cases = [dict(case_id=f"{row.site}_{row.subject}_{row.session}", subject=row.subject, site=row.site,
                  session=row.session) for row in rows.itertuples()]
    by_subject = {case["subject"]: case["case_id"] for case in cases}
    splits = {"ood_centers": ["SOOP"], "excluded_centers": ["R032"], "cases": cases,
              "train": [by_subject["sub-r009s001"]], "id_test": [by_subject["sub-r009s002"]]}
    path = tmp_path / "splits.json"
    path.write_text(json.dumps(splits), encoding="utf-8")
    return path


def test_les_references_excluent_le_test_id(dataset_root, toy_splits):
    chosen = reference_records(["R009"], 8, dataset_root, splits=toy_splits)
    assert chosen["subject"].tolist() == ["sub-r009s001"]
    assert len(reference_records(["R009"], 8, dataset_root)) == 2  # sans splits : tout le centre


def test_restriction_aux_sujets_d_entrainement(records, dataset_root, toy_splits):
    splits = load_splits(toy_splits)
    assert restrict_to_training(records.reset_index(drop=True), splits)["subject"].tolist() == ["sub-r009s001"]
    assert select_records(dataset_root, ["R009"], ["sub-r009s002"], splits)[0]["subject"] == "sub-r009s001"


def test_formule_du_gamma():
    # Positions 0,4 / 0,6 / 0,7 : le centre médian (0,6) donne gamma = 1.
    table = pd.DataFrame({"position": [0.4, 0.6, 0.7]}, index=["A", "B", "C"])
    result = gamma_range(table)
    assert result["sites"] == ["C", "A"]
    assert result["raw"] == [round(np.log(0.7) / np.log(0.6), 3), round(np.log(0.4) / np.log(0.6), 3)]
    assert result["range"][0] <= result["raw"][0] and result["range"][1] >= result["raw"][1]
