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
    assert [reference["fold"] for reference in calibration["fold_references"]] == list(range(len(splits["folds"])))
    for reference in calibration["fold_references"]:
        sites = [reference["site"]] + [row["site"] for row in reference["ranking"]]
        check_training_sites(sites, splits, f"la référence du pli {reference['fold']}", reference["fold"])


@pytest.mark.parametrize("config", sorted(path.name for path in CONFIGS.glob("*.yaml")))
def test_la_reference_d_un_pli_n_est_ni_un_centre_de_validation_ni_un_centre_de_test(splits, config):
    for entry in load_config(CONFIGS / config).get("transforms") or []:
        if "reference_sites" in entry:
            by_fold = entry.get("fold_reference_sites")
            assert by_fold and len(by_fold) == len(splits["folds"]), (config, entry["name"])
            for fold, site in enumerate(by_fold):
                check_training_sites(site, splits, f"{config} / {entry['name']} / pli {fold}", fold)


def test_les_configs_reprennent_le_calibrage(calibration):
    assert config_entry("intensity.yaml", "gamma")["gamma"] == calibration["gamma"]["range"]
    assert config_entry("intensity.yaml", "contrast")["factor"] == calibration["contrast"]["range"]
    assert config_entry("bigaug.yaml", "gamma")["gamma"] == calibration["bigaug_gamma"]["range"]
    assert config_entry("fda.yaml", "fda")["beta"] == calibration["fda_beta"]["range"]
    for name, transform in [("fda.yaml", "fda"), ("histogram_matching.yaml", "histogram_matching")]:
        assert config_entry(name, transform)["reference_sites"] == [calibration["reference"]["site"]]
        assert config_entry(name, transform)["fold_reference_sites"] == [
            reference["site"] for reference in calibration["fold_references"]]


# --- Le garde-fou lui-même ---

def test_le_garde_fou_refuse_les_centres_ood_exclus_et_inconnus(splits):
    first_training_site = training_sites(splits)[0]
    assert check_training_sites(first_training_site, splits) == [first_training_site]
    for site in held_out_sites(splits):
        with pytest.raises(ValueError, match="interdits"):
            check_training_sites([first_training_site, site], splits)
    with pytest.raises(ValueError, match="absents"):
        check_training_sites(["R999"], splits)
    for fold, content in enumerate(splits["folds"]):
        with pytest.raises(ValueError, match=f"validation du pli {fold}"):
            check_training_sites(content["validation_centers"][0], splits, fold=fold)


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


@pytest.fixture
def fold_splits(records, tmp_path):
    """Deux plis synthétiques : pli 0 = R009 en entraînement, R014 en validation ; pli 1 = l'inverse."""
    cases = [dict(case_id=f"{row.site}_{row.subject}_{row.session}", subject=row.subject, site=row.site,
                  session=row.session) for row in records[records["site"].isin(["R009", "R014"])].itertuples()]
    by_site = {site: [case["case_id"] for case in cases if case["site"] == site] for site in ("R009", "R014")}
    splits = {"ood_centers": ["SOOP"], "excluded_centers": ["R032"], "cases": cases,
              "train": by_site["R009"] + by_site["R014"], "id_test": [],
              "folds": [{"train": by_site["R009"], "val": by_site["R014"], "validation_centers": ["R014"]},
                        {"train": by_site["R014"], "val": by_site["R009"], "validation_centers": ["R009"]}]}
    path = tmp_path / "splits_plis.json"
    path.write_text(json.dumps(splits), encoding="utf-8")
    return path


def fold_entry(name, by_fold, splits_path):
    return {"name": name, "reference_sites": ["R009"], "fold_reference_sites": by_fold,
            "splits": str(splits_path), "n_references": 4, "p": 1.0}


def test_build_transforms_prend_la_reference_du_pli(dataset_root, fold_splits):
    entry = fold_entry("fda", ["R009", "R014"], fold_splits)
    for fold, site in enumerate(["R009", "R014"]):
        bank = build_transforms({"transforms": [dict(entry)]}, data_root=dataset_root, fold=fold)[0].references
        assert {source.record["site"] for source in bank.sources} == {site}
    # Sans pli : `reference_sites`, pour l'entraînement final.
    bank = build_transforms({"transforms": [dict(entry)]}, data_root=dataset_root)[0].references
    assert {source.record["site"] for source in bank.sources} == {"R009"}


@pytest.mark.parametrize("transform", ["fda", "histogram_matching"])
def test_build_transforms_refuse_un_centre_de_validation_du_pli(dataset_root, fold_splits, transform):
    # R014 est un centre de validation du pli 0.
    entry = fold_entry(transform, ["R014", "R014"], fold_splits)
    with pytest.raises(ValueError, match="validation du pli 0"):
        build_transforms({"transforms": [entry]}, data_root=dataset_root, fold=0)


def test_un_pli_exige_ses_references(dataset_root, fold_splits):
    entry = {"name": "fda", "reference_sites": ["R009"], "splits": str(fold_splits)}
    with pytest.raises(ValueError, match="fold_reference_sites"):
        build_transforms({"transforms": [entry]}, data_root=dataset_root, fold=0)
    with pytest.raises(ValueError, match="Pli 5"):
        build_transforms({"transforms": [fold_entry("fda", ["R009", "R014"], fold_splits)]},
                         data_root=dataset_root, fold=5)


def test_le_calcul_par_pli_ignore_la_validation(monkeypatch):
    # Quatre centres fictifs à histogrammes ponctuels. Sur tout le pool, D est le plus central : candidat,
    # il gagnerait. Il est centre de validation du pli 0, donc la référence de ce pli est B.
    import generalization.calibration as calibration
    monkeypatch.setattr(calibration, "REFERENCE_MIN_SUBJECTS", 1)
    position = {"A": 50, "B": 60, "C": 64, "D": 61}
    count = {"A": 1, "B": 2, "C": 2, "D": 2}
    cases, rows = [], []
    for site, bin_index in position.items():
        for index in range(1, count[site] + 1):
            subject = f"sub-{site.lower()}{index}"
            cases.append(dict(case_id=f"{site}_{subject}_ses-1", subject=subject, site=site, session="ses-1"))
            histogram = np.zeros(len(calibration.HIST_EDGES) - 1)
            histogram[bin_index] = 1.0
            rows.append(dict(site=site, subject=subject, session="ses-1", p50=1.0, p99=2.0, cv=0.4, levels=500,
                             voxel_min=1.0, voxel_max=1.0,
                             **{f"h{i:03d}": value for i, value in enumerate(histogram)}))
    ids = lambda sites: [case["case_id"] for case in cases if case["site"] in sites]  # noqa: E731
    splits = {"ood_centers": [], "excluded_centers": [], "cases": cases, "train": ids("ABCD"),
              "folds": [{"train": ids("ABC"), "validation_centers": ["D"]},
                        {"train": ids("BCD"), "validation_centers": ["A"]}]}
    stats = pd.DataFrame(rows)
    assert calibration.reference_ranking(stats, calibration.site_table(stats, splits))[0]["site"] == "D"
    references = calibration.fold_references(stats, splits)
    assert [reference["site"] for reference in references] == ["B", "D"]
    for reference, excluded, n in zip(references, ["D", "A"], [5, 6]):
        assert excluded not in [row["site"] for row in reference["ranking"]]
        assert reference["n_subjects"] == n


def test_formule_du_gamma():
    # Positions 0,4 / 0,6 / 0,7 : le centre médian (0,6) donne gamma = 1.
    table = pd.DataFrame({"position": [0.4, 0.6, 0.7]}, index=["A", "B", "C"])
    result = gamma_range(table)
    assert result["sites"] == ["C", "A"]
    assert result["raw"] == [round(np.log(0.7) / np.log(0.6), 3), round(np.log(0.4) / np.log(0.6), 3)]
    assert result["range"][0] <= result["raw"][0] and result["range"][1] >= result["raw"][1]
