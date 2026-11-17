"""Adaptation continue sur le jeu C1 synthétique : deux centres OOD, quatre stratégies, CPU."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from adaptation.continual import ContinualConfig, Controlled, detection_threshold, run
from adaptation.train import RunConfig, train


@pytest.fixture(scope="module")
def setting(tmp_path_factory):
    from generalization.synthetic import write_processed_dataset
    folder = tmp_path_factory.mktemp("continu")
    root, splits_path = write_processed_dataset(folder / "processed")
    cfg = RunConfig(technique="baseline", fold=0, data_root=str(root), splits=str(splits_path), output=str(folder),
                    patch=32, patches_per_volume=2, volumes_per_batch=1, channels=(8, 16), strides=(2,),
                    num_res_units=0, max_iterations=2, validate_every=2, num_workers=0, time_budget_hours=0.1,
                    evaluation_splits=())
    train(cfg)
    # Deux centres OOD pour le flux : SOOP puis R014 (sujets de validation, uniquement pour ce test).
    splits = json.loads(splits_path.read_text())
    splits["ood_centers"] = ["SOOP", "R014"]
    splits["ood_test"] = splits["ood_test"] + splits["folds"][0]["val"]
    stream_splits = folder / "splits_flux.json"
    stream_splits.write_text(json.dumps(splits))
    return folder / cfg.name / "best.pt", root, stream_splits, folder


def test_flux_prequentiel(setting):
    checkpoint, root, splits, folder = setting
    ccfg = ContinualConfig(subjects_per_center=None, lot_size=1, patches_per_subject=2, guard_subjects=1,
                           source_replay_subjects=1, forgetting_id_subjects=1, forgetting_per_center=1,
                           reference_per_site=3)
    output = folder / "continu"
    summary = run(checkpoint, root, splits, output, ccfg, device=torch.device("cpu"))
    assert summary["centres"] == ["SOOP", "R014"] and summary["lots"] == 3
    assert set(summary["strategies"]) == {"fige", "auto_apprentissage", "tent", "controlee"}
    prequential = pd.read_csv(output / "prequentiel.csv")
    assert len(prequential) == 4 * 3  # chaque sujet noté une fois par stratégie
    # Avant toute adaptation, le premier lot est noté de la même façon par toutes les stratégies.
    first = prequential[prequential["lot"] == 1]["dice"].to_numpy()
    assert np.allclose(first, first[0], equal_nan=True)
    forgetting = pd.read_csv(output / "oubli.csv")
    assert set(forgetting["apres_centre"]) == {"depart", "SOOP", "R014"}
    assert {"id_source", "ood_SOOP"} <= set(forgetting["ensemble"])
    fige = forgetting[forgetting["strategie"] == "fige"]
    assert fige[fige["ensemble"] == "id_source"]["dice"].nunique() == 1  # le modèle figé n'oublie rien
    decisions = pd.read_csv(output / "decisions.csv")
    assert len(decisions) == 4 * 3 and set(decisions[decisions["strategie"] == "controlee"]["raison"]) <= {
        "promu", "garde en baisse", "entropie en hausse", "pas de décalage détecté"}
    assert summary["strategies"]["fige"]["oubli_id"] == 0
    assert json.loads((output / "resume.json").read_text(encoding="utf-8"))["lots"] == 3


def test_seuil_de_detection():
    rng = np.random.default_rng(0)
    reference = rng.dirichlet(np.ones(150), size=20)
    threshold = detection_threshold(reference, 4, 0.95, 0)
    assert threshold > 0
    with pytest.warns(UserWarning, match="seuil nul"):
        assert detection_threshold(reference[:2], 4, 0.95, 0) == 0.0


def test_champion_garde_si_garde_en_baisse(setting):
    """Un challenger qui dégrade le jeu de garde n'est pas promu, même si son entropie baisse."""
    from adaptation.train import load_model
    checkpoint, *_ = setting
    model, cfg = load_model(checkpoint, torch.device("cpu"))
    controlled = Controlled(model, cfg, ContinualConfig(), guard=[], source_replay=[])
    controlled.champion_guard = 0.9
    controlled.guard_dice = lambda network: 0.5
    controlled.start_lot(detected=True)
    patches = (torch.randn(2, 1, 32, 32, 32), torch.zeros(2, 1, 32, 32, 32), torch.ones(2, 1, 32, 32, 32).bool())
    champion = controlled.model
    decision = controlled.end_lot(patches, "SOOP", [])
    assert decision["promu"] is False and decision["raison"] == "garde en baisse" and controlled.model is champion
