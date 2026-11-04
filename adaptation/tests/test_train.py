"""Boucle d'entraînement sur le jeu C1 synthétique (moins de 64³), CPU, quelques itérations."""
import csv
import json

import pytest
import torch

from adaptation.train import RESULT_COLUMNS, RunConfig, load_model, predict, run_evaluation, sample_patch, train


@pytest.fixture(scope="module")
def synthetic_c1(tmp_path_factory):
    from generalization.synthetic import write_processed_dataset
    return write_processed_dataset(tmp_path_factory.mktemp("c1") / "processed")


def tiny_config(synthetic_c1, tmp_path, technique="baseline"):
    root, splits = synthetic_c1
    return RunConfig(technique=technique, fold=0, data_root=str(root), splits=str(splits), output=str(tmp_path),
                     patch=32, patches_per_volume=2, volumes_per_batch=1, channels=(8, 16), strides=(2,),
                     num_res_units=0, max_iterations=3, validate_every=2, num_workers=0, amp=False,
                     time_budget_hours=0.1, tent_patches=1)


@pytest.mark.parametrize("technique", ["baseline", "dann"])
def test_entrainement_puis_evaluation(synthetic_c1, tmp_path, technique):
    cfg = tiny_config(synthetic_c1, tmp_path, technique)
    model, summary = train(cfg)
    output = tmp_path / cfg.name
    assert {"best.pt", "last.pt", "config.json", "journal_entrainement.csv", "resultats_id_test.csv",
            "resultats_ood_test.csv", "resume.json"} <= {path.name for path in output.iterdir()}
    with open(output / "resultats_ood_test.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == RESULT_COLUMNS and {row["site"] for row in rows} == {"SOOP"}
    assert {row["split"] for row in rows} == {"ood_test"} and rows[0]["technique"] == technique
    assert summary["entrainement"]["iteration"] == 3 and "ecart_id_ood" in summary
    journal = list(csv.DictReader(open(output / "journal_entrainement.csv", encoding="utf-8")))
    assert [row["iteration"] for row in journal] == ["2"]
    if technique == "dann":
        assert journal[0]["perte_centre"] != ""
    # Le checkpoint se recharge avec sa configuration (évaluation seule, TENT compris).
    reloaded, reloaded_cfg = load_model(output / "best.pt", torch.device("cpu"))
    assert reloaded_cfg.technique == technique and reloaded_cfg.channels == (8, 16)
    tent_summary = run_evaluation(reloaded, reloaded_cfg, output, tent=True)
    assert tent_summary["technique"] == technique + "+tent" and (output / "resultats_ood_test_tent.csv").is_file()


def test_reprise(synthetic_c1, tmp_path):
    cfg = tiny_config(synthetic_c1, tmp_path)
    train(cfg)
    cfg.max_iterations = 5
    _, summary = train(cfg, resume=True)
    assert summary["entrainement"]["iteration"] == 5


def test_predict_binaire_et_limite_au_cerveau():
    from adaptation.train import build_network
    cfg = RunConfig(patch=32, channels=(8, 16), strides=(2,), num_res_units=0, amp=False)
    model = build_network(cfg).eval()
    image = torch.randn(1, 40, 36, 30)  # plus petit que le patch sur un axe : rembourrage puis découpe
    brain = torch.zeros(1, 40, 36, 30, dtype=torch.bool)
    brain[:, 10:30, 10:26, 5:25] = True
    mask, probability = predict(model, image, brain, cfg)
    assert mask.shape == image.shape and set(mask.unique().tolist()) <= {0, 1}
    assert float(probability[~brain].abs().max()) == 0


def test_patch_centre_sur_la_lesion():
    image = torch.randn(1, 50, 50, 50)
    lesion = torch.zeros(1, 50, 50, 50)
    lesion[:, 40:44, 5:9, 20:24] = 1
    brain = torch.ones(1, 50, 50, 50, dtype=torch.bool)
    generator = torch.Generator().manual_seed(0)
    patch, patch_lesion, patch_brain = sample_patch(image, lesion, brain, 32, 1.0, generator)
    assert patch.shape == (1, 32, 32, 32) and patch_lesion.sum() > 0 and patch_brain.dtype == torch.bool
    _, patch_lesion, _ = sample_patch(image, torch.zeros_like(lesion), brain, 32, 1.0, generator)
    assert patch_lesion.sum() == 0  # masque vide : patch tiré dans le cerveau


def test_technique_inconnue(synthetic_c1, tmp_path):
    from adaptation.train import technique_hooks
    with pytest.raises(ValueError, match="Technique inconnue"):
        technique_hooks(tiny_config(synthetic_c1, tmp_path, "mixup"), ["R009"])
