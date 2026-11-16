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


# --- best.pt et garde-fou de divergence ---

def test_dice_nan_ne_sauvegarde_pas_best(synthetic_c1, tmp_path, monkeypatch):
    import adaptation.train as train_module
    monkeypatch.setattr(train_module, "validate", lambda *args, **kwargs: float("nan"))
    cfg = tiny_config(synthetic_c1, tmp_path)
    _, summary = train(cfg)
    output = tmp_path / cfg.name
    assert not (output / "best.pt").exists() and (output / "last.pt").is_file()
    assert summary["entrainement"]["best_dice"] == float("-inf") and summary["entrainement"]["best_iteration"] is None


def test_best_exige_dice_et_poids_finis(synthetic_c1, tmp_path, monkeypatch):
    import adaptation.train as train_module
    from adaptation.train import finite_parameters, improves
    assert improves(0.0, float("-inf")) and not improves(float("nan"), float("-inf"))
    assert not improves(float("inf"), 0.5) and not improves(0.4, 0.5)
    model = torch.nn.Linear(2, 1)
    assert finite_parameters(model)
    with torch.no_grad():
        model.weight[0, 0] = float("nan")
    assert not finite_parameters(model)
    # Dice fini (0,3) mais poids devenus NaN au moment de la validation : pas de best.pt, puis arrêt.
    def poisoned_validate(model, *args, **kwargs):
        with torch.no_grad():
            next(model.parameters()).fill_(float("nan"))
        return 0.3

    monkeypatch.setattr(train_module, "validate", poisoned_validate)
    cfg = tiny_config(synthetic_c1, tmp_path)  # validation à l'itération 2
    with pytest.raises(train_module.TrainingDiverged, match="Itération 3 : perte non finie"):
        train(cfg)
    assert not (tmp_path / cfg.name / "best.pt").exists()


def test_garde_fou():
    from adaptation.train import DivergenceGuard, TrainingDiverged
    model = torch.nn.Linear(2, 1)
    guard = DivergenceGuard(max_skipped=50)
    with pytest.raises(TrainingDiverged, match="perte non finie"):
        guard.check(7, float("nan"), model, skipped=False)
    for iteration in range(50):
        guard.check(iteration, 0.5, model, skipped=True)
    guard.check(50, 0.5, model, skipped=False)  # un pas appliqué remet le compteur à zéro
    assert guard.consecutive_skipped == 0 and guard.skipped == 50
    for iteration in range(50):
        guard.check(iteration, 0.5, model, skipped=True)
    with pytest.raises(TrainingDiverged, match="51 pas sautés d'affilée"):
        guard.check(101, 0.5, model, skipped=True)
    with torch.no_grad():
        model.bias.fill_(float("inf"))
    with pytest.raises(TrainingDiverged, match="poids non finis"):
        DivergenceGuard().check(3, 0.5, model, skipped=False)


class NaNOutput(torch.nn.Module):
    """Réseau dont la sortie est NaN : la perte l'est aussi dès la première itération."""

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x):
        return self.network(x) * float("nan")


def test_entrainement_arrete_sur_perte_nan(synthetic_c1, tmp_path):
    from adaptation.train import TrainingDiverged
    cfg = tiny_config(synthetic_c1, tmp_path)
    with pytest.raises(TrainingDiverged, match="Itération 1 : perte non finie"):
        train(cfg, wrap_model=NaNOutput)
    assert not (tmp_path / cfg.name / "best.pt").exists()


def test_fp32_et_clipping_par_defaut(synthetic_c1, tmp_path, monkeypatch):
    defaults = RunConfig()
    assert defaults.amp is False and defaults.grad_clip == 12.0
    assert defaults.validate_every == 2000 and defaults.log_every == 100
    calls = []
    original = torch.nn.utils.clip_grad_norm_

    def spy(parameters, max_norm, *args, **kwargs):
        calls.append(max_norm)
        return original(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    cfg = tiny_config(synthetic_c1, tmp_path)
    cfg.grad_clip = 12.0
    train(cfg)
    assert calls == [12.0] * cfg.max_iterations
    calls.clear()
    cfg.grad_clip = None
    train(cfg)
    assert calls == []


# --- Reprise exacte (sessions coupées) ---

class Interruption(RuntimeError):
    pass


@pytest.mark.parametrize("technique, workers", [("baseline", 0), ("baseline", 2), ("intensity", 2)])
def test_reprise_meme_trajectoire(synthetic_c1, tmp_path, monkeypatch, technique, workers):
    """Coupure après l'itération 4 (last.pt à 3), reprise : mêmes poids finaux qu'une run sans coupure."""
    import adaptation.train as train_module
    import types
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # CPU : calcul déterministe, comparaison exacte
    # Horloge figée : le LR de la première itération dépend sinon de quelques secondes d'horloge
    # (progress = max(temps / budget, itération / max)), y compris entre deux runs sans coupure.
    clock = types.SimpleNamespace(time=lambda: 1000.0, perf_counter=train_module.time.perf_counter)
    monkeypatch.setattr(train_module, "time", clock)

    def config(folder):
        cfg = tiny_config(synthetic_c1, tmp_path / folder, technique)
        cfg.max_iterations, cfg.validate_every, cfg.num_workers, cfg.checkpoint_minutes = 6, 3, workers, -1.0
        cfg.evaluation_splits = ()
        return cfg

    straight = config("continu")
    train(straight)
    original_check = train_module.DivergenceGuard.check

    def crash(self, iteration, *args, **kwargs):
        if iteration == 4:
            raise Interruption("coupure simulée")
        return original_check(self, iteration, *args, **kwargs)

    resumed = config("coupe")
    monkeypatch.setattr(train_module.DivergenceGuard, "check", crash)
    with pytest.raises(Interruption):
        train(resumed)
    monkeypatch.setattr(train_module.DivergenceGuard, "check", original_check)
    interrupted = torch.load(tmp_path / "coupe" / resumed.name / "last.pt", weights_only=False)
    assert interrupted["state"]["iteration"] == 3 and len(interrupted["reprise"]["rng_lots"]) == max(workers, 1)
    train(resumed, resume=True)
    a = torch.load(tmp_path / "continu" / straight.name / "last.pt", weights_only=False)
    b = torch.load(tmp_path / "coupe" / resumed.name / "last.pt", weights_only=False)
    assert a["state"]["iteration"] == b["state"]["iteration"] == 6
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key]), key
    assert a["state"]["best_iteration"] == b["state"]["best_iteration"]


def test_reprise_dann_exacte_hors_dropout(synthetic_c1, tmp_path, monkeypatch):
    """DANN : reprise identique à la run sans coupure, sauf les masques du Dropout du classifieur de centre.

    Ils sont tirés dans le générateur du processus principal (CUDA sur GPU), dont l'état n'est pas dans last.pt :
    après une reprise, ces masques diffèrent de la run sans coupure. Lots, augmentations, LR, lambda et poids
    restaurés sont exacts ; le Dropout neutralisé, les poids finaux sont identiques au bit près.
    """
    monkeypatch.setattr(torch.nn.Dropout, "forward", lambda self, x: x)
    test_reprise_meme_trajectoire(synthetic_c1, tmp_path, monkeypatch, "dann", 2)


def test_echantillonneur_identique_a_randomsampler():
    from adaptation.train import ResumableSampler
    data = list(range(7))
    reference = list(torch.utils.data.RandomSampler(data, replacement=True, num_samples=100,
                                                    generator=torch.Generator().manual_seed(2026)))
    assert list(ResumableSampler(data, 100, 2026)) == reference
    assert list(ResumableSampler(data, 100, 2026, start=37)) == reference[37:]


def test_sauvegarde_atomique(tmp_path, monkeypatch):
    """Une coupure pendant l'écriture laisse l'ancien checkpoint lisible."""
    import adaptation.train as train_module
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    path = tmp_path / "last.pt"
    train_module.save_checkpoint(path, model, optimizer, scaler, dict(iteration=1), RunConfig())

    def broken_save(obj, target):
        open(target, "wb").write(b"tronque")
        raise OSError("coupure")

    monkeypatch.setattr(torch, "save", broken_save)
    with pytest.raises(OSError):
        train_module.save_checkpoint(path, model, optimizer, scaler, dict(iteration=2), RunConfig())
    monkeypatch.undo()
    assert torch.load(path, weights_only=False)["state"]["iteration"] == 1


def test_garde_fou_perte_trop_grande():
    from adaptation.train import DivergenceGuard, TrainingDiverged
    model = torch.nn.Linear(2, 1)
    DivergenceGuard().check(1, 99.0, model, skipped=False)
    with pytest.raises(TrainingDiverged, match=r"Itération 7 : perte totale 1.051e\+04 > 100 \(divergence\)"):
        DivergenceGuard().check(7, 10508.0, model, skipped=False)
