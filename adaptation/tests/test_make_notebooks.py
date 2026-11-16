"""Notebooks RQ3 (Kaggle et Colab) exécutés en local contre une fausse arborescence d'entrée et un faux Drive."""
import json
import os
import pathlib
import shutil
import sys
import zipfile

import pytest

from adaptation.kaggle import make_notebooks as mk

FAKE_TRAIN = '''
import argparse, json, sys
from pathlib import Path
parser = argparse.ArgumentParser()
for name in ("--technique", "--fold", "--seed", "--data-root", "--output", "--num-workers", "--validate-every",
             "--log-every"):
    parser.add_argument(name)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()
assert (Path(args.data_root) / "subjects.parquet").exists() and any(Path(args.data_root).glob("*/qc.json"))
run = Path(args.output) / f"unet3d_{args.technique}_pli{args.fold}"
run.mkdir(parents=True, exist_ok=True)
print({'progression': 100, 'heures': 0.03, 's_par_iteration': 0.82, 'perte': 1.2, 'lr': '1.00e-03'})
print({'progression': 200, 'heures': 0.05, 's_par_iteration': 0.5, 'perte': 1.1, 'lr': '1.00e-03'})
(run / "resume.json").write_text(json.dumps({"run": run.name, "technique": args.technique, "reprise": args.resume}))
(run / "resultats_ood_test.csv").write_text("case_id,dice\\nR008_s001,0.5\\n")
(run / "last.pt").write_bytes(b"0" * 10)
'''


@pytest.fixture
def kaggle(tmp_path, monkeypatch):
    """Données montées sous datasets/<propriétaire>/<slug>/ (cas d'un dataset partagé), code en .zip ailleurs."""
    from generalization.synthetic import write_processed_dataset
    root, splits = write_processed_dataset(tmp_path / "source" / "processed")
    data = tmp_path / "input" / "datasets" / "fayfahachicha" / "atlas-c1-processed"
    for folder in root.iterdir():
        if folder.is_dir():
            shutil.copytree(folder, data / "processed_partie1" / "processed" / folder.name)
    shutil.copy(root / "subjects.parquet", data / "subjects.parquet")
    shutil.copy(splits, data / "splits.json")
    code = tmp_path / "input" / "projet-avc-code"
    code.mkdir(parents=True)
    with zipfile.ZipFile(code / "projet_avc_code.zip", "w") as archive:
        archive.writestr("VERSION.json", json.dumps({"commit": "abc1234", "message": "essai"}))
        archive.write(splits, "splits.json")
        archive.writestr("adaptation/__init__.py", "")
        archive.writestr("adaptation/train.py", FAKE_TRAIN)
    for key, value in dict(RQ3_INPUT=tmp_path / "input", RQ3_WORKING=tmp_path / "working",
                           RQ3_TMP=tmp_path / "tmp", RQ3_PIP="0", RQ3_GPU="0").items():
        monkeypatch.setenv(key, str(value))
    if os.name == "nt":  # pas de liens symboliques sans droits d'administrateur sous Windows : copie
        def copy_instead(self, target, target_is_directory=False):
            (shutil.copytree if target_is_directory else shutil.copy)(target, self)
        monkeypatch.setattr(pathlib.Path, "symlink_to", copy_instead)
    return tmp_path


def execute(notebook, stop=None):
    namespace = {}
    for index, cell in enumerate(c for c in notebook["cells"] if c["cell_type"] == "code"):
        if stop is not None and index >= stop:
            break
        exec(compile("".join(cell["source"]), f"cellule {index}", "exec"), namespace)
    return namespace


def test_run_complet_et_alerte_de_lenteur(kaggle, capsys):
    notebook = mk.notebook("03_fda", "fda", "FDA", "7 à 9 h", commit="abc1234")
    namespace = execute(notebook)
    assert namespace["VERSION"]["commit"] == "abc1234"
    run = kaggle / "working" / "runs" / "unet3d_fda_pli0"
    log = (run / "entrainement.log").read_text(encoding="utf-8")
    assert "ATTENTION : 0.820 s/it à l'itération 100" in log and "La run continue" in log
    assert log.count("ATTENTION") == 1  # rien à l'itération 200
    assert (run / "resume.json").is_file() and "resume.json" in capsys.readouterr().out
    assert not any((kaggle / "working").rglob("qc.json"))  # aucune donnée dans les sorties


def test_commit_different(kaggle):
    notebook = mk.notebook("03_fda", "fda", "FDA", "7 à 9 h", commit="0000000")
    with pytest.raises(RuntimeError, match="commit \\['abc1234'\\] au lieu de 0000000"):
        execute(notebook, stop=2)


def test_sans_gpu(kaggle, monkeypatch):
    import torch
    monkeypatch.setenv("RQ3_GPU", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="AUCUN GPU DISPONIBLE"):
        execute(mk.notebook("02_bigaug", "bigaug", "BigAug", "6 h 30", commit="abc1234"), stop=1)


def test_dataset_de_donnees_absent(kaggle):
    shutil.rmtree(kaggle / "input" / "datasets")
    with pytest.raises(FileNotFoundError, match="atlas-c1-processed"):
        execute(mk.notebook("02_bigaug", "bigaug", "BigAug", "6 h 30", commit="abc1234"), stop=3)


def test_fichiers_generes(tmp_path, monkeypatch):
    monkeypatch.setattr(mk, "HERE", tmp_path / "ref")
    monkeypatch.setattr(sys, "argv", ["make_notebooks", "--kernels-dir", str(tmp_path / "k"),
                                      "--colab-dir", str(tmp_path / "c")])
    mk.main()
    colab = sorted(path.name for path in (tmp_path / "c").glob("*.ipynb"))
    assert colab == ["rq3-adaptation-continue-pli0.ipynb", "rq3-dann-v2-pli0.ipynb", "rq3-histogram-matching-pli0.ipynb"]
    kernels = {path.parent.name: json.loads(path.read_text()) for path in (tmp_path / "k").rglob("kernel-metadata.json")}
    assert set(kernels) == {"01_intensite", "02_bigaug", "03_fda", "07_adaptation_continue"}
    assert kernels["02_bigaug"]["id"] == "fayfahachicha/rq3-bigaug-pli0"
    assert kernels["01_intensite"]["id"] == "fayfahachicha/rq3-intensity-pli0"
    assert kernels["07_adaptation_continue"]["kernel_sources"] == ["fayfahachicha/rq3-baseline-fp32-pli0"]
    kaggle_cells = "".join(map("".join, (c["source"] for c in json.loads(
        (tmp_path / "k" / "01_intensite" / "01_intensite.ipynb").read_text(encoding="utf-8"))["cells"])))
    assert "--resume\"]" not in kaggle_cells and "google.colab" not in kaggle_cells  # Kaggle : comme avant
    colab_notebook = json.loads((tmp_path / "c" / "rq3-dann-v2-pli0.ipynb").read_text(encoding="utf-8"))
    colab_source = "".join(map("".join, (c["source"] for c in colab_notebook["cells"])))
    assert colab_notebook["metadata"]["accelerator"] == "GPU"
    assert '"--technique", "dann"' in colab_source and '"--num-workers", "4"' in colab_source
    assert '"--resume"]' in colab_source


@pytest.fixture
def colab(kaggle, monkeypatch):
    """Faux modules google.colab : Drive dans un dossier temporaire, secret KAGGLE_API_TOKEN, wake lock actif."""
    import types
    mounted = []
    modules = {name: types.ModuleType(name) for name in ("google", "google.colab", "google.colab.drive",
                                                         "google.colab.userdata", "google.colab.output")}
    modules["google.colab.drive"].mount = mounted.append
    modules["google.colab.userdata"].get = lambda name: {"KAGGLE_API_TOKEN": "jeton-factice"}[name]
    modules["google.colab.output"].eval_js = lambda script: "actif"
    modules["google"].colab = modules["google.colab"]
    for name in ("drive", "userdata", "output"):
        setattr(modules["google.colab"], name, modules[f"google.colab.{name}"])
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    # Datasets déjà téléchargés (marqueurs) : pas d'appel à l'API Kaggle.
    for folder in ("projet-avc-code", "atlas-c1-processed"):
        (kaggle / "input" / folder).mkdir(parents=True, exist_ok=True)
        (kaggle / "input" / folder / ".complet").touch()
    monkeypatch.setenv("RQ3_DRIVE", str(kaggle / "drive"))
    for key in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY", "RQ3_RUNS", "RQ3_PLATFORM"):
        monkeypatch.delenv(key, raising=False)
    return kaggle, mounted


def test_colab_drive_reprise_et_jeton(colab):
    root, mounted = colab
    notebook = mk.notebook("01_intensite", "intensity", "Intensité", "7 h", commit="abc1234", colab=True)
    execute(notebook)
    drive_run = root / "drive" / "MyDrive" / "PROJET_AVC" / "rq3-intensity-pli0"
    run = drive_run / "runs" / "unet3d_intensity_pli0"
    assert mounted == [str(root / "drive")] and os.environ["KAGGLE_API_TOKEN"] == "jeton-factice"
    assert json.loads((run / "resume.json").read_text())["reprise"] is True  # toujours --resume sur Colab
    environment = json.loads(next((drive_run / "runs").glob("environnement_colab_*.json")).read_text())
    assert environment["plateforme"] == "colab" and environment["avertissements"]  # pas de T4 ni torch Kaggle ici
    with zipfile.ZipFile(drive_run / "runs" / "resultats_unet3d_intensity_pli0.zip") as archive:
        assert sorted(archive.namelist()) == ["runs/unet3d_intensity_pli0/entrainement.log",
                                              "runs/unet3d_intensity_pli0/resultats_ood_test.csv",
                                              "runs/unet3d_intensity_pli0/resume.json"]
    assert "jeton-factice" not in json.dumps(notebook)


def test_colab_sans_jeton(colab, monkeypatch):
    sys.modules["google.colab.userdata"].get = lambda name: (_ for _ in ()).throw(KeyError(name))
    notebook = mk.notebook("01_intensite", "intensity", "Intensité", "7 h", commit="abc1234", colab=True)
    with pytest.raises(RuntimeError, match="Jeton Kaggle absent"):
        execute(notebook, stop=2)


def test_adaptation_continue_trouve_best(kaggle):
    import torch
    notebook = mk.continual_notebook(commit="abc1234")
    with pytest.raises(FileNotFoundError, match="best.pt de la baseline introuvable"):
        execute(notebook, stop=4)
    run = kaggle / "input" / "rq3-baseline-fp32-pli0" / "runs" / "unet3d_baseline_pli0"
    run.mkdir(parents=True)
    torch.save(dict(model={"w": torch.ones(2)}, state=dict(iteration=40000, best_dice=0.5)), run / "best.pt")
    namespace = execute(notebook, stop=4)
    assert namespace["CHECKPOINT"] == run / "best.pt"
    torch.save(dict(model={"w": torch.tensor([float("nan")])}, state={}), run / "best.pt")
    with pytest.raises(AssertionError, match="poids non finis"):
        execute(notebook, stop=4)


def test_colab_adaptation_continue(colab):
    import torch
    root, _ = colab
    run = root / "input" / "baseline" / "runs" / "unet3d_baseline_pli0"  # sortie Kaggle déjà téléchargée
    run.mkdir(parents=True)
    torch.save(dict(model={"w": torch.ones(2)}, state=dict(iteration=40000)), run / "best.pt")
    namespace = execute(mk.continual_notebook(commit="abc1234", colab=True), stop=8)
    assert namespace["CHECKPOINT"] == run / "best.pt"
    assert namespace["OUTPUT"] == root / "drive" / "MyDrive" / "PROJET_AVC" / "rq3-adaptation-continue-pli0" / "adaptation_continue"


def test_commits_acceptes_par_ordre_de_preference(kaggle):
    """Deux versions du code présentes : la première acceptée l'emporte ; une seule acceptée suffit."""
    second = kaggle / "input" / "autre-version"
    second.mkdir()
    with zipfile.ZipFile(next((kaggle / "input" / "projet-avc-code").glob("*.zip"))) as source,             zipfile.ZipFile(second / "projet_avc_code.zip", "w") as target:
        for item in source.infolist():
            data = source.read(item)
            target.writestr(item, json.dumps({"commit": "def5678", "message": "b"}) if item.filename == "VERSION.json" else data)
    namespace = execute(mk.notebook("01_intensite", "intensity", "I", "7 h", commit=("def5678", "abc1234")), stop=2)
    assert namespace["VERSION"]["commit"] == "def5678"
    namespace = execute(mk.notebook("01_intensite", "intensity", "I", "7 h", commit=("0000000", "abc1234")), stop=2)
    assert namespace["VERSION"]["commit"] == "abc1234"


def test_version_du_code_epinglee():
    metadata = mk.kernel_metadata("01_intensite", "rq3-intensity-pli0", code_version=7)
    assert metadata["dataset_sources"] == ["fayfahachicha/atlas-c1-processed", "fayfahachicha/projet-avc-code/7"]


# --- Reprise sur Kaggle de DANN v2 coupée sur Colab ---

FAKE_RESUME_TRAIN = '''
import argparse, json, os, sys
from pathlib import Path
import torch
from adaptation.dann import ganin_lambda
parser = argparse.ArgumentParser()
for name in ("--technique", "--fold", "--seed", "--data-root", "--output", "--num-workers", "--validate-every",
             "--log-every"):
    parser.add_argument(name)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()
assert args.resume and args.num_workers == "4"
run = Path(args.output) / f"unet3d_{args.technique}_pli{args.fold}"
last = torch.load(run / "last.pt", weights_only=False)
config = json.loads(json.dumps(last["config"]))
config.update(data_root=args.data_root, output=args.output, splits="/ailleurs/splits.json")
config["dann_weight"] = float(os.environ.get("FAKE_POIDS", config["dann_weight"]))
(run / "config.json").write_text(json.dumps(config))
it = last["state"]["iteration"]
print(f"Reprise \\u00e0 l'it\\u00e9ration {it} (4.70 h) ; indices et \\u00e9tats al\\u00e9atoires restaur\\u00e9s")
k = (it // 100 + 1) * 100
progress = (k - 1) / 40000
lam = ganin_lambda(progress, 5.0, 0.05) + float(os.environ.get("FAKE_ECART", "0"))
print({'progression': k, 'heures': 4.72, 's_par_iteration': 0.36, 'perte': 0.69, 'lr': f"{1e-3 * (1 - progress) ** 0.9:.2e}",
       'heures_restantes': 0.4, 'perte_segmentation': 0.35, 'perte_centre': 3.38, 'lambda': round(lam, 4)}, flush=True)
(run / "resume.json").write_text(json.dumps({"run": run.name, "technique": "dann"}))
'''


@pytest.fixture
def reprise(kaggle, monkeypatch):
    """Code avec le vrai adaptation/dann.py, checkpoints de la run coupée (last.pt à 35 950) en entrée."""
    import hashlib
    monkeypatch.setattr(sys, "path", list(sys.path))  # le notebook ajoute le code extrait à sys.path
    for key in ("RQ3_RUNS", "RQ3_PLATFORM"):  # posées par les notebooks Colab des autres tests
        monkeypatch.delenv(key, raising=False)
    import torch
    from dataclasses import asdict
    from adaptation.train import RunConfig
    repo = pathlib.Path(mk.__file__).resolve().parents[2]
    code = next((kaggle / "input" / "projet-avc-code").glob("*.zip"))
    with zipfile.ZipFile(code) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    files["adaptation/train.py"] = FAKE_RESUME_TRAIN.encode()
    files["adaptation/dann.py"] = (repo / "adaptation" / "dann.py").read_bytes()
    with zipfile.ZipFile(code, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    folder = kaggle / "input" / mk.RESUME["checkpoints"]
    folder.mkdir()
    config = asdict(RunConfig(technique="dann", fold=0, num_workers=4))
    it = 35950
    state = dict(iteration=it, elapsed=4.7 * 3600, best_dice=0.4362, best_iteration=24000)
    model = {"poids": torch.ones(3)}
    torch.save(dict(model=model, optimizer={}, scaler={}, state=state, config=config,
                    reprise=dict(rng_lots=[b"x"] * 4, num_workers=4, lr=1e-3 * (1 - (it - 1) / 40000) ** 0.9,
                                 volumes_consommes=2 * it)), folder / "last.pt")
    torch.save(dict(model=model, optimizer={}, scaler={}, state=dict(iteration=24000, elapsed=3.2 * 3600,
                    best_dice=0.4362, best_iteration=24000), config=config, reprise=None), folder / "best.pt")
    (folder / "config.json").write_text(json.dumps(config, default=str))
    rows = "".join(f"{i},0.1,1e-04,0.9,0.7,0.36,3.38,0.41\n" for i in range(2000, 36001, 2000))
    (folder / "journal_entrainement.csv").write_text(
        "iteration,heures,lr,lambda,perte,perte_segmentation,perte_centre,dice_validation\n" + rows)
    (folder / "sha256.txt").write_text("".join(
        f"{hashlib.sha256((folder / n).read_bytes()).hexdigest()}  {n}\n" for n in ("last.pt", "best.pt")))
    return kaggle


def test_reprise_dann_controlee(reprise, capsys):
    namespace = execute(mk.resume_notebook("abc1234"))
    run = reprise / "working" / "runs" / "unet3d_dann_pli0"
    log = (run / "entrainement.log").read_text(encoding="utf-8")
    assert "CONTRÔLE DE REPRISE : OK" in log
    assert namespace["EXPECTED"]["premiere_ligne"] == 36000 and namespace["EXPECTED"]["sampler_start"] == 71900
    journal = (run / "journal_entrainement.csv").read_text().splitlines()
    assert journal[-1].startswith("34000,") and len(journal) == 18  # 36 000 refaite après la reprise
    assert len((run / "journal_entrainement_avant_reprise.csv").read_text().splitlines()) == 19
    assert (run / "best.pt").is_file() and (run / "resume.json").is_file()
    assert "sha256 vérifiés" in capsys.readouterr().out


@pytest.mark.parametrize("variable, valeur", [("FAKE_ECART", "0.01"), ("FAKE_POIDS", "1.0")])
def test_reprise_dann_inexacte_arretee(reprise, monkeypatch, variable, valeur):
    monkeypatch.setenv(variable, valeur)
    with pytest.raises(RuntimeError, match="REPRISE INEXACTE"):
        execute(mk.resume_notebook("abc1234"))


def test_reprise_dann_checkpoint_altere(reprise):
    last = reprise / "input" / mk.RESUME["checkpoints"] / "last.pt"
    last.write_bytes(last.read_bytes() + b"0")
    with pytest.raises(AssertionError, match="last.pt altéré"):
        execute(mk.resume_notebook("abc1234"))


def test_reprise_dann_metadonnees():
    metadata = mk.resume_metadata()
    assert metadata["id"] == "fayfahachicha/rq3-dann-v2-reprise-pli0" and metadata["code_file"] == "08_dann_reprise.ipynb"
    assert metadata["dataset_sources"] == ["fayfahachicha/atlas-c1-processed", "fayfahachicha/projet-avc-code/8",
                                           "fayfahachicha/rq3-dann-v2-checkpoints-pli0"]
    assert metadata["is_private"] and metadata["enable_gpu"] and mk.RESUME["commit"] == "7449c49"
