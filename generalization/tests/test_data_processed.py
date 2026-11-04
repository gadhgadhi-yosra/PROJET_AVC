"""Chargeur C1 (data_processed.py), sur un jeu prétraité synthétique écrit dans un dossier temporaire."""
import json
import pickle
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
import torchio as tio

from generalization.data_processed import (ProcessedFormatError, list_processed, load_processed,
                                           load_processed_subject)
from generalization.synthetic import synthetic_volume, write_processed_dataset, write_processed_subject
from generalization.transforms import build_transforms

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
pytestmark = pytest.mark.filterwarnings("ignore:.*masque vide:UserWarning")


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    """(racine, chemin de splits.json) d'un petit jeu prétraité, volumes de moins de 64³."""
    return write_processed_dataset(tmp_path_factory.mktemp("c1") / "processed")


@pytest.fixture
def copy(processed, tmp_path):
    """Copie modifiable du jeu, pour les cas d'erreur."""
    root, splits = processed
    shutil.copytree(root, tmp_path / "processed")
    shutil.copy(splits, tmp_path / "splits.json")
    return tmp_path / "processed", tmp_path / "splits.json"


def by_subject(subjects):
    return {subject["subject"]: subject for subject in subjects}


# --- Contenu des sujets ---

def test_sujets_d_entrainement(processed):
    root, splits = processed
    subjects = load_processed(root, splits=splits)
    assert sorted(by_subject(subjects)) == ["sub-r009s001", "sub-r009s002", "sub-r014s001", "sub-r014s002",
                                            "sub-r029s001"]
    for subject in subjects:
        assert isinstance(subject, tio.Subject)
        assert {"t1", "mask", "brain", "site"} <= set(subject.keys())
        assert subject["t1"].orientation == ("R", "A", "S")
        assert np.allclose(subject["t1"].spacing, 1.0)
        assert max(subject["t1"].spatial_shape) <= 64
        assert subject["t1"].data.dtype == torch.float32
        assert set(subject["mask"].data.unique().tolist()) <= {0, 1}
        brain = subject["brain"].data.bool()
        # z-score dans le cerveau (stocké en int16, pas de 0,001), fond à 0.
        assert abs(float(subject["t1"].data[brain].mean())) < 1e-3
        assert abs(float(subject["t1"].data[brain].std()) - 1) < 1e-3
        assert float(subject["t1"].data[~brain].abs().max()) == 0


def test_chargement_paresseux_et_serialisable(processed):
    root, splits = processed
    subject = load_processed(root, splits=splits)[0]
    assert not subject["t1"]._loaded
    clone = pickle.loads(pickle.dumps(subject))  # DataLoader sous Windows
    assert torch.equal(clone["t1"].data, subject["t1"].data)


def test_masque_seuille_a_0_5(processed):
    # sub-r014s001 : masque stocké en float64 à 0,99999998.
    root, splits = processed
    raw = np.asarray(nib.load(root / "sub-r014s001" / "mask.nii.gz").dataobj)
    assert raw.max() < 1 and raw.max() > 0.5
    subject = by_subject(load_processed(root, splits=splits))["sub-r014s001"]
    assert int(subject["mask"].data.sum()) == int((raw > 0.5).sum()) > 0


def test_le_cerveau_vient_de_brain_nii(processed):
    root, splits = processed
    subject = load_processed(root, splits=splits)[0]
    stored = np.asarray(nib.load(Path(subject["brain"].path)).dataobj).astype(bool)
    assert np.array_equal(subject["brain"].data[0].numpy(), stored)
    # Après z-score, environ la moitié du cerveau est négative : « > 0 » serait faux.
    assert not torch.equal(subject["t1"].data > 0, subject["brain"].data.bool())


def test_masque_vide_signale_sans_erreur(processed):
    root, splits = processed
    with pytest.warns(UserWarning, match="masque vide.*sub-r014s002"):
        subjects = by_subject(load_processed(root, splits=splits))
    assert subjects["sub-r014s002"]["empty_mask"] and int(subjects["sub-r014s002"]["mask"].data.sum()) == 0
    assert not subjects["sub-r009s001"]["empty_mask"]


def test_normalisation_reversible(processed):
    # norm_mean / norm_std (qc.json) : brut = z × norm_std + norm_mean, comme data_raw.
    root, splits = processed
    subject = load_processed(root, splits=splits)[0]
    qc = json.loads((root / subject["subject"] / "qc.json").read_text())
    assert (subject["norm_mean"], subject["norm_std"]) == (qc["raw_mean"], qc["raw_std"])


# --- Découpages ---

def test_decoupages_et_plis(processed):
    root, splits = processed
    names = lambda **kwargs: sorted(by_subject(load_processed(root, splits=splits, **kwargs)))  # noqa: E731
    assert names(split="id_test") == ["sub-r009s003"]
    assert names(split="ood_test") == ["sub-soop0001"]
    assert names(split="train", fold=0) == ["sub-r009s001", "sub-r009s002", "sub-r029s001"]
    assert names(split="val", fold=0) == ["sub-r014s001", "sub-r014s002"]
    assert names(split="id_test", fold=0) == ["sub-r009s003"]
    with pytest.raises(ValueError, match="exige un pli"):
        load_processed(root, split="val", splits=splits)
    with pytest.raises(ValueError, match="split inconnu"):
        load_processed(root, split="test", splits=splits)
    with pytest.raises(ValueError, match="Pli 7"):
        load_processed(root, fold=7, splits=splits)


def test_r032_exclu(copy):
    # Même ajouté au découpage, R032 n'est jamais chargé.
    root, splits_path = copy
    splits = json.loads(splits_path.read_text())
    case = "R032_sub-r032s005_ses-1"
    splits["cases"].append(dict(case_id=case, subject="sub-r032s005", site="R032", session="ses-1"))
    splits["train"].append(case)
    assert "R032" not in {subject["site"] for subject in load_processed(root, splits=splits)}


# --- Erreurs claires ---

def rewrite(root, subject, **kwargs):
    image, lesion, affine = synthetic_volume(fov_mm=(40.0, 44.0, 36.0), seed=99)
    if "spacing" in kwargs:
        affine = np.diag([*kwargs.pop("spacing"), 1.0])
    shutil.rmtree(root / subject)
    write_processed_subject(root, "R009", subject, image, lesion, affine, **kwargs)


def test_orientation_non_ras_refusee(copy):
    root, splits = copy
    rewrite(root, "sub-r009s001", orientation="LAS")
    with pytest.raises(ProcessedFormatError, match="sub-r009s001.*LAS au lieu de RAS"):
        load_processed(root, splits=splits)


def test_espacement_autre_que_1_mm_refuse(copy):
    root, splits = copy
    rewrite(root, "sub-r009s001", spacing=(1.0, 2.0, 1.0))
    with pytest.raises(ProcessedFormatError, match="au lieu de 1 mm"):
        load_processed(root, splits=splits)


def test_grilles_differentes_refusees(copy):
    root, splits = copy
    path = root / "sub-r009s001" / "brain.nii.gz"
    image = nib.load(path)
    affine = image.affine.copy()
    affine[0, 3] += 3
    nib.save(nib.Nifti1Image(np.asarray(image.dataobj), affine), path)
    with pytest.raises(ProcessedFormatError, match="brain.nii.gz n'est pas sur la grille"):
        load_processed(root, splits=splits)


def test_sujet_manquant(copy):
    root, splits = copy
    shutil.rmtree(root / "sub-r029s001")
    with pytest.raises(ProcessedFormatError, match="1 sujets.*sans volumes"):
        load_processed(root, splits=splits)
    with pytest.warns(UserWarning, match="écartés"):
        assert len(load_processed(root, splits=splits, on_missing="warn")) == 4


def test_table_absente(copy):
    root, splits = copy
    (root / "subjects.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="subjects.parquet"):
        load_processed(root, splits=splits)


# --- De bout en bout : load_processed -> build_transforms(monai=True) ---

def test_bout_en_bout_monai(processed):
    from monai.transforms import Compose, RandFlipd
    root, splits = processed
    subjects = load_processed(root, splits=splits, fold=0)
    transform = Compose([RandFlipd(keys=["image", "label", "brain"], prob=0.0),
                         build_transforms([CONFIGS / "intensity.yaml", CONFIGS / "style_gin.yaml"], monai=True)])
    torch.manual_seed(0)
    for subject in subjects:
        data = {"image": subject["t1"].data.clone(), "label": subject["mask"].data.clone(),
                "brain": subject["brain"].data.clone()}
        out = transform(data)
        image = torch.as_tensor(out["image"])
        assert image.shape == subject["t1"].data.shape and torch.isfinite(image).all()
        assert torch.equal(torch.as_tensor(out["label"]), subject["mask"].data)
        brain = subject["brain"].data.bool()
        assert torch.equal(image[~brain], subject["t1"].data[~brain])  # fond intact


@pytest.mark.parametrize("transform", ["fda", "histogram_matching"])
def test_references_prises_dans_les_donnees_pretraitees(processed, transform):
    root, splits = processed
    entry = {"name": transform, "reference_sites": ["R029"], "fold_reference_sites": ["R009", "R014"],
             "splits": str(splits), "n_references": 4, "p": 1.0}
    built = build_transforms({"transforms": [entry]}, data_root=root, fold=0, source="processed", monai=True)
    if transform == "fda":
        sources = built.transform[0].references.sources
        # Pli 0 : R009, sujets d'entraînement du pli seulement (pas sub-r009s003, test ID).
        assert sorted(source.record["subject"] for source in sources) == ["sub-r009s001", "sub-r009s002"]
    subject = load_processed(root, splits=splits, fold=0)[0]
    out = built({"image": subject["t1"].data.clone(), "label": subject["mask"].data.clone(),
                 "brain": subject["brain"].data.clone()})
    assert torch.isfinite(torch.as_tensor(out["image"])).all()
    with pytest.raises(ValueError, match="validation du pli 0"):
        build_transforms({"transforms": [dict(entry, fold_reference_sites=["R014", "R014"])]},
                         data_root=root, fold=0, source="processed")


def test_source_inconnue():
    with pytest.raises(ValueError, match="source inconnue"):
        build_transforms({"transforms": []}, source="nnunet")


# --- Scripts avec --source processed ---

def test_calibration_sur_donnees_pretraitees(tmp_path):
    # Les grandeurs reconstruites (brut = z × raw_std + raw_mean) sont celles du volume d'origine.
    from generalization.calibration import intensity_stats, native_spacing, processed_subject_stats
    image, lesion, affine = synthetic_volume(fov_mm=(40.0, 44.0, 36.0), seed=5, scale=500.0)
    write_processed_subject(tmp_path / "processed", "R009", "sub-r009s001", image, lesion, affine)
    qc = json.loads((tmp_path / "processed" / "sub-r009s001" / "qc.json").read_text())
    record = dict(site="R009", subject="sub-r009s001", session="ses-1", norm_mean=qc["raw_mean"],
                  norm_std=qc["raw_std"], native_voxel_min=1.0, native_voxel_max=1.0,
                  **{f"{name}_path": str(tmp_path / "processed" / "sub-r009s001" / f"{name}.nii.gz")
                     for name in ("t1", "mask", "brain")})
    measured = processed_subject_stats(record)
    expected = intensity_stats(record, image[image > 0], (1.0, 1.0, 1.0))
    for key in ("p50", "p99", "cv"):
        assert measured[key] == pytest.approx(expected[key], rel=1e-3), key
    # Sans données brutes, l'espacement natif est inconnu et signalé.
    import pandas as pd
    with pytest.warns(UserWarning, match="Espacement natif inconnu"):
        spaced = native_spacing(pd.DataFrame([record]), tmp_path / "pas_de_donnees_brutes")
    assert np.isnan(spaced["native_voxel_min"]).all()


def test_visual_check_sur_donnees_pretraitees_synthetiques(tmp_path):
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-m", "generalization.visual_check", "--synthetic", "--source",
                             "processed", "--sites", "R009", "R014", "--families", "intensite", "--draws", "1",
                             "--stat-draws", "1", "--out", str(tmp_path)],
                            capture_output=True, text=True, cwd=CONFIGS.parents[1])
    assert result.returncode == 0, result.stderr[-2000:]
    assert (tmp_path / "resume.csv").is_file()


# --- Vraies données C1, si elles sont sur la machine (jamais dans Git) ---

REAL_C1 = Path(__file__).resolve().parents[2] / "data" / "processed"


@pytest.mark.skipif(not (REAL_C1 / "subjects.parquet").is_file(), reason="données C1 absentes")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_donnees_c1_reelles():
    from generalization.calibration import load_splits
    splits = load_splits()
    for split in ("train", "id_test", "ood_test"):
        records = list_processed(REAL_C1, split, on_missing="warn")
        assert len(records) >= len(splits[split]) - 1, split  # R065_sub-r065s019 en échec côté C1
        assert set(records["case_id"]) <= set(splits[split])
    subject = load_processed_subject(records.iloc[0])
    brain = subject["brain"].data.bool()
    assert subject["t1"].orientation == ("R", "A", "S") and np.allclose(subject["t1"].spacing, 1.0)
    assert abs(float(subject["t1"].data[brain].mean())) < 1e-3
    assert float(subject["t1"].data[~brain].abs().max()) == 0


def test_sujet_isole(processed):
    root, splits = processed
    record = list_processed(root, splits=splits).iloc[0]
    assert load_processed_subject(record)["case_id"] == record["case_id"]
