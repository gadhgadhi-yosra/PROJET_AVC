"""Chargement des données brutes : pièges relevés pendant l'exploration, sur volumes synthétiques."""
import warnings

import nibabel as nib
import numpy as np
import pytest
import torch

from generalization.contrast import subject_ratio
from generalization.data_raw import RawAtlasDataset, background_value, list_subjects, load_subject


def world_centroid(label_map):
    voxels = torch.nonzero(label_map.data[0]).float().mean(dim=0).numpy()
    return nib.affines.apply_affine(label_map.affine, voxels)


def test_list_exclut_r032_par_defaut(dataset_root):
    assert "R032" not in set(list_subjects(dataset_root)["site"])
    assert "R032" in set(list_subjects(dataset_root, exclude_sites=())["site"])


def test_list_colonnes_et_filtre_par_centre(dataset_root):
    table = list_subjects(dataset_root, sites=["SOOP"])
    assert set(table["site"]) == {"SOOP"} and len(table) == 3
    for column in ("subject", "site", "t1_path", "mask_path", "metadata_path", "known_empty_mask"):
        assert column in table


def test_list_dossier_absent(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_subjects(tmp_path / "absent")


def test_metadonnees(records):
    r009, soop, vide = records.loc["sub-r009s001"], records.loc["sub-soop0001"], records.loc["sub-soop0003"]
    assert r009["days_post_stroke"] == 426 and r009["chronicity"] and r009["atlas2_dataset"] == "Training"
    # SOOP : pas de délai, drapeau codé '1.0'.
    assert np.isnan(soop["days_post_stroke"]) and soop["chronicity"] and not soop["metadata_empty"]
    assert vide["metadata_empty"] and np.isnan(vide["days_post_stroke"])
    assert not records.loc["sub-r014s001", "chronicity"]


@pytest.mark.parametrize("name", ["sub-r009s001", "sub-soop0001", "sub-soop0002", "sub-r029s001", "sub-r014s001"])
def test_ras_1mm_masque_binaire(records, name):
    subject = load_subject(records.loc[name])
    for key in ("t1", "mask", "brain"):
        assert subject[key].orientation == ("R", "A", "S")
        assert np.allclose(subject[key].spacing, 1.0, atol=1e-4)
        assert subject[key].shape == subject["t1"].shape
        assert np.allclose(subject[key].affine, subject["t1"].affine)
    assert subject["mask"].data.dtype == torch.uint8
    assert set(subject["mask"].data.unique().tolist()) == {0, 1}
    assert torch.isfinite(subject["t1"].data).all()


@pytest.mark.parametrize("name", ["sub-r009s001", "sub-soop0002"])
def test_reorientation_garde_la_lesion_a_droite(records, name):
    # La lésion synthétique est à droite (x > 0), devant et au-dessus du centre, quel que soit
    # l'ordre de stockage (LAS pour R009, PSR pour le second sujet SOOP).
    subject = load_subject(records.loc[name])
    lesion, brain = world_centroid(subject["mask"]), world_centroid(subject["brain"])
    assert lesion[0] - brain[0] > 5 and lesion[1] - brain[1] > 2 and lesion[2] - brain[2] > 1


def test_masque_code_0_99999998(records):
    subject = load_subject(records.loc["sub-r014s001"])
    assert subject["mask"].data.sum() > 50 and not subject["empty_mask"]


def test_volume_de_lesion_conserve_au_reechantillonnage(records):
    native = load_subject(records.loc["sub-soop0001"], spacing=None)
    assert np.allclose(native["t1"].spacing, (0.5, 0.47, 0.47), atol=1e-3)
    volume_native = float(native["mask"].data.sum()) * 0.5 * 0.47 * 0.47
    volume_1mm = float(load_subject(records.loc["sub-soop0001"])["mask"].data.sum())
    assert volume_1mm == pytest.approx(volume_native, rel=0.1)


def test_r029_fond_non_nul_et_valeurs_negatives(records):
    raw = load_subject(records.loc["sub-r029s001"], spacing=None, normalize=False)
    data, brain = raw["t1"].data, raw["brain"].data.bool()
    assert background_value(data) == pytest.approx(0.0007, abs=1e-5)
    assert (data < 0).any()
    # Un masque « > 0 » prendrait tout le champ de vue ; les valeurs négatives ne sont pas du cerveau.
    assert 0.2 < brain.float().mean() < 0.6
    assert not brain[data <= 0.0007].any()


@pytest.mark.parametrize("name", ["sub-r009s001", "sub-r029s001"])
def test_zscore_dans_le_cerveau(records, name):
    subject = load_subject(records.loc[name])
    data, brain = subject["t1"].data, subject["brain"].data.bool()
    assert float(data[brain].mean()) == pytest.approx(0, abs=1e-3)
    assert float(data[brain].std()) == pytest.approx(1, abs=1e-3)
    assert (data[~brain] == 0).all()
    # norm_mean / norm_std redonnent les intensités brutes.
    raw = load_subject(records.loc[name], normalize=False)["t1"].data
    assert torch.allclose(data[brain] * subject["norm_std"] + subject["norm_mean"], raw[brain], atol=1e-2, rtol=1e-3)


def test_masque_vide_signale(records):
    with pytest.warns(UserWarning, match="masque vide"):
        subject = load_subject(records.loc["sub-soop0003"])
    assert subject["empty_mask"]
    # Un masque vide déjà répertorié ne déclenche pas d'avertissement.
    known = dict(records.loc["sub-soop0003"], known_empty_mask=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert load_subject(known)["empty_mask"]


def test_grilles_differentes_refusees(records, tmp_path):
    record = dict(records.loc["sub-r009s001"])
    mask = nib.load(record["mask_path"])
    shifted = mask.affine.copy()
    shifted[0, 3] += 3
    record["mask_path"] = str(tmp_path / "mask.nii.gz")
    nib.save(nib.Nifti1Image(np.asarray(mask.dataobj), shifted), record["mask_path"])
    with pytest.raises(ValueError, match="grilles"):
        load_subject(record)


def test_rapport_lesion_tissu_voisin(subjects):
    # Lésions synthétiques construites à 0,70 (R009) et 0,91 (R014) de l'intensité locale.
    assert 0.6 < subject_ratio(subjects["iso"]) < 0.85
    assert 0.85 < subject_ratio(subjects["r014"]) < 0.97


def test_dataset(records):
    dataset = RawAtlasDataset(records[records["site"] == "R009"], transform=lambda subject: subject["site"])
    assert len(dataset) == 2 and dataset[1] == "R009"
