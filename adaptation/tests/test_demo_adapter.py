"""Adaptateur de la démo : masque NIfTI uint8 0/1, même forme et même affine que l'IRM d'entrée."""
from dataclasses import asdict

import nibabel as nib
import numpy as np
import pytest
import torch

from adaptation import demo_adapter, inference
from adaptation.train import RunConfig, build_network


class Constant(torch.nn.Module):
    """Réseau factice : logit constant."""

    def __init__(self, value):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(float(value)))

    def forward(self, x):
        return torch.ones_like(x) * self.value


def small_config():
    return RunConfig(channels=(8, 16, 32), strides=(2, 2), num_res_units=1, patch=32, sw_batch=1)


def synthetic_t1(path, shape=(40, 44, 36), zooms=(1.2, 1.0, 0.9)):
    """T1 « sans crâne » : ellipsoïde sur fond 0, orientation LAS et voxels anisotropes."""
    rng = np.random.default_rng(0)
    grid = np.stack(np.meshgrid(*[np.linspace(-1, 1, n) for n in shape], indexing="ij"))
    brain = (grid ** 2).sum(0) < 0.8
    data = np.where(brain, 300 + 40 * rng.standard_normal(shape), 0).astype(np.float32)
    affine = np.diag([-zooms[0], zooms[1], zooms[2], 1.0])
    affine[:3, 3] = [zooms[0] * (shape[0] - 1), -20.0, 5.0]
    nib.save(nib.Nifti1Image(data, affine), path)
    return brain


@pytest.fixture(autouse=True)
def clear_cache():
    inference.load_segmenter.cache_clear()
    yield
    inference.load_segmenter.cache_clear()


def check_mask(result, source):
    image, mask = nib.load(source), nib.load(result["mask_path"])
    assert set(result) == {"mask_path"}
    assert mask.shape == image.shape
    np.testing.assert_allclose(mask.affine, image.affine, atol=1e-5)
    data = np.asarray(mask.dataobj)
    assert data.dtype == np.uint8 and set(np.unique(data)) <= {0, 1}
    return data


def test_masque_tout_lesion(tmp_path, monkeypatch):
    source = tmp_path / "sub-test_T1w.nii.gz"
    synthetic_t1(source)
    monkeypatch.setenv("AVC_MASK_DIR", str(tmp_path / "masques"))
    monkeypatch.setattr(inference, "load_segmenter", lambda *a, **k: (Constant(10.0), small_config()))
    result = demo_adapter.predict(str(source))
    assert result["mask_path"] == str(tmp_path / "masques" / "sub-test_T1w_masque_c8.nii.gz")
    data = check_mask(result, source)
    assert data.any()


def test_modele_reel_et_dossier_temporaire(tmp_path, monkeypatch):
    cfg = small_config()
    torch.save(dict(model=build_network(cfg).state_dict(), config=asdict(cfg)), tmp_path / "best.pt")
    monkeypatch.setenv("PROJET_AVC_MODELE", str(tmp_path / "best.pt"))
    monkeypatch.delenv("AVC_MASK_DIR", raising=False)
    source = tmp_path / "irm.nii"
    synthetic_t1(source)
    result = demo_adapter.predict(str(source))
    assert result["mask_path"].endswith("irm_masque_c8.nii.gz")
    check_mask(result, source)


def test_nom_du_masque():
    assert demo_adapter.mask_name("/a/b/sub-r034_T1w.nii.gz") == "sub-r034_T1w_masque_c8.nii.gz"
    assert demo_adapter.mask_name("x.NII") == "x_masque_c8.nii.gz"
