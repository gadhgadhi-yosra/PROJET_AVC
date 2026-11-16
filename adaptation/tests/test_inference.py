"""Contrat C8 : prétraitement identique à S4, chargement des checkpoints, forme et volume du masque."""
import sys
from dataclasses import asdict
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from adaptation import inference
from adaptation.dann import DANNWrapper
from adaptation.train import RunConfig, build_network

REPO_ROOT = Path(__file__).resolve().parents[2]


def raw_t1(shape=(40, 44, 36), zooms=(1.2, 1.0, 0.9), flip=True, seed=0):
    """T1 « sans crâne » synthétique : ellipsoïde texturé sur fond 0, lésion hypo-intense, axes non RAS."""
    rng = np.random.default_rng(seed)
    grid = np.stack(np.meshgrid(*[np.linspace(-1, 1, n) for n in shape], indexing="ij"))
    brain = (grid ** 2).sum(0) < 0.8
    image = np.where(brain, 300 + 40 * rng.standard_normal(shape), 0).astype(np.float32)
    lesion = brain & (np.abs(grid[0] - 0.2) < 0.15) & (np.abs(grid[1]) < 0.15) & (np.abs(grid[2] + 0.1) < 0.2)
    image[lesion] -= 120
    affine = np.diag([*zooms, 1.0])
    if flip:  # orientation LAS : as_closest_canonical doit la remettre en RAS
        affine[0, 0] = -zooms[0]
        affine[0, 3] = zooms[0] * (shape[0] - 1)
    return nib.Nifti1Image(image, affine), nib.Nifti1Image(lesion.astype(np.uint8), affine)


@pytest.fixture(autouse=True)
def clear_cache():
    inference.load_segmenter.cache_clear()
    yield
    inference.load_segmenter.cache_clear()


def test_pretraitement_identique_a_s4(tmp_path):
    sys.path.insert(0, str(REPO_ROOT / "preprocessing"))
    try:
        import week4
    finally:
        sys.path.remove(str(REPO_ROOT / "preprocessing"))
    image, mask = raw_t1()
    nib.save(image, tmp_path / "t1.nii.gz")
    nib.save(mask, tmp_path / "mask.nii.gz")
    row = dict(site="R000", subject="sub-test", session="ses-1", t1_path=str(tmp_path / "t1.nii.gz"),
               mask_path=str(tmp_path / "mask.nii.gz"))
    (tmp_path / "s4").mkdir()
    week4.transform(row, tmp_path / "s4", resume=False, reserve_bytes=0)
    expected = nib.load(tmp_path / "s4" / "sub-test" / "t1.nii.gz")
    expected_brain = np.asarray(nib.load(tmp_path / "s4" / "sub-test" / "brain.nii.gz").dataobj) > 0

    prepared = inference.preprocess(inference.as_image(tmp_path / "t1.nii.gz"))
    assert prepared["t1"].shape[1:] == expected.shape
    np.testing.assert_allclose(prepared["t1"][0].numpy(), expected.get_fdata(dtype=np.float32), atol=1e-6)
    np.testing.assert_array_equal(prepared["brain"][0].numpy(), expected_brain)
    start = [s.start for s in prepared["crop"]]
    np.testing.assert_allclose(prepared["grid"].affine @ np.r_[start, 1], expected.affine[:, 3], atol=1e-5)


def test_entrees_acceptees():
    image, _ = raw_t1(flip=False)
    array = np.asarray(image.dataobj)
    assert inference.as_image(array).shape == array.shape
    assert inference.as_image(image) is image
    with pytest.raises(ValueError):
        inference.as_image(array[0])
    with pytest.raises(ValueError, match="cerveau vide"):
        inference.preprocess(nib.Nifti1Image(np.zeros((20, 20, 20), np.float32), np.eye(4)))


def small_config():
    return RunConfig(channels=(8, 16, 32), strides=(2, 2), num_res_units=1, patch=32, sw_batch=1)


def test_chargement_checkpoint_et_dann(tmp_path):
    cfg = small_config()
    plain = build_network(cfg)
    torch.save(dict(model=plain.state_dict(), config=asdict(cfg)), tmp_path / "plain.pt")
    network, loaded_cfg = inference.load_segmenter(str(tmp_path / "plain.pt"), "cpu")
    assert loaded_cfg.patch == 32 and not network.training
    for key, value in plain.state_dict().items():
        torch.testing.assert_close(network.state_dict()[key], value)

    dann_cfg = RunConfig(**{**asdict(cfg), "technique": "dann"})
    wrapper = DANNWrapper(build_network(dann_cfg), n_centers=3)
    torch.save(dict(model=wrapper.state_dict(), config={**asdict(dann_cfg), "inconnue": 1}), tmp_path / "dann.pt")
    network, _ = inference.load_segmenter(str(tmp_path / "dann.pt"), "cpu")
    for key, value in wrapper.unet.state_dict().items():
        torch.testing.assert_close(network.state_dict()[key], value)

    with pytest.raises(FileNotFoundError):
        inference.load_segmenter(str(tmp_path / "absent.pt"), "cpu")


class Constant(torch.nn.Module):
    """Réseau factice : logit constant (tout lésion si positif, rien si négatif)."""

    def __init__(self, value):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(float(value)))

    def forward(self, x):
        return torch.ones_like(x) * self.value


@pytest.mark.parametrize("flip", [False, True])
def test_masque_dans_la_grille_d_entree(monkeypatch, flip):
    image, _ = raw_t1(zooms=(1.0, 1.0, 1.0), flip=flip)
    monkeypatch.setattr(inference, "load_segmenter", lambda *a, **k: (Constant(10.0), small_config()))
    mask, volume_ml = inference.predict(image)
    brain = np.asarray(image.dataobj) > 0
    assert mask.shape == image.shape and mask.dtype == np.uint8
    # Grille de 1 mm : le masque « tout lésion » est exactement le cerveau, et le volume son nombre de voxels.
    np.testing.assert_array_equal(mask, brain.astype(np.uint8))
    assert volume_ml == pytest.approx(brain.sum() / 1000)

    monkeypatch.setattr(inference, "load_segmenter", lambda *a, **k: (Constant(-10.0), small_config()))
    mask, volume_ml = inference.predict(np.asarray(image.dataobj), affine=image.affine)
    assert volume_ml == 0 and not mask.any()


def test_prediction_bout_en_bout(tmp_path):
    cfg = small_config()
    torch.save(dict(model=build_network(cfg).state_dict(), config=asdict(cfg)), tmp_path / "best.pt")
    image, _ = raw_t1()
    mask, volume_ml = inference.predict(image, checkpoint=tmp_path / "best.pt", device="cpu")
    assert mask.shape == image.shape
    assert set(np.unique(mask)) <= {0, 1}
    assert 0 <= volume_ml <= (np.asarray(image.dataobj) > 0).sum() * 1.2 * 0.9 / 1000 + 1
