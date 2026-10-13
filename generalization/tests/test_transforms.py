"""Transformations : invariants communs puis comportement propre à chacune, sur volumes synthétiques."""
from pathlib import Path

import numpy as np
import pytest
import torch
import torchio as tio

from generalization.contrast import subject_ratio
from generalization.transforms import (FDA, GIN, INTENSITY_NAMES, REGISTRY, HistogramMatching, MonaiAdapter,
                                       RandConv, RandomBrainBiasField, RandomBrainContrast, RandomBrainGamma,
                                       RandomBrainNoise, RandomLowResolution, build_transforms, fda_swap)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
KINDS = ["iso", "aniso", "r029"]


@pytest.fixture
def make(other_site, reference_quantiles):
    """Construit une transformation par son nom, avec ses paramètres par défaut et p = 1."""
    def factory(name, **kwargs):
        kwargs.setdefault("p", 1.0)
        if name == "fda":
            kwargs.setdefault("references", [subject["t1"].data[0] for subject in other_site])
            kwargs.setdefault("beta", (0.02, 0.03))
        if name == "histogram_matching":
            kwargs.setdefault("reference_quantiles", reference_quantiles)
        return REGISTRY[name](**kwargs)
    return factory


def apply(transform, subject, seed=0):
    torch.manual_seed(seed)
    return transform(subject)


def brain_values(subject):
    return subject["t1"].data[subject["brain"].data.bool()]


# --- Invariants communs à toutes les transformations d'intensité ---

@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", INTENSITY_NAMES)
def test_invariants(make, subjects, name, kind):
    subject = subjects[kind]
    before = subject["t1"].data.clone()
    out = apply(make(name), subject)
    # Règle essentielle : le masque de lésion (et celui du cerveau) ne change jamais.
    assert torch.equal(out["mask"].data, subject["mask"].data)
    assert torch.equal(out["brain"].data, subject["brain"].data)
    assert out["t1"].shape == subject["t1"].shape
    assert np.array_equal(out["t1"].affine, subject["t1"].affine)
    assert np.array_equal(out["mask"].affine, subject["mask"].affine)
    assert torch.isfinite(out["t1"].data).all()
    assert out["t1"].data.dtype == torch.float32
    # Le fond reste intact, l'image change dans le cerveau, et le sujet d'entrée n'est pas modifié.
    outside = ~subject["brain"].data.bool()
    assert torch.equal(out["t1"].data[outside], before[outside])
    assert not torch.equal(out["t1"].data, before)
    assert torch.equal(subject["t1"].data, before)


@pytest.mark.parametrize("name", INTENSITY_NAMES)
def test_reproductible_avec_graine(make, subjects, name):
    transform = make(name)
    first = apply(transform, subjects["iso"], seed=7)["t1"].data
    assert torch.equal(first, apply(transform, subjects["iso"], seed=7)["t1"].data)
    if name != "histogram_matching":  # déterministe par construction
        assert not torch.equal(first, apply(transform, subjects["iso"], seed=8)["t1"].data)


@pytest.mark.parametrize("name", INTENSITY_NAMES)
def test_probabilite_nulle(make, subjects, name):
    out = apply(make(name, p=0.0), subjects["iso"])
    assert torch.equal(out["t1"].data, subjects["iso"]["t1"].data)


@pytest.mark.parametrize("name", [n for n in INTENSITY_NAMES if n not in ("noise", "randconv", "gin")])
@pytest.mark.parametrize("kind", ["iso", "r014"])
def test_lesion_reste_plus_sombre_que_son_voisinage(make, subjects, name, kind):
    # Le bruit et le style sont suivis par visual_check.py (ils changent la texture par nature).
    transform = make(name)
    for seed in range(5):
        assert subject_ratio(apply(transform, subjects[kind], seed)) < 0.99


def test_sans_masque_du_cerveau(subjects):
    # Sans clé « brain », le cerveau est estimé par les voxels différents du fond.
    subject = tio.Subject(t1=subjects["iso"]["t1"], mask=subjects["iso"]["mask"])
    out = apply(RandomBrainGamma(p=1.0), subject)
    reference = apply(RandomBrainGamma(p=1.0), subjects["iso"])
    assert torch.allclose(out["t1"].data, reference["t1"].data, atol=1e-5)


# --- Augmentation d'intensité ---

def test_gamma_monotone_avec_valeurs_negatives(subjects):
    subject = subjects["r029"]
    assert (subject["t1"].data < 0).any()
    before = brain_values(subject)
    for gamma in (0.7, 1.85):
        after = brain_values(apply(RandomBrainGamma(gamma=(gamma, gamma), p=1.0), subject))
        order = before.argsort()
        assert (after[order].diff() >= -1e-5).all()
    identity = apply(RandomBrainGamma(gamma=(1.0, 1.0), p=1.0), subject)
    assert torch.allclose(identity["t1"].data, subject["t1"].data, atol=1e-5)


def test_gamma_deplace_la_mediane(subjects):
    before = brain_values(subjects["iso"]).median()
    darker = brain_values(apply(RandomBrainGamma(gamma=(1.85, 1.85), p=1.0), subjects["iso"])).median()
    brighter = brain_values(apply(RandomBrainGamma(gamma=(0.7, 0.7), p=1.0), subjects["iso"])).median()
    assert darker < before < brighter


def test_contraste_change_l_ecart_type_pas_la_moyenne(subjects):
    after = brain_values(apply(RandomBrainContrast(factor=(1.5, 1.5), p=1.0), subjects["iso"]))
    assert float(after.std()) == pytest.approx(1.5, rel=1e-3)
    assert float(after.mean()) == pytest.approx(0.0, abs=1e-3)


def test_bruit_a_l_ecart_type_demande(subjects):
    out = apply(RandomBrainNoise(std=(0.1, 0.1), p=1.0), subjects["iso"])
    noise = brain_values(out) - brain_values(subjects["iso"])
    assert float(noise.std()) == pytest.approx(0.1, rel=0.05)


def test_biais_de_champ_lisse_et_centre(subjects):
    subject = subjects["iso"]
    data, brain = subject["t1"].data, subject["brain"].data.bool()
    out = apply(RandomBrainBiasField(coefficients=0.3, p=1.0), subject)["t1"].data
    low = data[brain].min()
    keep = brain & (data - low > 0.5)
    log_field = torch.zeros_like(data)
    log_field[keep] = ((out[keep] - low) / (data[keep] - low)).log()
    assert abs(float(log_field[keep].mean())) < 0.05
    assert 0.02 < float(log_field[keep].std()) < 0.6
    # Lisse : deux voxels voisins du cerveau voient presque le même champ.
    both = keep[:, 1:] & keep[:, :-1]
    assert float((log_field[:, 1:] - log_field[:, :-1])[both].abs().max()) < 0.05


@pytest.mark.parametrize("kind", ["iso", "aniso"])
def test_basse_resolution_retire_les_hautes_frequences(subjects, kind):
    subject = subjects[kind]
    transform = RandomLowResolution(isotropic_mm=(1.5, 1.5), axis_mm=(3.0, 3.0), p=1.0)

    def roughness(item):
        data, brain = item["t1"].data, item["brain"].data.bool()
        return float((data[:, 1:] - data[:, :-1])[brain[:, 1:] & brain[:, :-1]].abs().mean()
                     + (data[:, :, 1:] - data[:, :, :-1])[brain[:, :, 1:] & brain[:, :, :-1]].abs().mean()
                     + (data[..., 1:] - data[..., :-1])[brain[..., 1:] & brain[..., :-1]].abs().mean())

    for seed in range(4):  # tirages isotropes et sur un seul axe
        assert roughness(apply(transform, subject, seed)) < 0.95 * roughness(subject)


def test_basse_resolution_sans_effet_si_deja_moins_resolue(subjects):
    transform = RandomLowResolution(isotropic_mm=(0.5, 1.0), axis_mm=(0.5, 1.0), p=1.0)
    out = apply(transform, subjects["iso"])
    assert torch.equal(out["t1"].data, subjects["iso"]["t1"].data)


# --- Randomisation du style ---

@pytest.mark.parametrize("cls", [RandConv, GIN])
def test_style_garde_les_statistiques_et_le_signe_du_contraste(subjects, cls):
    before = brain_values(subjects["iso"])
    for seed in range(6):
        after = brain_values(apply(cls(p=1.0), subjects["iso"], seed))
        assert float(after.mean()) == pytest.approx(float(before.mean()), abs=1e-2)
        assert float(after.std()) == pytest.approx(float(before.std()), rel=1e-2)
        assert float(torch.corrcoef(torch.stack([before, after]))[0, 1]) > 0


def test_style_melange_nul_rend_l_original(subjects):
    for cls in (RandConv, GIN):
        out = apply(cls(mix=(0.0, 0.0), p=1.0), subjects["iso"])
        assert torch.allclose(out["t1"].data, subjects["iso"]["t1"].data, atol=1e-4)


# --- FDA ---

def test_fda_beta_nul_rend_l_image_d_origine(make, subjects, other_site):
    source, target = subjects["iso"]["t1"].data[0], other_site[0]["t1"].data[0]
    target = torch.nn.functional.interpolate(target[None, None], size=source.shape)[0, 0]
    assert torch.equal(fda_swap(source, target, 0.0), source)
    out = apply(make("fda", beta=(0.0, 0.0)), subjects["iso"])
    assert torch.equal(out["t1"].data, subjects["iso"]["t1"].data)


def test_fda_garde_la_phase_et_prend_l_amplitude_basse_frequence():
    generator = torch.Generator().manual_seed(0)
    source, target = torch.rand(24, 30, 20, generator=generator), torch.rand(24, 30, 20, generator=generator) * 3
    out = fda_swap(source, target, 0.1)  # demi-côté 2
    spectrum = lambda volume: torch.fft.fftshift(torch.fft.fftn(volume.double()))  # noqa: E731
    cube = tuple(slice(n // 2 - 2, n // 2 + 3) for n in source.shape)
    assert torch.allclose(spectrum(out).abs()[cube], spectrum(target).abs()[cube], rtol=1e-3, atol=1e-2)
    outside = torch.ones(source.shape, dtype=torch.bool)
    outside[cube] = False
    assert torch.allclose(spectrum(out).abs()[outside], spectrum(source).abs()[outside], rtol=1e-3, atol=1e-2)
    phase_shift = (spectrum(out) * spectrum(source).conj()).angle()
    assert float(phase_shift.abs().max()) < 1e-2


def test_fda_avec_soi_meme_est_l_identite(subjects):
    source = subjects["iso"]["t1"].data[0]
    assert torch.allclose(fda_swap(source, source, 0.05), source, atol=1e-3)


def test_fda_sans_reference_refuse():
    with pytest.raises(ValueError):
        FDA(references=[])


# --- Histogram matching ---

def quantile_distance(values, reference_quantiles):
    levels = torch.linspace(0, 1, len(reference_quantiles))
    return float((torch.quantile(values, levels) - reference_quantiles).abs().mean())


# R029 n'a qu'une trentaine de niveaux de gris : les voxels de même valeur restent égaux après
# l'appariement, l'histogramme ne peut donc se rapprocher qu'en partie de la référence.
@pytest.mark.parametrize("kind, remaining", [("iso", 0.2), ("r029", 0.6)])
def test_histogram_matching_rapproche_les_histogrammes(make, subjects, reference_quantiles, kind, remaining):
    before = quantile_distance(brain_values(subjects[kind]), reference_quantiles)
    after = quantile_distance(brain_values(apply(make("histogram_matching"), subjects[kind])), reference_quantiles)
    assert before > 0.02
    assert after < remaining * before


def test_histogram_matching_vers_soi_meme(subjects):
    values = brain_values(subjects["iso"])
    own = torch.quantile(values, torch.linspace(0, 1, 1001))
    out = apply(HistogramMatching(own), subjects["iso"])
    # Identité à l'interpolation près : l'écart se concentre dans les queues extrêmes (1 % des voxels).
    difference = (brain_values(out) - values).abs()
    low, high = torch.quantile(values, torch.tensor([0.01, 0.99]))
    assert float(difference.mean()) < 0.005
    assert float(difference[(values > low) & (values < high)].max()) < 0.05


# --- Construction depuis les fichiers YAML ---

@pytest.mark.parametrize("config", sorted(path.name for path in CONFIGS.glob("*.yaml")))
def test_chaque_config_se_construit_et_s_applique(dataset_root, subjects, config):
    transform = build_transforms(CONFIGS / config, data_root=dataset_root)
    spatial = config == "bigaug.yaml"
    for seed in range(3):
        out = apply(transform, subjects["iso"], seed)
        assert out["t1"].shape == subjects["iso"]["t1"].shape
        assert torch.isfinite(out["t1"].data).all()
        assert set(out["mask"].data.unique().tolist()) <= {0, 1}
        if not spatial:
            assert torch.equal(out["mask"].data, subjects["iso"]["mask"].data)


@pytest.mark.parametrize("config", ["intensity.yaml", "fda.yaml", "histogram_matching.yaml"])
def test_serialisable_pour_les_dataloader(dataset_root, subjects, config):
    # Sous Windows, les processus des DataLoader reçoivent la transformation par pickle.
    import pickle
    transform = build_transforms(CONFIGS / config, data_root=dataset_root)
    clone = pickle.loads(pickle.dumps(transform))
    assert torch.equal(apply(clone, subjects["iso"])["t1"].data, apply(transform, subjects["iso"])["t1"].data)


def test_toutes_les_probabilites_a_un(dataset_root, subjects):
    # Chaque transformation des configs, forcée à p = 1, laisse le masque intact.
    for path in sorted(CONFIGS.glob("*.yaml")):
        for entry in build_config_entries(path):
            if entry["name"] in ("affine", "elastic", "flip"):
                continue
            transform = build_transforms({"transforms": [dict(entry, p=1.0, enabled=True)]}, data_root=dataset_root)
            out = apply(transform, subjects["iso"])
            assert torch.equal(out["mask"].data, subjects["iso"]["mask"].data), (path.name, entry["name"])


def build_config_entries(path):
    from generalization.transforms import load_config
    return load_config(path).get("transforms") or []


def test_build_desactivation_combinaison_et_erreurs(dataset_root):
    gamma = {"name": "gamma", "p": 1.0, "gamma": [0.7, 1.85]}
    assert len(build_transforms({"transforms": [gamma, dict(gamma, enabled=False)]})) == 1
    combined = build_transforms([CONFIGS / "intensity.yaml", CONFIGS / "style_gin.yaml"])
    assert len(combined) == 7
    assert len(build_transforms(CONFIGS / "baseline.yaml")) == 0
    with pytest.raises(KeyError, match="inconnue"):
        build_transforms({"transforms": [{"name": "inexistante"}]})
    # Le centre de référence n'est jamais codé en dur : il doit venir de la config.
    with pytest.raises(ValueError, match="reference_sites"):
        build_transforms({"transforms": [{"name": "fda"}]}, data_root=dataset_root)


def test_cache_de_l_histogramme_de_reference(dataset_root, tmp_path):
    entry = {"name": "histogram_matching", "reference_sites": ["SOOP"], "n_references": 2,
             "reference_file": str(tmp_path / "ref.pt")}
    first = build_transforms({"transforms": [entry]}, data_root=dataset_root)
    assert (tmp_path / "ref.pt").is_file()
    second = build_transforms({"transforms": [entry]}, data_root=tmp_path / "plus_de_donnees")
    assert torch.equal(first[0].reference_quantiles, second[0].reference_quantiles)


# --- Adaptateur MONAI ---

def test_adaptateur_monai(subjects):
    from monai.data import MetaTensor
    subject = subjects["iso"]
    affine = torch.as_tensor(subject["t1"].affine)
    data = {"image": MetaTensor(subject["t1"].data.clone(), affine=affine, meta={"filename_or_obj": "x.nii.gz"}),
            "label": MetaTensor(subject["mask"].data.clone(), affine=affine)}
    transform = build_transforms(CONFIGS / "intensity.yaml", monai=True)
    assert isinstance(transform, MonaiAdapter)
    torch.manual_seed(3)
    out = transform(data)
    assert isinstance(out["image"], MetaTensor) and out["image"].meta["filename_or_obj"] == "x.nii.gz"
    assert out["image"].shape == data["image"].shape and out["image"].dtype == torch.float32
    assert torch.equal(out["label"].as_tensor(), subject["mask"].data)
    assert out["label"].dtype == torch.uint8
    assert not torch.equal(out["image"].as_tensor(), subject["t1"].data)
    assert torch.equal(data["image"].as_tensor(), subject["t1"].data)  # entrée intacte
    # Identique à l'appel TorchIO sans clé « brain » (cerveau estimé depuis le fond).
    torch.manual_seed(3)
    expected = build_transforms(CONFIGS / "intensity.yaml")(tio.Subject(t1=subject["t1"], mask=subject["mask"]))
    assert torch.allclose(out["image"].as_tensor(), expected["t1"].data, atol=1e-5)


def test_adaptateur_monai_dans_un_compose(subjects):
    from monai.transforms import Compose, RandFlipd
    subject = subjects["iso"]
    data = {"image": subject["t1"].data.clone(), "label": subject["mask"].data.clone(),
            "brain": subject["brain"].data.clone()}
    pipeline = Compose([RandFlipd(keys=["image", "label", "brain"], prob=1.0, spatial_axis=0),
                        build_transforms(CONFIGS / "style_gin.yaml", monai=True)])
    out = pipeline(data)
    assert torch.equal(torch.as_tensor(out["label"]), subject["mask"].data.flip(1))
    assert torch.isfinite(torch.as_tensor(out["image"])).all()
