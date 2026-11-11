"""TENT sur un petit UNet MONAI 3D (32³, CPU)."""
import pytest
import torch

from adaptation.tent import Tent, collect_params, configure_tent, voxel_entropy


NORMS = {"batch": "batch", "instance affine": ("instance", {"affine": True}), "group": ("group", {"num_groups": 1})}


@pytest.mark.parametrize("norm", list(NORMS))
def test_seuls_les_parametres_de_normalisation_sont_entrainables(norm, small_unet):
    model = configure_tent(small_unet(NORMS[norm]))
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert trainable == {id(parameter) for parameter in collect_params(model)} and trainable
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm3d):
            assert module.running_mean is None and not module.track_running_stats  # statistiques du lot
    assert model.training


def test_unet_monai_par_defaut_refuse(small_unet):
    # InstanceNorm3d sans paramètres affines : TENT n'aurait rien à adapter.
    with pytest.raises(ValueError, match="Aucune couche de normalisation affine"):
        configure_tent(small_unet("instance"))


@pytest.mark.parametrize("norm", list(NORMS))
def test_adaptation(batch, norm, small_unet):
    image, _, brain = batch
    model = configure_tent(small_unet(NORMS[norm]))
    norm_ids = {id(parameter) for parameter in collect_params(model)}
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    tent = Tent(model, torch.optim.Adam(collect_params(model), lr=1e-2), steps=1)
    for _ in range(10):
        tent(image, brain)
    # L'entropie baisse sur le lot synthétique.
    assert tent.history[-1] < tent.history[0]
    # Seuls les paramètres de normalisation ont bougé.
    for name, parameter in model.named_parameters():
        if id(parameter) not in norm_ids:
            assert torch.equal(before[name], parameter.detach()), name
    assert any(not torch.equal(before[name], parameter.detach())
               for name, parameter in model.named_parameters() if id(parameter) in norm_ids)


def test_reset_restaure_exactement(batch, small_unet):
    image, _, brain = batch
    model = configure_tent(small_unet("batch"))
    optimizer = torch.optim.Adam(collect_params(model), lr=1e-2)
    tent = Tent(model, optimizer, steps=3)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        reference = model(image)
    tent(image, brain)
    assert any(not torch.equal(state[key], value) for key, value in model.state_dict().items())
    tent.reset()
    assert all(torch.equal(state[key], value) for key, value in model.state_dict().items())
    assert optimizer.state_dict()["state"] == {} and tent.history == []
    with torch.no_grad():
        assert torch.equal(model(image), reference)


def test_entropie():
    certain = torch.full((1, 1, 4, 4, 4), 20.0)
    uncertain = torch.zeros(1, 1, 4, 4, 4)
    assert float(voxel_entropy(certain)) < 1e-6
    assert float(voxel_entropy(uncertain)) == pytest.approx(float(torch.log(torch.tensor(2.0))), rel=1e-5)
    brain = torch.zeros(1, 1, 4, 4, 4, dtype=torch.bool)
    brain[..., :2] = True
    mixed = torch.cat([uncertain[..., :2], certain[..., 2:]], dim=-1)
    assert float(voxel_entropy(mixed, brain)) == pytest.approx(float(torch.log(torch.tensor(2.0))), rel=1e-5)
    two_channels = torch.zeros(1, 2, 4, 4, 4)
    assert float(voxel_entropy(two_channels)) == pytest.approx(float(torch.log(torch.tensor(2.0))), rel=1e-5)
    with pytest.raises(ValueError, match="vide"):
        voxel_entropy(uncertain, torch.zeros_like(brain))
