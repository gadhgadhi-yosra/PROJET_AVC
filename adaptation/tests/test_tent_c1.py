"""TENT sur 3 vrais volumes C1 recadrés en patches de 64³, sur CPU (ignoré sans données C1)."""
import torch

from adaptation.tent import Tent, collect_params, configure_tent


def test_adaptation_sur_patches_c1(c1_patches, small_unet):
    images, _, brains, _ = c1_patches
    model = configure_tent(small_unet(("instance", {"affine": True})))
    norm_ids = {id(parameter) for parameter in collect_params(model)}
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    state = {key: value.clone() for key, value in model.state_dict().items()}
    tent = Tent(model, torch.optim.Adam(collect_params(model), lr=1e-2), steps=1)
    for _ in range(8):
        tent(images, brains)
    assert tent.history[-1] < tent.history[0]  # l'entropie dans le cerveau baisse
    for name, parameter in model.named_parameters():
        if id(parameter) not in norm_ids:
            assert torch.equal(before[name], parameter.detach()), name
    tent.reset()
    assert all(torch.equal(state[key], value) for key, value in model.state_dict().items())
