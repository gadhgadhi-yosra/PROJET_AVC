"""DANN sur un petit UNet MONAI 3D (32³, CPU)."""
import math

import pytest
import torch

from adaptation.dann import (CenterClassifier, DANNWrapper, GradientReversal, center_labels, dann_loss,
                             find_bottleneck, ganin_lambda)


def test_inversion_du_gradient():
    features = torch.randn(3, 4, requires_grad=True)
    layer = GradientReversal(0.7)
    out = layer(features)
    assert torch.equal(out, features)  # identité à l'aller
    upstream = torch.randn(3, 4)
    out.backward(upstream)
    assert torch.allclose(features.grad, -0.7 * upstream)


def test_planification_de_ganin():
    assert ganin_lambda(0) == 0
    assert ganin_lambda(1) == pytest.approx(2 / (1 + math.exp(-10)) - 1)
    values = [ganin_lambda(p / 10) for p in range(11)]
    assert all(a < b for a, b in zip(values, values[1:]))
    assert ganin_lambda(-1) == 0 and ganin_lambda(2) == ganin_lambda(1)


def test_formes_et_bottleneck(batch, small_unet):
    unet = small_unet()
    bottleneck = find_bottleneck(unet)
    seen = []
    bottleneck.register_forward_hook(lambda module, inputs, output: seen.append(output.shape))
    model = DANNWrapper(unet, n_centers=5)
    segmentation, logits = model(batch[0])
    assert segmentation.shape == batch[0].shape
    assert logits.shape == (2, 5)
    assert seen[0][1] == unet.channels[-1]  # le classifieur lit bien la sortie la plus profonde


def test_l_encodeur_recoit_le_gradient_oppose(batch, small_unet):
    # Même réseau, même lot : avec lambda = 1, le gradient de la perte du centre sur un paramètre
    # de l'encodeur est l'opposé de celui obtenu sans inversion ; le classifieur, lui, ne change pas.
    model = DANNWrapper(small_unet(), n_centers=3)
    labels = torch.tensor([0, 2])
    encoder = next(model.unet.model[0].parameters())
    head = model.classifier.head[-1].weight

    def gradients(lambda_):
        model.zero_grad()
        model.grl.lambda_ = lambda_
        torch.nn.functional.cross_entropy(model(batch[0])[1], labels).backward()
        return encoder.grad.clone(), head.grad.clone()

    model.eval()  # dropout désactivé : deux passages identiques
    reversed_encoder, reversed_head = gradients(1.0)
    plain_encoder, plain_head = gradients(-1.0)
    assert reversed_encoder.abs().sum() > 0
    assert torch.allclose(reversed_encoder, -plain_encoder, atol=1e-6)
    assert torch.allclose(reversed_head, plain_head, atol=1e-6)
    zero_encoder, _ = gradients(0.0)
    assert zero_encoder.abs().max() == 0


def test_un_pas_d_optimisation(batch, small_unet):
    image, lesion, _ = batch
    model = DANNWrapper(small_unet(), n_centers=4)
    lambda_ = model.set_progress(0.5)
    assert lambda_ == pytest.approx(ganin_lambda(0.5))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    segmentation, logits = model(image)
    labels = center_labels(["R009", "R059"], ["R001", "R009", "R059", "R071"])
    total, parts = dann_loss(segmentation, lesion, logits, labels)
    assert torch.isfinite(total) and set(parts) == {"segmentation", "centre", "total"}
    assert parts["total"] == pytest.approx(parts["segmentation"] + parts["centre"], rel=1e-5)
    total.backward()
    optimizer.step()
    changed = [not torch.equal(old, new.detach()) for old, new in zip(before, model.parameters())]
    assert all(changed)  # segmentation, encodeur et classifieur bougent


def test_poids_de_la_perte_du_centre(batch, small_unet):
    image, lesion, _ = batch
    model = DANNWrapper(small_unet(), n_centers=2).eval()
    segmentation, logits = model(image)
    labels = torch.tensor([0, 1])
    full, parts = dann_loss(segmentation, lesion, logits, labels, weight=1.0)
    half, _ = dann_loss(segmentation, lesion, logits, labels, weight=0.5)
    assert float((full - half).detach()) == pytest.approx(0.5 * parts["centre"], rel=1e-5)


def test_erreurs():
    with pytest.raises(ValueError, match="absents"):
        center_labels(["SOOP"], ["R009"])
    assert CenterClassifier(8, 3)(torch.randn(2, 8, 4, 4, 4)).shape == (2, 3)
