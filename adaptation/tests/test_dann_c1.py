"""DANN sur 3 vrais volumes C1 recadrés en patches de 64³, sur CPU (ignoré sans données C1)."""
import torch

from adaptation.dann import DANNWrapper, center_labels, dann_loss


def test_pas_d_optimisation_sur_patches_c1(c1_patches, small_unet):
    images, lesions, _, sites = c1_patches
    assert images.shape == (3, 1, 64, 64, 64) and len(set(sites)) == 3
    centers = sorted(sites)
    model = DANNWrapper(small_unet(), n_centers=len(centers))
    model.set_progress(0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    for _ in range(5):
        segmentation, logits = model(images)
        total, parts = dann_loss(segmentation, lesions, logits, center_labels(sites, centers))
        assert torch.isfinite(total)
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        history.append(parts)
    assert segmentation.shape == lesions.shape and logits.shape == (3, 3)
    # Cinq pas sur le même lot : la perte de segmentation baisse.
    assert history[-1]["segmentation"] < history[0]["segmentation"]
