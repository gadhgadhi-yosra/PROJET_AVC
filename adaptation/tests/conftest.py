"""Fixtures communes : petits UNet MONAI 3D sur CPU, graines fixées, aucune donnée patient."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def build_small_unet(norm="instance"):
    """UNet MONAI 3D de quelques milliers de paramètres ; `norm` au format MONAI."""
    from monai.networks.nets import UNet
    return UNet(spatial_dims=3, in_channels=1, out_channels=1, channels=(8, 16, 32), strides=(2, 2),
                num_res_units=1, norm=norm)


@pytest.fixture
def small_unet():
    """Fabrique de petits UNet : small_unet(norm)."""
    return build_small_unet


@pytest.fixture
def batch():
    """Deux volumes 32³ : image, masque de lésion (cube) et cerveau (sphère)."""
    grid = torch.stack(torch.meshgrid(*[torch.arange(32.0)] * 3, indexing="ij"))
    brain = ((grid - 15.5) ** 2).sum(0) < 14 ** 2
    lesion = torch.zeros(32, 32, 32, dtype=torch.bool)
    lesion[18:24, 10:16, 12:18] = True
    image = torch.randn(2, 1, 32, 32, 32) * 0.3 + brain.float()
    image[:, :, lesion] -= 1.0
    return image, lesion.expand(2, 1, 32, 32, 32).float(), brain.expand(2, 1, 32, 32, 32)


# --- Vraies données C1 (jamais dans Git) : patches de 64³ sur CPU ---

REAL_C1 = Path(__file__).resolve().parents[2] / "data" / "processed"
PATCH = 64


def crop_around(tensor, centre, size=PATCH):
    """Patch `size`³ centré sur `centre`, décalé pour rester dans le volume (forme 1 × X × Y × Z)."""
    starts = [min(max(int(c) - size // 2, 0), max(n - size, 0)) for c, n in zip(centre, tensor.shape[1:])]
    patch = tensor[:, starts[0]:starts[0] + size, starts[1]:starts[1] + size, starts[2]:starts[2] + size]
    if tuple(patch.shape[1:]) != (size,) * 3:
        raise ValueError(f"Volume trop petit pour un patch de {size}³ : {tuple(tensor.shape[1:])}")
    return patch


@pytest.fixture(scope="session")
def c1_patches():
    """3 sujets d'entraînement C1 de centres différents : patches 64³ centrés sur la lésion.

    Renvoie (images, lésions, cerveaux, centres), tenseurs N × 1 × 64 × 64 × 64.
    """
    if not (REAL_C1 / "subjects.parquet").is_file():
        pytest.skip("données C1 absentes")
    import warnings
    from generalization.data_processed import list_processed, load_processed_subject
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        records = list_processed(REAL_C1, "train", on_missing="warn", warn_empty=False)
    records = records[~records["empty_mask"]].drop_duplicates("site").head(3)
    images, lesions, brains = [], [], []
    for _, record in records.iterrows():
        subject = load_processed_subject(record)
        centre = subject["mask"].data[0].nonzero().float().mean(dim=0)
        images.append(crop_around(subject["t1"].data, centre))
        lesions.append(crop_around(subject["mask"].data, centre).float())
        brains.append(crop_around(subject["brain"].data, centre).bool())
    return torch.stack(images), torch.stack(lesions), torch.stack(brains), records["site"].tolist()
