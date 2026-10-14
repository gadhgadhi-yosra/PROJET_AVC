"""Fixtures communes : un petit jeu synthétique multi-centres, aucune donnée patient."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from generalization.data_raw import list_subjects, load_subject  # noqa: E402
from generalization.synthetic import write_synthetic_dataset  # noqa: E402
from generalization.transforms import fit_reference_quantiles  # noqa: E402


@pytest.fixture(scope="session")
def dataset_root(tmp_path_factory):
    return write_synthetic_dataset(tmp_path_factory.mktemp("atlas") / "ATLAS3_Training_Raw")


@pytest.fixture(scope="session")
def records(dataset_root):
    return list_subjects(dataset_root).set_index("subject", drop=False)


@pytest.fixture(scope="session")
def subjects(records):
    """Sujets normalisés : isotrope 1 mm, anisotrope 0,5 x 0,47 x 0,47 mm (grille d'origine), R029, R014."""
    return {
        "iso": load_subject(records.loc["sub-r009s001"]),
        "aniso": load_subject(records.loc["sub-soop0001"], spacing=None),
        "r029": load_subject(records.loc["sub-r029s001"]),
        "r014": load_subject(records.loc["sub-r014s001"]),
    }


@pytest.fixture(scope="session")
def other_site(records):
    """Sujets d'un autre centre (SOOP à 1 mm), références de FDA et d'histogram matching."""
    return [load_subject(records.loc[name]) for name in ("sub-soop0001", "sub-soop0002")]


@pytest.fixture(scope="session")
def reference_quantiles(other_site):
    return fit_reference_quantiles(other_site)
