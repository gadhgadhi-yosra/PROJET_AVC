"""Analyse des échecs RQ3 sur des sorties synthétiques (aucune donnée patient)."""
import math

import numpy as np
import pandas as pd
import pytest

from adaptation import analyse_echecs as ae


def write_run(folder, technique, rows):
    path = folder / technique / "resultats_ood_test.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame([dict(model="unet3d", technique=technique, fold=0, split="ood_test", hd95_mm=1.0, lesion_f1=1.0,
                       abs_vol_diff_ml=0.0, lesion_count_diff=0, pred_ml=1.0, seconds=1.0, **row)
                  for row in rows]).to_csv(path, index=False)


@pytest.fixture
def outputs(tmp_path):
    cases = [("A", 1, 0.5, 0.9), ("A", 2, 20.0, 0.8), ("B", 1, 0.3, 0.0), ("B", 2, 0.0, math.nan),
             ("C", 1, 60.0, 0.6), ("D", 1, 5.0, 0.4)]
    for technique, shift in (("baseline", 0.0), ("intensity", 0.05)):
        write_run(tmp_path, technique, [dict(case_id=f"{s}_{i}", site=s, days_post_stroke=10, gt_ml=ml,
                                             dice=d + shift if np.isfinite(d) else d) for s, i, ml, d in cases])
    continual = tmp_path / "continue" / "prequentiel.csv"
    continual.parent.mkdir()
    pd.DataFrame([dict(strategie="controlee", ordre=1, centre=s, lot=1, case_id=f"{s}_{i}", gt_ml=ml, dice=d)
                  for s, i, ml, d in cases]).to_csv(continual, index=False)
    distances = tmp_path / "distances.csv"
    pd.DataFrame(dict(site=list("ABCD"), wasserstein=[0.1, 0.4, 0.2, 0.3], mmd=[0.1, 0.4, 0.2, 0.3])).to_csv(
        distances, index=False)
    return tmp_path, distances


def test_analyse_sans_volumes(outputs, tmp_path):
    folder, distances = outputs
    result = ae.analyse([folder], tmp_path / "out", tmp_path / "fig", distances_path=distances)
    centers = result["centers"].set_index("site")
    assert centers.loc["B", "n_sujets"] == 2 and centers.loc["B", "n_lesion"] == 1
    assert centers.loc["A", "volume_median_ml"] == pytest.approx(10.25)
    assert centers.loc["A", "intensity"] == pytest.approx(0.9)
    assert centers.loc["B", "continue_controlee"] == pytest.approx(0.0)
    sizes = result["sizes"]
    small = sizes[(sizes["taille"] == "< 1 ml") & (sizes["technique"] == "baseline")]
    assert small["n"].item() == 2 and small["dice"].item() == pytest.approx(0.45)
    row = result["correlation"].query("niveau == 'centres OOD' and x == 'wasserstein' and y == 'Dice baseline'")
    assert row["rho"].item() == pytest.approx(-1.0) and row["n"].item() == 4
    assert (tmp_path / "out" / "echecs.md").read_text(encoding="utf-8").startswith("# RQ3")
    assert len(result["figures"]) == 2


def test_contraste_de_lesion():
    t1 = np.zeros((20, 20, 20), np.float32)
    brain = np.ones_like(t1)
    mask = np.zeros_like(t1)
    mask[8:12, 8:12, 8:12] = 1
    t1[mask > 0] = -1.0
    measured = ae.lesion_contrast(t1, mask, brain)
    assert measured["contraste"] == pytest.approx(-1.0) and measured["n_composantes"] == 1
    assert math.isnan(ae.lesion_contrast(t1, np.zeros_like(mask), brain)["contraste"])
