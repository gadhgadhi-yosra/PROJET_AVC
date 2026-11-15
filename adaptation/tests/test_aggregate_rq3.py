"""Agrégation RQ3 sur des sorties factices : six runs répartis sur trois « comptes », un doublon et une archive .zip."""
import json
import math
import zipfile

import numpy as np
import pandas as pd
import pytest

from adaptation.aggregate_rq3 import aggregate, bootstrap_mean, holm, wilcoxon_p
from adaptation.train import RESULT_COLUMNS

SITES_ID, SITES_OOD = ["R001", "R002"], ["R008", "SOOP"]
EFFECTS = {"baseline": 0.0, "intensity": 0.02, "bigaug": 0.10, "fda": 0.0, "histogram_matching": -0.05, "dann": 0.01}


def fake_rows(technique, split, rng, n=40):
    sites = SITES_ID if split == "id_test" else SITES_OOD
    base = 0.7 if split == "id_test" else 0.5
    rows = []
    for index in range(n):
        gt = [0.5, 5.0, 20.0, 80.0][index % 4] if index != 0 else 0.0  # le premier sujet : masque vide
        subject_rng = np.random.default_rng(1000 * (split == "ood_test") + index)  # même difficulté pour tous
        dice = float(np.clip(base + subject_rng.normal(0, 0.1) + EFFECTS[technique] + rng.normal(0, 0.01), 0, 1))
        rows.append(dict(model="unet3d", technique=technique, fold=0, case_id=f"{sites[index % 2]}_s{index:03d}",
                         site=sites[index % 2], split=split, days_post_stroke=1, dice=math.nan if gt == 0 else dice,
                         hd95_mm=10.0, abs_vol_diff_ml=1.0, lesion_f1=0.5, lesion_count_diff=1, gt_ml=gt,
                         pred_ml=gt, seconds=1.0))
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def write_run(folder, technique, rng):
    run = folder / "runs" / f"unet3d_{technique}_pli0"
    run.mkdir(parents=True)
    for split in ("id_test", "ood_test"):
        fake_rows(technique, split, rng).to_csv(run / f"resultats_{split}.csv", index=False)
    state = dict(iteration=40000, elapsed=6 * 3600, best_dice=0.6, best_iteration=38000)
    (run / "resume.json").write_text(json.dumps(dict(run=run.name, technique=technique, fold=0, entrainement=state)))
    return run


@pytest.fixture
def outputs(tmp_path):
    rng = np.random.default_rng(0)
    accounts = {"fayfa": ["baseline", "fda"], "yosra": ["intensity", "histogram_matching"], "anas": ["bigaug"]}
    for account, techniques in accounts.items():
        for technique in techniques:
            write_run(tmp_path / "sorties" / account / f"rq3-{technique}-pli0", technique, rng)
    write_run(tmp_path / "sorties" / "fayfa" / "doublon", "baseline", rng)  # même run téléchargée deux fois
    dann = write_run(tmp_path / "zip_source", "dann", rng)
    with zipfile.ZipFile(tmp_path / "sorties" / "anas" / "rq3-dann-pli0.zip", "w") as archive:
        for path in dann.iterdir():
            archive.write(path, f"runs/{dann.name}/{path.name}")
    return tmp_path


def test_agregation_complete(outputs):
    with pytest.warns(UserWarning, match="lignes en double"):
        result = aggregate([outputs / "sorties"], outputs / "agregation", outputs / "figures", n_boot=500)
    table = result["table"].set_index("technique")
    assert list(table.index) == ["baseline", "intensity", "bigaug", "fda", "histogram_matching", "dann"]
    assert (table["n_id"] == 39).all() and (table["n_ood"] == 39).all()  # masque vide exclu
    bigaug = table.loc["bigaug"]
    assert bigaug.delta_ood == pytest.approx(0.10, abs=0.02) and bigaug.p_wilcoxon_ood < 1e-4
    assert bigaug.delta_ood_ic_bas < bigaug.delta_ood < bigaug.delta_ood_ic_haut
    assert table.loc["histogram_matching"].delta_ood < 0
    assert table.loc["fda"].p_holm_ood >= table.loc["fda"].p_wilcoxon_ood
    assert table.loc["baseline"].ecart == pytest.approx(0.2, abs=0.05)
    assert set(result["centers"]["site"]) == set(SITES_ID + SITES_OOD)
    assert set(result["sizes"]["taille"]) == {"< 1 ml", "1-10 ml", "10-50 ml", ">= 50 ml"}
    assert len(result["training"]) == 7 and set(result["training"]["iterations"]) == {40000}
    assert len(result["figures"]) == 4 and all(path.stat().st_size > 0 for path in result["figures"])
    text = (outputs / "agregation" / "tableau_rq3.md").read_text(encoding="utf-8")
    assert "| BigAug |" in text and "rq3-dann-pli0.zip" in text


def test_statistiques():
    mean, low, high = bootstrap_mean([1.0, 2.0, 3.0, 4.0], n=2000)
    assert mean == 2.5 and 1.0 <= low < mean < high <= 4.0
    assert all(math.isnan(v) for v in bootstrap_mean([]))
    assert math.isnan(wilcoxon_p([0.5, 0.5], [0.5, 0.5])) and math.isnan(wilcoxon_p([0.5], [0.4]))
    assert holm([0.01, 0.04, math.nan, 0.03]) == pytest.approx([0.03, 0.06, math.nan, 0.06], nan_ok=True)


def test_sans_resultats(tmp_path):
    with pytest.raises(SystemExit, match="Aucun fichier"):
        aggregate([tmp_path], tmp_path / "out", tmp_path / "fig")
