"""Rapatriement des sorties Kaggle : statut exigé, copie du code retirée, sorties incomplètes refusées."""
import json

import pytest

from adaptation.kaggle import rapatrier


def fake_output(folder, with_results=True):
    run = folder / "runs" / "unet3d_bigaug_pli0"
    run.mkdir(parents=True)
    (folder / "PROJET_AVC" / "adaptation").mkdir(parents=True)
    (folder / "PROJET_AVC" / "VERSION.json").write_text(json.dumps({"commit": "7449c49"}), encoding="utf-8")
    (folder / "PROJET_AVC" / "adaptation" / "train.py").write_text("", encoding="utf-8")
    if with_results:
        for split in ("id_test", "ood_test"):
            (run / f"resultats_{split}.csv").write_text("technique\nbigaug\n", encoding="utf-8")


def test_nettoyage(tmp_path):
    fake_output(tmp_path)
    assert rapatrier.clean(tmp_path) == {"commit": "7449c49"}
    assert not (tmp_path / "PROJET_AVC").exists()
    assert (tmp_path / "version_code.json").is_file()


def test_sorties_incompletes_ou_lourdes(tmp_path):
    fake_output(tmp_path, with_results=False)
    with pytest.raises(SystemExit, match="incomplètes"):
        rapatrier.clean(tmp_path)
    (tmp_path / "best.pt").write_bytes(b"0")
    with pytest.raises(SystemExit, match=r"\.pt"):
        rapatrier.clean(tmp_path)


def test_kernel_non_termine(tmp_path, monkeypatch):
    calls = []

    def fake(*args):
        calls.append(args)
        return 'fayfahachicha/rq3-bigaug-pli0 has status "KernelWorkerStatus.RUNNING"'

    monkeypatch.setattr(rapatrier, "kaggle", fake)
    with pytest.raises(SystemExit, match="pas terminé"):
        rapatrier.fetch("fayfahachicha/rq3-bigaug-pli0", tmp_path)
    assert calls == [("kernels", "status", "fayfahachicha/rq3-bigaug-pli0")]
    assert not any(tmp_path.iterdir())
