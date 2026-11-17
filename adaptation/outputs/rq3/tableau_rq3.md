# RQ3 - tableau d'ablation (pli 0)

Bootstrap : 10000 tirages, IC 95 % percentile. Wilcoxon apparié sur les sujets communs avec la baseline, correction de Holm sur les techniques. Dice sur les sujets à lésion non vide.

| Technique | n ID / OOD | Dice ID [IC 95 %] | Dice OOD [IC 95 %] | Écart ID - OOD | Δ OOD vs baseline | p Wilcoxon OOD (Holm) | p Wilcoxon ID (Holm) |
|---|---|---|---|---|---|---|---|
| Baseline | 185 / 265 | 0.646 [0.611 ; 0.680] | 0.553 [0.517 ; 0.589] | 0.093 [0.044 ; 0.142] | - | - (-) | - (-) |
| Intensité | 185 / 265 | 0.656 [0.621 ; 0.690] | 0.577 [0.543 ; 0.611] | 0.080 [0.031 ; 0.126] | 0.024 [0.008 ; 0.041] | 0.0014 (0.0068) | 0.05 (0.2) |
| BigAug | 185 / 265 | 0.653 [0.619 ; 0.686] | 0.577 [0.543 ; 0.611] | 0.075 [0.028 ; 0.122] | 0.025 [0.010 ; 0.040] | 0.035 (0.14) | 0.28 (0.56) |
| FDA | 185 / 265 | 0.660 [0.626 ; 0.693] | 0.559 [0.524 ; 0.593] | 0.101 [0.053 ; 0.148] | 0.006 [-0.007 ; 0.020] | 0.32 (0.49) | 0.012 (0.061) |
| Histogram matching | 185 / 265 | 0.646 [0.611 ; 0.681] | 0.570 [0.536 ; 0.604] | 0.077 [0.027 ; 0.125] | 0.017 [0.001 ; 0.034] | 0.24 (0.49) | 0.081 (0.24) |
| DANN | 185 / 265 | 0.650 [0.616 ; 0.684] | 0.549 [0.513 ; 0.584] | 0.102 [0.053 ; 0.150] | -0.004 [-0.018 ; 0.009] | 0.13 (0.39) | 0.45 (0.56) |
| Baseline + TENT | 185 / 265 | 0.645 [0.610 ; 0.679] | 0.542 [0.506 ; 0.578] | 0.104 [0.054 ; 0.153] | -0.011 [-0.016 ; -0.006] | < 1e-4 (< 1e-4) | 0.00017 (0.001) |

## Comparaisons directes (hors correction de Holm)

| A - B | Test | n | Dice A | Dice B | Δ moyen [IC 95 %] | Médiane | Sujets A > B / A < B | p Wilcoxon |
|---|---|---|---|---|---|---|---|---|
| BigAug - Intensité | ID | 185 | 0.653 | 0.656 | -0.003 [-0.017 ; 0.011] | -0.0015 | 84 / 96 | 0.35 |
| BigAug - Intensité | OOD | 265 | 0.577 | 0.577 | 0.001 [-0.010 ; 0.012] | +0.0000 | 123 / 128 | 0.53 |
| Histogram matching - Intensité | ID | 185 | 0.646 | 0.656 | -0.010 [-0.025 ; 0.005] | -0.0015 | 82 / 98 | 0.35 |
| Histogram matching - Intensité | OOD | 265 | 0.570 | 0.577 | -0.007 [-0.023 ; 0.009] | -0.0035 | 110 / 142 | 0.062 |

Par centre OOD :

| A - B | Centre | n | Dice A | Dice B | Δ moyen | Sujets A > B / A < B | p Wilcoxon |
|---|---|---|---|---|---|---|---|
| BigAug - Intensité | R008 | 7 | 0.006 | 0.000 | +0.006 | 1 / 0 | 1 |
| BigAug - Intensité | R028 | 26 | 0.652 | 0.649 | +0.003 | 16 / 10 | 0.5 |
| BigAug - Intensité | R031 | 37 | 0.663 | 0.672 | -0.009 | 15 / 22 | 0.32 |
| BigAug - Intensité | R034 | 24 | 0.642 | 0.653 | -0.011 | 11 / 12 | 0.76 |
| BigAug - Intensité | R053 | 4 | 0.195 | 0.197 | -0.002 | 1 / 1 | 1 |
| BigAug - Intensité | SOOP | 167 | 0.571 | 0.567 | +0.004 | 79 / 83 | 0.6 |
| Histogram matching - Intensité | R008 | 7 | 0.019 | 0.000 | +0.019 | 1 / 0 | 1 |
| Histogram matching - Intensité | R028 | 26 | 0.659 | 0.649 | +0.010 | 16 / 9 | 0.23 |
| Histogram matching - Intensité | R031 | 37 | 0.614 | 0.672 | -0.059 | 10 / 27 | 0.0018 |
| Histogram matching - Intensité | R034 | 24 | 0.648 | 0.653 | -0.005 | 14 / 9 | 0.41 |
| Histogram matching - Intensité | R053 | 4 | 0.237 | 0.197 | +0.040 | 1 / 1 | 1 |
| Histogram matching - Intensité | SOOP | 167 | 0.566 | 0.567 | -0.000 | 68 / 96 | 0.11 |

## Sources

- Baseline : data\kaggle_sorties\rq3-baseline-fp32-pli0\runs\unet3d_baseline_pli0\resultats_id_test.csv, data\kaggle_sorties\rq3-baseline-fp32-pli0\runs\unet3d_baseline_pli0\resultats_ood_test.csv
- Intensité : data\kaggle_sorties\rq3-intensity-pli0\runs\unet3d_intensity_pli0\resultats_id_test.csv, data\kaggle_sorties\rq3-intensity-pli0\runs\unet3d_intensity_pli0\resultats_ood_test.csv
- BigAug : data\kaggle_sorties\rq3-bigaug-pli0\runs\unet3d_bigaug_pli0\resultats_id_test.csv, data\kaggle_sorties\rq3-bigaug-pli0\runs\unet3d_bigaug_pli0\resultats_ood_test.csv
- FDA : data\kaggle_sorties\rq3-fda-pli0\runs\unet3d_fda_pli0\resultats_id_test.csv, data\kaggle_sorties\rq3-fda-pli0\runs\unet3d_fda_pli0\resultats_ood_test.csv
- Histogram matching : data\kaggle_sorties\rq3-histogram-matching-pli0\runs\unet3d_histogram_matching_pli0\resultats_id_test.csv, data\kaggle_sorties\rq3-histogram-matching-pli0\runs\unet3d_histogram_matching_pli0\resultats_ood_test.csv
- DANN : data\kaggle_sorties\rq3-dann-v2-reprise-pli0\runs\unet3d_dann_pli0\resultats_id_test.csv, data\kaggle_sorties\rq3-dann-v2-reprise-pli0\runs\unet3d_dann_pli0\resultats_ood_test.csv
- Baseline + TENT : data\kaggle_sorties\rq3-baseline-fp32-pli0\runs\unet3d_baseline_pli0\resultats_id_test_tent.csv, data\kaggle_sorties\rq3-baseline-fp32-pli0\runs\unet3d_baseline_pli0\resultats_ood_test_tent.csv
