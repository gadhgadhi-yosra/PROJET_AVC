# RQ3 - tableau d'ablation (pli 0)

Bootstrap : 10000 tirages, IC 95 % percentile. Wilcoxon apparié sur les sujets communs avec la baseline, correction de Holm sur les techniques. Dice sur les sujets à lésion non vide.

| Technique | n ID / OOD | Dice ID [IC 95 %] | Dice OOD [IC 95 %] | Écart ID - OOD | Δ OOD vs baseline | p Wilcoxon OOD (Holm) | p Wilcoxon ID (Holm) |
|---|---|---|---|---|---|---|---|
| Baseline | 185 / 265 | 0.646 [0.611 ; 0.680] | 0.553 [0.517 ; 0.589] | 0.093 [0.044 ; 0.142] | - | - (-) | - (-) |
| Intensité | 185 / 265 | 0.656 [0.621 ; 0.690] | 0.577 [0.543 ; 0.611] | 0.080 [0.031 ; 0.126] | 0.024 [0.008 ; 0.041] | 0.0014 (0.0027) | 0.05 (0.05) |
| FDA | 185 / 265 | 0.660 [0.626 ; 0.693] | 0.559 [0.524 ; 0.593] | 0.101 [0.053 ; 0.148] | 0.006 [-0.007 ; 0.020] | 0.32 (0.32) | 0.012 (0.024) |
| Baseline + TENT | 185 / 265 | 0.645 [0.610 ; 0.679] | 0.542 [0.506 ; 0.578] | 0.104 [0.054 ; 0.153] | -0.011 [-0.016 ; -0.006] | < 1e-4 (< 1e-4) | 0.00017 (0.00052) |

## Sources

- Baseline : data\kaggle_sorties\rq3-baseline-fp32-pli0\runs\unet3d_baseline_pli0\resultats_id_test.csv, data\kaggle_sorties\rq3-baseline-fp32-pli0\runs\unet3d_baseline_pli0\resultats_ood_test.csv
- Intensité : data\kaggle_sorties\rq3-intensity-pli0\runs\unet3d_intensity_pli0\resultats_id_test.csv, data\kaggle_sorties\rq3-intensity-pli0\runs\unet3d_intensity_pli0\resultats_ood_test.csv
- FDA : data\kaggle_sorties\rq3-fda-pli0\runs\unet3d_fda_pli0\resultats_id_test.csv, data\kaggle_sorties\rq3-fda-pli0\runs\unet3d_fda_pli0\resultats_ood_test.csv
- Baseline + TENT : data\kaggle_sorties\rq3-baseline-fp32-pli0\runs\unet3d_baseline_pli0\resultats_id_test_tent.csv, data\kaggle_sorties\rq3-baseline-fp32-pli0\runs\unet3d_baseline_pli0\resultats_ood_test_tent.csv
