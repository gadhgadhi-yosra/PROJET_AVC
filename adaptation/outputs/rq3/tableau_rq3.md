# RQ3 - tableau d'ablation (pli 0)

Bootstrap : 10000 tirages, IC 95 % percentile. Wilcoxon apparié sur les sujets communs avec la baseline, correction de Holm sur les techniques. Dice sur les sujets à lésion non vide.

| Technique | n ID / OOD | Dice ID [IC 95 %] | Dice OOD [IC 95 %] | Écart ID - OOD | Δ OOD vs baseline | p Wilcoxon OOD (Holm) | p Wilcoxon ID (Holm) |
|---|---|---|---|---|---|---|---|
| Baseline | 185 / 265 | 0.646 [0.610 ; 0.680] | 0.553 [0.517 ; 0.589] | 0.093 [0.044 ; 0.143] | - | - (-) | - (-) |
| Intensité | 185 / 265 | 0.656 [0.622 ; 0.689] | 0.577 [0.542 ; 0.610] | 0.080 [0.031 ; 0.127] | 0.024 [0.008 ; 0.040] | 0.0014 (0.0041) | 0.05 (0.099) |
| FDA | 185 / 265 | 0.660 [0.627 ; 0.693] | 0.559 [0.524 ; 0.593] | 0.101 [0.052 ; 0.149] | 0.006 [-0.008 ; 0.020] | 0.32 (0.32) | 0.012 (0.036) |
| DANN | 185 / 265 | 0.650 [0.616 ; 0.683] | 0.549 [0.514 ; 0.584] | 0.102 [0.054 ; 0.150] | -0.004 [-0.018 ; 0.009] | 0.13 (0.26) | 0.45 (0.45) |
| Baseline + TENT | 185 / 265 | 0.645 [0.610 ; 0.679] | 0.542 [0.507 ; 0.577] | 0.104 [0.054 ; 0.152] | -0.011 [-0.016 ; -0.006] | < 1e-4 (< 1e-4) | 0.00017 (0.00069) |

## Sources

- Baseline : C:\Users\Anas\Desktop\bureau\fayfa\tek-up_3\PROJET_AVC\data\kaggle_sorties\rq3-baseline-fp32-pli0\runs\unet3d_baseline_pli0\resultats_id_test.csv, C:\Users\Anas\Desktop\bureau\fayfa\tek-up_3\PROJET_AVC\data\kaggle_sorties\rq3-baseline-fp32-pli0\runs\unet3d_baseline_pli0\resultats_ood_test.csv
- Intensité : C:\Users\Anas\Desktop\bureau\fayfa\tek-up_3\PROJET_AVC\data\kaggle_sorties\rq3-intensity-pli0\runs\unet3d_intensity_pli0\resultats_id_test.csv, C:\Users\Anas\Desktop\bureau\fayfa\tek-up_3\PROJET_AVC\data\kaggle_sorties\rq3-intensity-pli0\runs\unet3d_intensity_pli0\resultats_ood_test.csv
- FDA : C:\Users\Anas\Desktop\bureau\fayfa\tek-up_3\PROJET_AVC\data\kaggle_sorties\rq3-fda-pli0\runs\unet3d_fda_pli0\resultats_id_test.csv, C:\Users\Anas\Desktop\bureau\fayfa\tek-up_3\PROJET_AVC\data\kaggle_sorties\rq3-fda-pli0\runs\unet3d_fda_pli0\resultats_ood_test.csv
- DANN : C:\Users\Anas\Desktop\bureau\fayfa\tek-up_3\PROJET_AVC\data\kaggle_sorties\rq3-dann-v2-reprise-pli0\runs\unet3d_dann_pli0\resultats_id_test.csv, C:\Users\Anas\Desktop\bureau\fayfa\tek-up_3\PROJET_AVC\data\kaggle_sorties\rq3-dann-v2-reprise-pli0\runs\unet3d_dann_pli0\resultats_ood_test.csv
- Baseline + TENT : C:\Users\Anas\Desktop\bureau\fayfa\tek-up_3\PROJET_AVC\data\kaggle_sorties\rq3-baseline-fp32-pli0\runs\unet3d_baseline_pli0\resultats_id_test_tent.csv, C:\Users\Anas\Desktop\bureau\fayfa\tek-up_3\PROJET_AVC\data\kaggle_sorties\rq3-baseline-fp32-pli0\runs\unet3d_baseline_pli0\resultats_ood_test_tent.csv
