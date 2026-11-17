# RQ3 - analyse des échecs (pli 0)

Dice moyen sur les sujets à lésion non vide. Distance C6 : Wasserstein entre histogrammes d'intensité du cerveau (référence : tous les autres centres). Contraste : z lésion - z anneau 1-4 mm (médiane du centre). « Continue » : Dice préquentiel, sur un sous-ensemble des sujets du centre.

## Centres OOD

| Centre | Sujets (lésion) | Vol. médian (ml) | J post-AVC | Distance W1 | Contraste | Baseline | Intensité | BigAug | FDA | Histogram matching | DANN | Baseline + TENT | Continue (contrôlée) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R008 | 7 (7) | 0.95 | 1 | 0.216 | -0.17 | 0.00 | 0.00 | 0.01 | 0.00 | 0.02 | 0.00 | 0.00 | 0.00 |
| R031 | 37 (37) | 6.74 | 138 | 0.164 | -0.98 | 0.62 | 0.67 | 0.66 | 0.65 | 0.61 | 0.63 | 0.62 | 0.58 |
| R028 | 26 (26) | 4.94 | 576 | 0.157 | -0.82 | 0.66 | 0.65 | 0.65 | 0.64 | 0.66 | 0.65 | 0.66 | 0.66 |
| SOOP | 169 (167) | 22.75 | - | 0.023 | -0.69 | 0.54 | 0.57 | 0.57 | 0.54 | 0.57 | 0.53 | 0.52 | 0.44 |
| R053 | 5 (4) | 0.81 | 25 | 0.020 | -0.50 | 0.20 | 0.20 | 0.20 | 0.22 | 0.24 | 0.25 | 0.20 | 0.20 |
| R034 | 24 (24) | 2.38 | 1506 | 0.014 | -0.64 | 0.63 | 0.65 | 0.64 | 0.66 | 0.65 | 0.64 | 0.63 | 0.63 |

## Dice par volume de lésion (test OOD ; ID entre parenthèses)

| Volume | Baseline | Intensité | BigAug | FDA | Histogram matching | DANN | Baseline + TENT | Continue (contrôlée) |
|---|---|---|---|---|---|---|---|---|
| < 1 ml | 0.31 (n=41) (0.48) | 0.33 (n=41) (0.46) | 0.32 (n=41) (0.43) | 0.30 (n=41) (0.44) | 0.33 (n=41) (0.49) | 0.29 (n=41) (0.46) | 0.32 (n=41) (0.48) | 0.35 (n=23) |
| 1-10 ml | 0.46 (n=86) (0.58) | 0.50 (n=86) (0.59) | 0.48 (n=86) (0.59) | 0.46 (n=86) (0.61) | 0.49 (n=86) (0.58) | 0.45 (n=86) (0.58) | 0.46 (n=86) (0.58) | 0.48 (n=41) |
| 10-50 ml | 0.64 (n=80) (0.71) | 0.67 (n=80) (0.74) | 0.68 (n=80) (0.75) | 0.66 (n=80) (0.74) | 0.66 (n=80) (0.71) | 0.63 (n=80) (0.73) | 0.62 (n=80) (0.71) | 0.60 (n=30) |
| >= 50 ml | 0.74 (n=58) (0.83) | 0.74 (n=58) (0.83) | 0.77 (n=58) (0.83) | 0.75 (n=58) (0.84) | 0.74 (n=58) (0.83) | 0.75 (n=58) (0.83) | 0.71 (n=58) (0.82) | 0.81 (n=13) |

## Corrélations de Spearman

| Niveau | x | y | rho | p | n |
|---|---|---|---|---|---|
| centres OOD | wasserstein | Dice baseline | -0.31 | 0.544 | 6 |
| centres OOD | wasserstein | Dice intensity | -0.20 | 0.704 | 6 |
| centres OOD | wasserstein | Dice bigaug | -0.03 | 0.957 | 6 |
| centres OOD | wasserstein | Dice fda | -0.43 | 0.397 | 6 |
| centres OOD | wasserstein | Dice histogram_matching | -0.31 | 0.544 | 6 |
| centres OOD | wasserstein | Dice dann | -0.31 | 0.544 | 6 |
| centres OOD | wasserstein | Dice baseline+tent | -0.31 | 0.544 | 6 |
| centres OOD | wasserstein | Dice continue_controlee | -0.31 | 0.544 | 6 |
| centres OOD | wasserstein | volume médian des lésions | 0.14 | 0.787 | 6 |
| centres OOD | mmd | Dice baseline | -0.14 | 0.787 | 6 |
| centres OOD | mmd | Dice intensity | -0.26 | 0.623 | 6 |
| centres OOD | mmd | Dice bigaug | -0.09 | 0.872 | 6 |
| centres OOD | mmd | Dice fda | -0.37 | 0.468 | 6 |
| centres OOD | mmd | Dice histogram_matching | -0.14 | 0.787 | 6 |
| centres OOD | mmd | Dice dann | -0.14 | 0.787 | 6 |
| centres OOD | mmd | Dice baseline+tent | -0.14 | 0.787 | 6 |
| centres OOD | mmd | Dice continue_controlee | -0.14 | 0.787 | 6 |
| centres OOD | mmd | volume médian des lésions | -0.37 | 0.468 | 6 |
| centres de test ID + OOD | wasserstein | Dice baseline | 0.09 | 0.575 | 44 |
| centres de test ID + OOD | wasserstein | Dice baseline+tent | 0.09 | 0.558 | 44 |
| centres de test ID + OOD | wasserstein | Dice bigaug | 0.04 | 0.778 | 44 |
| centres de test ID + OOD | wasserstein | Dice dann | 0.05 | 0.760 | 44 |
| centres de test ID + OOD | wasserstein | Dice fda | 0.05 | 0.727 | 44 |
| centres de test ID + OOD | wasserstein | Dice histogram_matching | 0.03 | 0.822 | 44 |
| centres de test ID + OOD | wasserstein | Dice intensity | 0.12 | 0.444 | 44 |
| sujets OOD | volume de la lésion | Dice intensity | 0.53 | 0.000 | 265 |
| sujets OOD | contraste de la lésion | Dice intensity | -0.19 | 0.001 | 265 |

## R008 et R053, sujet par sujet (modèle intensité)

| case_id | days_post_stroke | gt_ml | pred_ml | dice | lesion_count_diff | n_composantes | z_lesion | z_anneau | contraste | z_p99 | part_z_sup_3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R008_sub-r008s001_ses-1 | 1.00 | 4.07 | 8.59 | 0.00 | 25.00 | 1.00 | 0.04 | 0.06 | -0.02 | 3.70 | 0.015 |
| R008_sub-r008s002_ses-1 | 1.00 | 0.26 | 16.12 | 0.00 | 38.00 | 1.00 | 0.18 | 0.73 | -0.55 | 3.39 | 0.012 |
| R008_sub-r008s003_ses-1 | - | 0.95 | 13.43 | 0.00 | 38.00 | 1.00 | -0.74 | -0.57 | -0.17 | 3.78 | 0.015 |
| R008_sub-r008s004_ses-1 | 1.00 | 1.15 | 7.94 | 0.00 | 28.00 | 3.00 | 0.29 | 0.42 | -0.13 | 3.09 | 0.011 |
| R008_sub-r008s006_ses-1 | 1.00 | 0.11 | 33.63 | 0.00 | 67.00 | 1.00 | 0.10 | 0.38 | -0.28 | 4.25 | 0.017 |
| R008_sub-r008s007_ses-1 | 1.00 | 0.33 | 5.01 | 0.00 | 20.00 | 1.00 | -0.08 | 0.29 | -0.37 | 3.60 | 0.013 |
| R008_sub-r008s008_ses-1 | 1.00 | 2.35 | 0.39 | 0.00 | 9.00 | 1.00 | 0.21 | 0.06 | 0.15 | 3.26 | 0.012 |
| R053_sub-r053s016_ses-1 | 26.00 | 0.00 | 0.00 | - | 0.00 | 0.00 | - | - | - | 1.83 | 0.000 |
| R053_sub-r053s021_ses-1 | 105.00 | 1.48 | 3.54 | 0.27 | 3.00 | 8.00 | -0.01 | 0.62 | -0.63 | 1.69 | 0.000 |
| R053_sub-r053s039_ses-1 | 18.00 | 0.13 | 0.01 | 0.00 | 1.00 | 1.00 | 0.50 | 0.88 | -0.38 | 1.77 | 0.000 |
| R053_sub-r053s042_ses-1 | 5.00 | 0.01 | 0.07 | 0.00 | 1.00 | 1.00 | 0.36 | 0.98 | -0.62 | 1.74 | 0.000 |
| R053_sub-r053s053_ses-1 | 25.00 | 2.61 | 1.66 | 0.52 | 2.00 | 1.00 | 0.37 | 0.76 | -0.39 | 1.81 | 0.000 |
