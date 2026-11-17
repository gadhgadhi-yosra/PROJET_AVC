# Analyses du domain shift

## S5 — statistiques après prétraitement et distance C6 v1

```bash
analysis/run_analysis.sh --workers 4
.venv-spark/bin/python -m unittest discover -s tests -v
```

Entrées : `data/processed/`, `data/qc_s4/subjects.parquet`, `splits.json`. Sorties : `analysis/reports/` (agrégats par centre), `figures/s5_*.png` et `data/processed/post_stats.parquet` (histogrammes par sujet, privés, hors Git). Le masque de lésion n'intervient dans aucune mesure.

## Contrat C6 v1

```python
from center_distance import brain_histogram, center_distance

scans = [brain_histogram(t1, brain) for t1, brain in volumes_du_centre]
d = center_distance(scans, histogrammes_reference)                  # Wasserstein
d = center_distance(scans, histogrammes_reference, metric='mmd')    # MMD
```

`histogrammes_reference` a la forme `n_scans × 150`. La référence doit être le pool d'entraînement **sans le centre mesuré**. Pour Wasserstein, l'unité est l'écart-type du z-score. La MMD exige au moins deux scans de référence distincts.

## Limites

- Résultats sur 1 402 sujets : 14 sujets `lesion_outside_image_based_crop` restent à traiter (S4). À relancer ensuite.
- Le z-score fixe la moyenne et l'écart-type de chaque scan : la distance mesure surtout la forme de l'histogramme restante.
- `relative_before`/`relative_after` ne sont qu'un ordre de grandeur : les histogrammes « avant » viennent du QC (32 classes, normalisés par p99), ceux d'« après » du z-score (150 classes).
- Les centres OOD ont été choisis avec une distance proche ; l'écart ID/OOD est donc en partie dû à cette sélection. Aucune relation avec le Dice n'est établie avant S7.
- Les centres de moins de 5 scans sont exclus des corrélations et de la figure, car leur distance est bruitée.

## S6 — empreinte du centre sur données prétraitées

```bash
.venv-spark/bin/python analysis/site_fingerprint.py
```

À lancer après `run_analysis.sh`. Sorties : `analysis/reports/site_fingerprint_s6*`, `figures/s6_empreinte_centre.png`. Protocole du script 03 : centres d'au moins 20 sujets, 5 plis stratifiés, exactitude équilibrée, contrôle à étiquettes mélangées.

Limites : l'empreinte mesure ce qu'un classifieur simple retrouve dans des histogrammes, pas ce qu'utilise un réseau (voir S8 et « après DANN »). Les plis mélangent les sujets d'un même centre : c'est voulu, car on cherche à reconnaître le centre. Résultats sur 1 130 scans, hors R032 et hors sujets non traités ; à relancer après correction des 14 sujets.
