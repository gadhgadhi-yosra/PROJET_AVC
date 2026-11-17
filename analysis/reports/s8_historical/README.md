# S8 — délai post-AVC et Dice des baselines historiques

Analyse exploratoire des scores fournis, sans nouvelle inférence. **Ce résultat ne concerne pas C1 v2.**

Modèle : `cnn_pretrained_freeze-none`. Partition historique : 903 train, 226 ID validation, 287 OOD.

| Groupe | Cas complets / lignes fournies | Centres | Gain de R² du délai après centre | p permutation exploratoire |
|---|---:|---:|---:|---:|
| ID | 167 / 226 | 31 | 0.0000 | 0.911 |
| OOD | 257 / 287 | 12 | 0.0381 | 0.001 |

Le gain de R² est la part supplémentaire de variance du Dice expliquée par log(1 + délai), après prise en compte du centre. Les permutations mélangent les délais à l’intérieur de chaque centre (nombre dans summary.json). Les groupes ID et OOD sont analysés séparément.

## Sensibilité au volume lésionnel

- ID : après ajustement supplémentaire sur log(1 + volume), gain de R² du délai = 0.0007.
- OOD : après ajustement supplémentaire sur log(1 + volume), gain de R² du délai = 0.0002.

## Limites

- ID est la validation historique ayant participé au choix du modèle.
- Analyses ID/OOD séparées, cas complets ; les Dice manquants ne sont pas imputés.
- Le cas à masque vide r065s002 a un Dice absent dans tous les CSV : compté dans les données manquantes, exclu des ajustements.
- P-values exploratoires non corrigées pour les deux groupes ; pas de conclusion causale.
- Prétraitement et partition historiques distincts de C1 v2.
- La sensibilité au volume ne supprime pas tous les facteurs de confusion.

La relation observée ne prouve pas un effet causal du temps. Les résultats finaux sur C1 v2, les représentations réseau et la comparaison DANN restent à produire.

Provenance, empreintes et valeurs détaillées : `summary.json`. Autres modèles : `model_summary.csv`. Métadonnées manquantes par centre : `missing_by_center.csv`.
