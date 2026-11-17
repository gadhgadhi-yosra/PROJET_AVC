# Analyses de Yosra — S7 à S10

Toutes les commandes partent de la racine et utilisent `.venv-spark/bin/python`.
Fixer modèle, technique et pli dans chaque expérience. Les outils refusent les
prédictions sur les cas d’entraînement, les doublons et les centres incompatibles avec le split.

## Entrées à recevoir de Fayfa et Anas

CSV C3 : `model,technique,fold,case_id,dice` ; `site` facultatif mais vérifié s’il est fourni.
Une seule prédiction par cas pour un modèle/technique/pli. Les cas de validation servent
à la calibration ; les tests ID/OOD servent à l’évaluation finale. Les identifiants
doivent être ceux de `splits.json`, par exemple `R001_sub-r001s001_ses-1`.

Les fonctions de modèle sont des modules Python locaux approuvés par l’équipe, pas
du code chargé depuis un fichier utilisateur dans l’interface :

```python
def extract_features(t1_path):
    # Charger le vrai checkpoint, eval(), sans gradients ; appliquer son preprocessing.
    # Pooling spatial de la couche choisie : un vecteur numérique fini par scan.
    ...

def predict(t1_path):
    # Renvoyer la prédiction dans la grille PHYSIQUE du NIfTI d'entrée.
    return {'mask_path': '/chemin/masque.nii.gz',
            'uncertainty_path': '/chemin/incertitude.nii.gz'}
```

Ce contrat est préparé pour intégration ; l’architecture, les poids, la couche à extraire,
la normalisation et le retour à la grille native doivent être fournis avec le checkpoint.

## S7 — distance et perte de performance

```sh
.venv-spark/bin/python analysis/failure_prediction.py \
  --results data/predictions/results.csv --model nnUNet --technique baseline --fold 0
```

Sorties : `analysis/reports/s7/centers.csv`, `loco.csv`, `calibration.json`, `distance_dice.png`.
La distance est recalculée avec le train **du pli**, sans le centre mesuré. La table S5
exploratoire, calculée sur le pool global, ne doit pas servir de référence au modèle d’un pli.

La régression prédit `1 − Dice`, avec un poids égal par centre. La corrélation de Spearman
porte sur les centres de validation ; LOCO entraîne la régression et choisit le seuil
sans le centre laissé de côté. Le seuil d’échec Dice (0,5 par défaut) doit être fixé avant
de voir le test. Le seuil de distance maximise l’exactitude équilibrée sur validation.
Avec une seule classe succès/échec, aucun seuil n’est inventé. La colonne `delta_id`
donne séparément `moyenne Dice du test ID − Dice du centre` ; elle n’entraîne pas l’alerte.
Les prédictions sur validation ajustées sur tous ses centres sont descriptives ; seule
la table `loco.csv` fournit des prédictions hors calibration sur ces centres.

## S8 — délai post-AVC

```sh
.venv-spark/bin/python analysis/temporal_shift.py \
  --results data/predictions/results.csv --model nnUNet --technique baseline --fold 0
```

Régressions comparées sur les mêmes cas complets : constante, centre, `log(1+jours)`,
centre + temps. Les gains de R² conditionnels comparent les deux effets. Le test de
permutation mélange les délais **au sein du centre**, graine 2026, 999 permutations.
Les délais manquants/invalides sont comptés par centre et exclus sans imputation.
`CHRONICITY` n’est jamais traduit en stade clinique. Ce sont des associations
descriptives, dépendantes de l’échantillon, pas des effets causaux ni une validation
prédictive externe. La colinéarité centre/temps provoque un échec explicite.

## S8 — caractéristiques du réseau et C6 v2

```sh
.venv-spark/bin/python analysis/network_features.py extract \
  --adapter equipe.modele:extract_features --checkpoint-sha256 SHA256_REEL \
  --output data/features/baseline.npz
.venv-spark/bin/python analysis/network_features.py distance \
  --features data/features/baseline.npz --fold 0 \
  --output analysis/reports/distances_network.csv
```

Le NPZ contient `features` (n × d) et `case_ids` (chaînes, sans pickle). Une métadonnée
JSON conserve l’adaptateur et l’empreinte du checkpoint déclarés. Utiliser la même couche,
le même pooling et la même cohorte pour comparer les modèles. La standardisation et la
largeur RBF sont calculées sur la référence seule. La MMD utilise l’estimateur biaisé,
positif ; Wasserstein est la moyenne des distances marginales, pas une distance jointe.
Ces mesures dépendent des effectifs et ne sont pas numériquement comparables à C6 v1.

## S9–S10 — empreinte avant/après DANN

```sh
.venv-spark/bin/python analysis/network_features.py fingerprint \
  --features data/features/baseline.npz data/features/augmentation.npz data/features/dann.npz \
  --fold 0 --min-subjects 20
```

La comparaison retient les sujets communs hors entraînement du réseau et les centres
d’au moins 20 scans (minimum technique : 5 pour les cinq plis). Mêmes sujets et plis
pour chaque représentation ; standardisation dans chaque pli du classifieur.
Contrôle à labels mélangés pour chaque représentation. Les ± écarts-types des plis
ne sont pas des intervalles de confiance. Une diminution de l’empreinte ne prouve
pas une amélioration du Dice : confronter les résultats à ceux de Fayfa.

## Alerte individuelle de la démo

```sh
.venv-spark/bin/python analysis/calibrate_scan_ood.py --fold 0
```

Le quantile 95 % des distances individuelles de validation donne un seuil descriptif.
Les histogrammes train/validation doivent être complets. Le test OOD n’intervient pas.
L’alerte mesure une atypie d’intensité, **pas une probabilité d’échec**. Ce seuil est
distinct de l’alerte par centre de S7. Le fichier privé `data/demo/reference_ood.npz`
est chargé via `AVC_REFERENCE_HISTOGRAMS` dans la démo.

## S8 historique disponible après intégration de Fayfa

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-spark/bin/python analysis/historical_baselines.py
```

Cette commande exploite les scores versionnés avec leur partition historique, sans
modifier `splits.json`. Résultats et limites : `analysis/reports/s8_historical/README.md`.
Elle ne remplace pas l’expérience finale C1 v2. Aucun checkpoint n’est requis pour
analyser ces scores existants ; les analyses réseau restent dépendantes des poids.
