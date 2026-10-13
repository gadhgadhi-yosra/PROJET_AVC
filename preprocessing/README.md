# Prétraitement — semaine 3 (12–18 octobre 2026)

Ce pipeline produit l’inventaire complet des sessions avec Spark et prépare la géométrie des IRM : orientation RAS, voxels isotropes de 1 mm, interpolation linéaire des intensités et plus proche voisin des masques. Les masques sont seuillés à `> 0.5` avant interpolation. Une différence de grille image/masque provoque une erreur explicite.

Il s’agit de volumes intermédiaires : la normalisation, le recadrage, le QC complet, le benchmark, les splits et l’export nnU-Net restent au programme de la semaine 4. Ne pas les présenter comme les données finales du contrat C1.

## Environnement

Python 3.12 et Java 17. Sur ce Mac, Java est installé localement dans `.tools/jdk17/Contents/Home` ; `run.sh` le sélectionne automatiquement. Sur un autre Mac, le script recherche Java 17 avec `java_home`. Sous Linux, définir `JAVA_HOME` puis exécuter directement le script Python.

```bash
python3.12 -m venv .venv-spark
.venv-spark/bin/pip install -r requirements.txt
.venv-spark/bin/pip check
.venv-spark/bin/python preprocessing/check_environment.py
```

ANTsPy 0.5.4 dispose d’un paquet binaire pour ce Mac Intel/Python 3.12 ; sa contrainte NumPy est prise en compte dans les dépendances. ANTsPy et SimpleITK sont préparés pour les étapes suivantes ; cette étape utilise nibabel et SciPy.

## Exécution depuis la racine

```bash
# Inventaire complet, sans réécrire les IRM
preprocessing/run.sh --inventory-only

# Petit essai (inventaire complet + géométrie des premiers sujets)
preprocessing/run.sh --limit 3 --workers 2

# Sélection d’un centre, sortie distincte pour conserver le premier essai
preprocessing/run.sh --sites R029 --limit 1 --output data/preprocessing_s3_r029

# Lot complet, après avoir libéré suffisamment d’espace
preprocessing/run.sh --workers 2

# Tests de géométrie sur données synthétiques
.venv-spark/bin/python -m unittest discover -s tests -v
```

Spark ouvre des ports locaux : l’exécution peut nécessiter l’autorisation de sortir du bac à sable de l’outil. Aucun service distant ni transfert de données n’est utilisé.

## Sorties locales privées

Dans `data/preprocessing_s3/` (ignoré par Git) :

- `subjects.parquet/` : table Spark de toutes les sessions, centre, sujet, session, chemins, délai post-AVC, drapeau CHRONICITY (booléen, vide si non marqué) et avertissements de métadonnées.
- `volumes/<centre>/<sujet>/<session>/{t1,mask}.nii.gz` : sorties géométriques du sous-ensemble sélectionné.
- `geometry_results.parquet/` : statuts `ok`, `excluded`, `error`, chemins et erreurs du dernier lot.
- `summary.json` : compteurs agrégés du dernier lancement.

Les tables sont remplacées à chaque lancement dans le même dossier ; les volumes précédents restent présents. Utiliser une sortie distincte pour chaque essai. Le centre R032 est exclu provisoirement à cause de l’espacement suspect identifié pendant l’exploration. Les métadonnées vides sont signalées et ne bloquent pas une paire d’images valide.

Un contrôle d’espace estime la taille des sorties compressées (2 × la taille des fichiers d’entrée, ajustée au nombre de voxels) et conserve 2 Gio de marge : environ 10,5 Gio pour le lot complet. Les données brutes restent intactes.
