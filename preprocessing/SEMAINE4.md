# Semaine 4 — données, partitions et benchmark

Le module `week4.py` complète le travail S3 sans modifier ses volumes. Les données brutes restent intactes. L’environnement est `.venv-spark` (Python 3.12, Java 17 local dans `.tools/jdk17`).

## Commandes

Depuis la racine du projet :

```bash
# QC complet sans duplication des IRM
preprocessing/run_week4.sh run --qc-only --output data/qc_s4 --workers 2

# Proposition reproductible figée à partir du QC (refuse d'écraser un split existant)
preprocessing/run_week4.sh splits

# Traitement complet : nécessite suffisamment d'espace libre
preprocessing/run_week4.sh run --output data/processed --workers 2

# Vérification sur un sous-ensemble seulement
preprocessing/run_week4.sh run --cases data/validation_s4/selection.json \
  --output data/validation_s4/processed --workers 2

# Relire et contrôler les sorties du lot de validation
.venv-spark/bin/python preprocessing/validate_s4.py data/validation_s4/processed

# Export complet pour Fayfa, après le traitement complet
preprocessing/run_week4.sh export

# Benchmark réel sur 8 sujets de centres distincts, 1/2/4/8 workers
preprocessing/run_week4.sh benchmark --repeats 1

# Tests synthétiques : aucun accès aux IRM requis
.venv-spark/bin/python -m unittest discover -s tests -v
```

Espace requis pour le traitement complet : environ 6,3 Gio de sorties (mesuré : 1,07 × la taille des T1 d’entrée) ; le script vérifie 1,3 × l’entrée + 2 Gio de réserve, soit environ 9,6 Gio libres. L’export utilise des liens physiques et n’ajoute presque rien sur le même disque.

Pour un disque externe : `--output /Volumes/NOM/atlas_processed`, puis `export --processed /Volumes/NOM/atlas_processed --output /Volumes/NOM/nnUNet_raw/Dataset001_ATLAS`. L’inventaire conserve des chemins absolus locaux, donc il faut le régénérer sur chaque poste. Les sorties personnalisées doivent rester hors Git.

## Traitement et contrôle qualité

1. Vérifier les dimensions, les grilles image/masque, l’affine, les valeurs finies et les valeurs binaires des masques (tolérance aux arrondis proches de 1).
2. Exclure R032, dont l’espacement est suspect. Les fichiers corrompus et paires manquantes produisent un statut `error` ; les erreurs n’arrêtent pas les autres sujets, mais le programme renvoie un code d’échec à la fin.
3. Détecter la valeur de fond par le mode d’un sous-échantillonnage régulier de l’image ; définir le cerveau par `intensité > fond`, y compris pour R029. Cette règle est héritée de l’exploration et reste à inspecter visuellement sur les cas atypiques.
4. Réorienter RAS et rééchantillonner à 1 mm : interpolation linéaire de l’image, plus proche voisin des masques cérébraux et lésionnels. La lésion est seuillée à `> 0.5` avant interpolation.
5. Calculer la moyenne et l’écart-type uniquement dans le masque cérébral rééchantillonné. Appliquer le z-score dans ce masque et mettre le fond à zéro. Aucun paramètre n’est appris sur les centres de test. Le z-score est stocké en `int16` avec `scl_slope = 0,001` (erreur ≤ 0,0005 écart-type, plage ±32,7) : environ deux fois moins d’espace qu’en `float32`. Une valeur hors plage fait échouer le sujet au lieu d’être tronquée ; nibabel et SimpleITK (nnU-Net) relisent directement des valeurs flottantes.
6. Recadrer autour du cerveau avec une marge de 5 voxels. L’affine est translatée pour conserver les coordonnées physiques. Le masque lésionnel ne détermine pas la boîte de recadrage ; si une lésion serait coupée, le cas échoue explicitement et doit être examiné.

Une lésion non vide qui disparaîtrait après rééchantillonnage provoque une erreur. Le volume lésionnel avant/après et sa variation relative sont enregistrés. Les masques initialement vides sont conservés comme cas négatifs et signalés par `empty_mask`. Les métadonnées vides restent des avertissements et n’empêchent pas l’utilisation d’une paire valide. Les délais négatifs/non finis sont signalés ; les valeurs manquantes ne sont pas imputées. CHRONICITY reste un drapeau, pas un stade clinique.

Chaque sujet produit `t1.nii.gz`, `mask.nii.gz`, `brain.nii.gz` et `qc.json` sous `data/processed/<sujet>/`. Le masque cérébral permet de vérifier le z-score sans confondre les vrais zéros intracérébraux avec le fond. `qc.json` est écrit en dernier et contient une signature des chemins, tailles, dates de modification et paramètres ; la reprise exige cette signature et les trois volumes. Ce n’est pas une preuve cryptographique d’intégrité des volumes : l’export relit les images.

`subjects.parquet` trace le dernier lot sélectionné (pas l’union de plusieurs petits lots). `site_summary.csv` résume les volumes et intensités par centre. Relancer sans `--limit` pour produire la table finale complète. Deux sessions d’un même sujet sont refusées pour éviter les collisions de sortie et les fuites inter-sujets.

## Protocole ID/OOD et validation

`splits.json` contient des identifiants pseudonymisés, sans chemins de fichiers ni mesures individuelles. Les distributions d’intensité utilisées pour la sélection restent locales. Les centres d’au moins 5 sujets sont divisés en trois strates de taille ; dans chacune, un centre proche et un centre éloigné sont réservés en OOD, d’après une distance Wasserstein discrète entre histogrammes moyens normalisés par le p99. La référence est la moyenne des centres du pool de conception, avant entraînement ; il s’agit d’une sélection exploratoire sur données non annotées, pas d’une mesure prédictive validée du risque d’échec. Aucun Dice n’intervient.

Les six centres OOD sont entièrement absents des données d’entraînement et de validation. Dans chaque autre centre, environ 20 % des sujets sont réservés pour le test ID (au moins un lorsque le centre a deux sujets). Le reste est partagé en cinq plis équilibrés par effectif, avec des centres entièrement distincts entre entraînement et validation. Pour chaque pli, `id_test` ne retient que les sujets dont le centre est présent dans l’entraînement de ce pli. Les tests ID/OOD ne doivent pas servir au choix des hyperparamètres.

La graine est 2026. Le script refuse de remplacer silencieusement un split déjà figé. Tout changement d’exclusion après QC des données prétraitées nécessite une nouvelle version explicitement revue du split, commune à toute l’équipe. Le champ `data_validation` rappelle que le QC prétraité complet doit encore être vérifié avant entraînement.

## Remise à Fayfa : nnU-Net v2

L’export complet refuse tout sujet manquant. Il vérifie à nouveau les grilles, RAS/1 mm, valeurs finies et masques binaires. Le dossier est assemblé temporairement puis renommé une fois terminé.

- `imagesTr/<case>_0000.nii.gz` et `labelsTr/<case>.nii.gz` : pool d’entraînement/validation uniquement.
- `imagesTs/` : images des tests ID/OOD.
- `evaluation_labels/id/` et `evaluation_labels/ood/` : vérités terrain de test séparées de l’entraînement.
- `dataset.json` : canal T1, labels 0/1, nombre de cas d’entraînement.
- `splits_final.json` : plis personnalisés ; le copier dans `nnUNet_preprocessed/Dataset001_ATLAS/` après le prétraitement nnU-Net et avant tout entraînement. Sans ce fichier à cet emplacement, nnU-Net peut générer ses propres plis.

Les liens physiques économisent l’espace lorsque source et destination partagent le même système de fichiers. Les fichiers doivent être considérés comme immuables ; sur un autre disque, ils sont copiés. nnU-Net peut appliquer sa propre normalisation du canal T1 : garder la même configuration pour tous les modèles comparés. Le contrôle officiel `nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity` reste à exécuter dans l’environnement d’entraînement de Fayfa.

Références officielles : [format de dataset](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md), [plis personnalisés](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/manual_data_splits.md).

## Lecture du benchmark

Même sélection de 8 sujets pour chaque essai, données brutes relues, sorties nouvelles et reprise désactivée. Chaque essai exécute réorientation, rééchantillonnage, normalisation, recadrage et écriture. Les sorties temporaires sont supprimées après l’essai. Ce pilote limité à 8 petits volumes vérifie la taille estimée du lot et conserve 512 Mio libres ; le traitement normal garde sa réserve de 2 Gio et son estimation du lot complet. L’ordre des workers est mélangé avec une graine fixe ; le cache système n’est pas purgé.

Les CSV et la figure sont dans `preprocessing/reports/`. `compute_seconds` mesure lecture/calcul/écriture et le lancement des tâches Spark ; le démarrage de la JVM et du contexte Spark est exclu. `total_seconds` inclut le contexte et le nettoyage. L’accélération vaut temps(1 worker)/temps(N workers). Les workers ne sont pas des cœurs physiques réservés.

Un seul essai sur 8 sujets est un benchmark pilote, pas une estimation du temps total sur 1 453 sujets. Pour un résultat plus robuste, relancer `--repeats 3` sur une machine sans autre charge ; éviter de comparer des mesures obtenues pendant d’autres traitements.

### Mémoire du poste local

Les volumes de plus de 40 millions de voxels sont exécutés dans une partition séquentielle, après les autres sujets, pour limiter la pression mémoire. Le contrôle des valeurs des masques utilise des tranches plutôt que plusieurs tableaux temporaires complets. Le pilote de benchmark sélectionne des volumes de moins de 20 millions de voxels : son accélération ne représente donc pas celle des très gros volumes.

Le QC sauvegarde également ses résultats par sujet dans `_qc_cache/` pour reprendre les prochains lancements après une interruption. La reprise vérifie les entrées et les métadonnées utilisées ; les erreurs sont retentées.
