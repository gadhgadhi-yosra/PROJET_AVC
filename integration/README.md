# Intégration du pipeline et des modèles

Les branches `generalisation-chargeur-c1` (`214647d`), `adaptation-dann` (`d0a9a93`)
et `adaptation-tent` (`5834f0e`) sont fusionnées localement dans `exploration-donnees`.
Elles incluent les baselines d’Anas (`00e2692`). Les modifications du pipeline sont conservées,
avec une sauvegarde Git préalable dans un stash.
Aucun push n’est effectué.

## Connexion des données

Le chargeur de Fayfa lit directement les sorties du pipeline : `data/processed/subjects.parquet`,
les trois volumes par sujet et le **split v2** actif. Aucune copie ni renormalisation n’est nécessaire.

```python
from generalization.data_processed import load_processed
from generalization.transforms import build_transforms

train = load_processed('data/processed', split='train', fold=0, splits='splits.json')
validation = load_processed('data/processed', split='val', fold=0, splits='splits.json')
# Les augmentations ne sont appliquées qu'au train.
transform = build_transforms('generalization/configs/baseline.yaml', source='processed', fold=0)
subject = transform(train[0])  # t1, mask, brain, site, case_id
```

`adaptation.dann.DANNWrapper` utilise les centres du **train du pli** pour son classifieur.
`adaptation.tent.Tent` utilise le masque cérébral et doit être réinitialisé entre centres
pour une adaptation indépendante. Les tests fournis vérifient aussi ces modules sur
les vrais patches C1 ; ils ne constituent pas une expérience de segmentation entraînée.

## Environnement de validation sur Mac Intel

Les dépendances CUDA Windows de Fayfa restent inchangées dans
`generalization/requirements-dl.txt`. L’environnement CPU séparé `.venv-integration`
utilise Python 3.11 ; sa liste figée est `requirements-macos-cpu.lock.txt`.

```sh
MPLCONFIGDIR=/tmp/avc-mpl OMP_NUM_THREADS=2 .venv-integration/bin/python integration/check_team.py
MPLCONFIGDIR=/tmp/avc-mpl OMP_NUM_THREADS=2 .venv-integration/bin/python -m pytest generalization/tests adaptation/tests -q
```

## Résultats historiques et checkpoints

Les CSV récupérés dans `baselines/outputs/fast_results/` contiennent de vrais résultats
versionnés, sur **un ancien protocole** (903 train, 226 ID validation, 287 OOD).
Ils ne sont pas les résultats des cinq plis du pipeline. `check_team.py` rattache les identifiants
à C1 sous `data/predictions/historical_baselines/` et produit les chevauchements agrégés dans
`reports/team_compatibility.json`, sans leur inventer un numéro de pli.
Ne pas les passer aux analyses S7 comme s’ils provenaient de C1 v2.

Aucun checkpoint `.pt`, `.pth` ou `.ckpt` n’est présent dans les branches récupérées.
Les analyses réseau et l’inférence de la démo attendent toujours les poids entraînés
et leur architecture, prétraitement et partition exacts. DANN et TENT sont des modules,
pas des poids déjà entraînés.

La calibration reçue de Fayfa correspondait au split v1. Elle est archivée ; la calibration
active a été réellement recalculée sur les 916 sujets disponibles du split v2. Les plages
gamma [0,68 ; 1,79], contraste [0,77 ; 1,53] et les références par pli restent identiques.
Son empreinte correspond maintenant au split actif.

## Recalcul après changement du split

L’ancienne calibration est archivée dans `archive/calibration_v1.json`. Pour recalculer
les statistiques C1 et synchroniser les configurations :

```sh
MPLCONFIGDIR=/tmp/avc-mpl OMP_NUM_THREADS=1 .venv-integration/bin/python -m generalization.calibration --source processed --workers 2 --stats generalization/figures/calibration/stats_processed_v2.csv
.venv-integration/bin/python integration/sync_calibration.py
```

Utiliser un nouveau fichier `--stats` si les volumes ou les sujets changent : le cache
contient des mesures individuelles et ne doit pas être repris après une modification des données.
Les observations des balayages FDA/BigAug héritées de Fayfa conservent explicitement
leur provenance v1 ; elles ne sont pas présentées comme un nouveau balayage v2.
Le pilote `generalization/preprocess_c1_local.py` applique maintenant aussi
`preprocessing/qc_exclusions.json`, comme le pipeline Spark.

## Vérification du 18 novembre 2026

Après fusion des PR #6 à #10 dans `main` : **310 tests réussis** (50 pipeline, 7 frontend,
224 généralisation/adaptation dont les patches C1 réels, 29 calibration). Sur ce Mac Intel,
14 tests d'entraînement (`adaptation/tests/test_train.py`, `test_continual.py`) échouent car
`torch.amp.GradScaler` n'existe pas dans torch 2.2.2, dernière version disponible pour macOS Intel.
L'entraînement tourne sur Kaggle ; l'inférence du checkpoint C8 est vérifiée localement.

## Vérification finale du 12 novembre 2026

**242 tests réussis** : 34 pour le pipeline, 179 généralisation/adaptation (dont les patches
C1 réels), 29 calibration. Le chargeur a été contrôlé sur les 1 415 cas et les cinq plis.
Les dépendances CPU passent `pip check`. Aucun conflit Git ne reste ouvert.
Résultats détaillés : `reports/verification.json` et `reports/team_compatibility.json`.

## Lancer S7/S8 après réception des scores

Configurer `experiments.json`, puis exécuter :

```sh
.venv-spark/bin/python integration/run_remaining.py
.venv-spark/bin/python integration/run_remaining.py --run
```

Le premier appel audite les entrées ; le second exécute les expériences prêtes.
Quatre tests vérifient les cas manquants, le mauvais split, les livraisons incomplètes
et une livraison complète. État actuel : `reports/remaining_readiness.json`.
Détails : `docs/pipeline/PROCHAINE_ETAPE.md`.
