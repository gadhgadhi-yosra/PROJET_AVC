# Exécuter les dernières analyses du pipeline

Le code C1, généralisation, DANN et TENT est intégré. L’analyse S8 historique est produite.
La vérification Git et locale du 12 novembre n’a trouvé ni nouveau commit distant,
ni checkpoint entraîné, ni CSV de scores C1 v2. Aucun entraînement supplémentaire
n’a été exécuté et aucun score n’a été remplacé par une simulation.

## Une commande pour contrôler, une pour exécuter S7/S8

```sh
.venv-spark/bin/python integration/run_remaining.py
.venv-spark/bin/python integration/run_remaining.py --run
```

Configurer `integration/experiments.json` avec le nom réel du modèle, sa technique,
les plis effectivement évalués et le chemin du CSV. Les valeurs `MODELE_C1` sont
explicitement des champs à compléter, pas des modèles existants. Un CSV peut regrouper
plusieurs plis ; le filtre modèle/technique/pli est appliqué séparément.

Le contrôle exige une correspondance avec le split réellement utilisé et une livraison
complète des scores de validation, test ID du pli et test OOD. Il refuse les scores du train,
les doublons et les identifiants inconnus. Le hash du split doit être fourni avec la
provenance de l’entraînement : le modifier pour faire passer d’anciens scores invaliderait
l’expérience. Le contrôle vérifie cette déclaration, pas l’historique réel du GPU.

Chaque étape S7/S8 conserve son journal, son statut et ses sorties sous
`data/analysis/final_runs/` (privé, ignoré par Git). Une étape échouée n’empêche pas l’autre
de s’exécuter ; le code retour global reste non nul en cas d’entrée manquante ou d’échec.
L’état est lisible dans `integration/reports/remaining_readiness.json`.

## Livraison attendue de l’équipe

| Élément | Nécessaire pour |
|---|---|
| CSV `model,technique,fold,case_id,dice`, scores val/ID/OOD | S7 distance–Dice, LOCO et S8 délai–Dice sur C1 |
| Split réellement utilisé + SHA-256, protocole de calcul du Dice | Vérifier la provenance et traiter correctement les masques vides |
| Checkpoints baseline/augmentation/DANN, architecture et paramètres | Extraction réseau et comparaison S8–S10 |
| Couche d’extraction et prétraitement de chaque modèle | Comparer des représentations compatibles |
| Modèle final et fonction retournant un masque dans la grille native | Démo S11 et vidéo d’inférence S13–S14 |

Le contrat des adaptateurs est détaillé dans `analysis/ANALYSES_S7_S10.md` et
`demo/README.md`. Le lancement ci-dessus couvre les analyses de scores S7/S8 ; il
ne prétend pas entraîner un modèle ni charger un checkpoint d’architecture inconnue.

## Message prêt à copier à l’équipe

« J’ai intégré les branches C1, généralisation, DANN et TENT et vérifié le raccordement.
Pour terminer mes analyses et la démo, pouvez-vous déposer les scores par cas et par pli,
les checkpoints et leurs architectures/prétraitements, avec la version exacte du split ?
Les CSV historiques disponibles utilisent une autre partition que C1 v2. Merci aussi
de préciser la convention Dice des masques vides et la couche à utiliser pour les features. »

Ce message est préparé, **pas envoyé**. Les répétitions et la relecture finale restent
à faire avec l’équipe ; le support actuel est `Soutenance_Projet_AVC.pptx`.
