# Adaptation de domaine — boucle d'entraînement, DANN et TENT

Modules de la question RQ3 (§3.5) et de l'adaptation continue contrôlée (§5.1), sur la branche `adaptation-entrainement` (fusion de `adaptation-dann` et `adaptation-tent`) : une boucle d'entraînement MONAI 3D commune, version de travail du contrat C4, sur laquelle sont branchés les techniques de `generalization/`, DANN et TENT, et les notebooks Kaggle du tableau d'ablation.

| Fichier | Rôle |
|---|---|
| `train.py` | Boucle MONAI 3D commune : entraînement, validation sur le pli, `predict()`, évaluation par sujet (CSV), TENT au test |
| `dann.py` | Couche d'inversion de gradient, planification de Ganin, classifieur de centre, `DANNWrapper`, `dann_loss` |
| `tent.py` | `configure_tent`, `collect_params`, entropie par voxel dans le cerveau, `Tent` avec `reset()` |
| `continual.py` | Adaptation continue (version réduite) : centres OOD un par un, évaluation préquentielle, 4 stratégies (figé, auto-apprentissage, TENT, contrôlée), oubli |
| `aggregate_rq3.py` | Tableau RQ3 à partir des sorties des runs : Dice ID / OOD / écart (IC 95 % bootstrap), Wilcoxon contre la baseline, par centre et par taille de lésion, figures |
| `kaggle/` | Notebooks Kaggle (une run par technique), leur générateur et la préparation du dataset privé |
| `tests/` | Tests pytest sur de petits UNet MONAI 3D (CPU), le jeu C1 synthétique et de vrais patches C1 de 64³ |

```powershell
.venv-dl\Scripts\python -m pytest generalization\tests adaptation\tests -q
```

## Boucle d'entraînement commune (`train.py`)

```powershell
.venv-dl\Scripts\python -m adaptation.train --technique baseline --fold 0 --smoke      # essai local, GPU de 4 Go, ~35 s
.venv-dl\Scripts\python -m adaptation.train --technique fda --fold 0                   # run complète (Kaggle)
.venv-dl\Scripts\python -m adaptation.train --evaluate-only --checkpoint adaptation\runs\unet3d_baseline_pli0\best.pt --tent
```

Techniques : `baseline`, `intensity`, `bigaug`, `fda`, `histogram_matching`, `dann`. Tout le reste est identique d'une run à l'autre :

| Élément | Choix |
|---|---|
| Réseau | `UNet` MONAI 3D, canaux (32, 64, 128, 256, 320), 2 unités résiduelles, **normalisation d'instance affine** (nécessaire à TENT), sortie sigmoïde à un canal ; 12,9 M paramètres |
| Données | `load_processed` (C1), sujets d'entraînement du pli ; validation sur les centres de validation du pli |
| Patches | 96³ ; une fois sur deux centrés sur un voxel de lésion, sinon sur un voxel du cerveau ; miroir gauche-droite (commun à toutes les runs) |
| Augmentation | `build_transforms(…, fold=k, source="processed")` appliqué au **volume entier avant le découpage**, pour rester fidèle au calibrage (FDA en particulier) |
| Lot | 2 volumes × 2 patches |
| Perte | `DiceCELoss(sigmoid=True)` (+ perte du centre pour DANN) |
| Optimisation | AdamW, taux 1e-3, décroissance polynomiale (puissance 0,9), précision mixte |
| Durée | **40 000 itérations pour toutes les techniques** ; budget de 10 h comme garde-fou ; reprise exacte avec `--resume` (mêmes lots et mêmes tirages aléatoires qu'une run sans coupure) |
| Validation | Dice moyen sur la validation du pli toutes les 4 000 itérations ; `best.pt` = meilleur Dice de validation |
| Évaluation | test ID du pli et test OOD, fenêtre glissante gaussienne (recouvrement 0,25), probabilité mise à 0 hors du cerveau |
| Graine | 2026 (Python, NumPy, PyTorch, échantillonneur, workers) |

- **Même nombre d'itérations, pas même durée.** Préparer un volume coûte 0,16 s sans augmentation, 0,26 s avec l'intensité, 0,51 s avec BigAug et 0,59 s avec FDA (mesuré sur C1, un fil). Avec un budget de temps commun, FDA ferait 3 à 4 fois moins d'itérations que la baseline et la comparaison serait faussée.
- **Le GPU n'est pas le goulot** : le `UNet` MONAI sous-échantillonne dès la première couche ; un lot de 2 patches de 96³ prend 0,5 Go et 0,07 s sur la RTX 3050 locale. Le chargement et l'augmentation des volumes dominent : 4 workers sur Kaggle.
- **Points d'accroche** (`train(cfg, augment=…, wrap_model=…, extra_loss=…, test_transform=…)`) : par défaut ceux de la technique (`technique_hooks`) ; l'histogram matching est aussi appliqué à la validation et au test, puisque c'est un prétraitement.
- **Sorties** dans `adaptation/runs/<run>/` (ignoré par Git via `adaptation/.gitignore`) : `best.pt`, `last.pt`, `config.json`, `journal_entrainement.csv`, `resultats_id_test.csv`, `resultats_ood_test.csv` (une ligne par sujet : modèle, technique, pli, sujet, centre, découpage, délai post-AVC, Dice, HD95, différence de volume, F1 par lésion, différence du nombre de lésions, volumes) et `resume.json` (Dice moyen ID, OOD et écart). Les métriques sont celles de `baselines/common.py`, importées telles quelles.
- **TENT** : `--evaluate-only --tent` repart du modèle d'origine pour chaque centre, adapte sur 4 patches du cerveau de chaque sujet (sans la lésion), puis prédit ; écrit `resultats_*_tent.csv` et `resume_tent.json`.
- **`--smoke`** : petit réseau, 4 sujets d'entraînement, 2 de validation, 2 par découpage de test, 20 itérations ; vérifié sur la RTX 3050 pour les six techniques et pour TENT.

## Kaggle : tableau d'ablation RQ3

Six notebooks dans `kaggle/`, générés par `kaggle/make_notebooks.py` : **pli 0, graine 2026, 40 000 itérations, fp32, clipping à 12** pour tous.

| Notebook | Slug | Technique | Plateforme | Durée estimée (T4, fp32) |
|---|---|---|---|---|
| `00_baseline.ipynb` | `rq3-baseline-fp32-pli0` | baseline, puis TENT au test | Kaggle | 6 h 30 à 7 h 30 |
| `01_intensite.ipynb` | `rq3-intensity-pli0` | augmentation d'intensité | Colab | 7 h 30 à 9 h |
| `02_bigaug.ipynb` | `rq3-bigaug-pli0` | preset BigAug | Colab | 9 h à 10 h 30 |
| `03_fda.ipynb` | `rq3-fda-pli0` | FDA | Kaggle | 7 h à 9 h |
| `04_histogram_matching.ipynb` | `rq3-histogram-matching-pli0` | histogram matching (optionnelle) | Colab | 8 h à 9 h 30 |
| `05_dann.ipynb` | `rq3-dann-pli0` | DANN | Kaggle | 6 h 30 à 7 h 30 |
| `07_adaptation_continue.ipynb` | `rq3-adaptation-continue-pli0` | adaptation continue sur `best.pt` de la baseline | Kaggle ou Colab | 30 à 50 min |

Répartition dans `ASSIGNMENT` : Kaggle (2 sessions GPU en arrière-plan, quota hebdomadaire) et Google Colab (1 session, coupures possibles). `--colab-dir` écrit les notebooks Colab (`<slug>.ipynb`) : jeton Kaggle lu dans les secrets Colab (`KAGGLE_API_TOKEN`, ou `KAGGLE_USERNAME` et `KAGGLE_KEY`), datasets téléchargés par l'API Kaggle sur le disque de la machine, sorties et checkpoints sur Google Drive (`MyDrive/PROJET_AVC/<slug>/`), `last.pt` toutes les 20 min (écriture atomique), `--resume` systématique : relancer « Tout exécuter » après une coupure reprend la même trajectoire. Sur Colab (2 processeurs), le chargement des volumes limite la vitesse. Chaque notebook compare son environnement à celui de Kaggle (T4, torch 2.11.0+cu130) et le signale sans s'arrêter. Si l'itération 100 dépasse 0,75 s/it, le journal le signale (40 000 itérations ne tiendraient pas dans le budget de 10 h) sans arrêter la run. Les autres plis viendront ensuite si le budget le permet ; la référence de FDA et de l'histogram matching suit alors le pli (`fold_reference_sites`).

Chaque notebook s'arrête avec un message clair sans GPU, cherche récursivement sous `/kaggle/input` le code (dataset privé `projet-avc-code`) et les données (un dataset partagé peut être monté sous un autre chemin), vérifie que `VERSION.json` porte le commit attendu (copies Kaggle), installe MONAI et TorchIO, recrée `data/processed` (hors des sorties) par des liens vers les dossiers de sujets du dataset `atlas-c1-processed` (que Kaggle ait décompressé les archives ou non), vérifie que `splits.json` est identique dans les données et dans le code, puis lance `train.py`. Réglages Kaggle : accélérateur *GPU T4*, Internet activé, les deux datasets privés en entrée, notebook privé. Une session interrompue reprend avec `--resume` après avoir copié `last.pt` de la sortie précédente.

### Lancer un notebook depuis le poste (`kaggle kernels push`)

```powershell
.venv-dl\Scripts\python adaptation\kaggle\make_notebooks.py --kernels-dir data\kaggle_kernels   # un dossier par notebook
kaggle kernels push -p data\kaggle_kernels\03_fda --accelerator NvidiaTeslaT4 -t 43200
kaggle kernels output fayfahachicha/rq3-fda-pli0 -p data\kaggle_sorties\rq3-fda-pli0   # à la fin
.venv-dl\Scripts\python -m adaptation.aggregate_rq3 data\kaggle_sorties                  # tableau RQ3 et figures
```

`kernel-metadata.json` : notebook privé, GPU, Internet, datasets `fayfahachicha/atlas-c1-processed` et `fayfahachicha/projet-avc-code`. Le CLI Kaggle 2.2.5 s'installe avec `uv tool install kaggle --with six` (dépendance `six` oubliée par le paquet) ; sous Windows, lancer `kaggle datasets create` depuis le dossier du dataset (`-p .`), sinon le CLI échoue sur un nom de fichier temporaire.

### Dataset de code privé

```powershell
.venv-dl\Scripts\python adaptation\kaggle\build_code_dataset.py      # code commité seulement (git archive HEAD)
cd data\kaggle_projet_avc_code ; kaggle datasets create -p . -r skip                 # première fois
cd data\kaggle_projet_avc_code ; kaggle datasets version -p . -r skip -m "message"   # après chaque commit utile
```

`projet_avc_code.zip` contient `generalization/`, `adaptation/`, `baselines/common.py` (métriques importées par `train.py`), `analysis/center_distance.py` (distance de domaine utilisée par `continual.py`), `splits.json` et `VERSION.json`. Le script refuse de construire si `generalization/` ou `adaptation/` ont des modifications non commitées, et si l'archive contient une image, un volume ou un checkpoint.

### Dataset Kaggle privé des données

```powershell
.venv-dl\Scripts\python adaptation\kaggle\build_dataset.py --username <identifiant-kaggle>   # ~20 s, 3,85 Go
```

Écrit dans `data\kaggle_atlas_c1\` (exclu de Git) : `processed_partie1.zip` (745 sujets, 2,00 Go) et `processed_partie2.zip` (670 sujets, 1,85 Go), stockés sans recompression ; `subjects.parquet`, `splits.json`, `manifeste.json` et `dataset-metadata.json` (titre, identifiant, mention de l'accord d'utilisation ATLAS). Le téléversement se fait à la main :

```powershell
uv tool install kaggle                                   # CLI Kaggle, isolé de .venv-dl
# Jeton : kaggle.com > Settings > API > Create New Token, puis placer kaggle.json dans %USERPROFILE%\.kaggle\
cd data\kaggle_atlas_c1 ; kaggle datasets create -p . -r skip     # privé par défaut : ne jamais ajouter --public
kaggle datasets status <identifiant-kaggle>/atlas-c1-processed
```

Les notebooks Colab lisent les deux datasets privés avec le jeton Kaggle du propriétaire : aucun partage nécessaire.

La cellule « données » des notebooks a été vérifiée en local sur un faux `/kaggle/input` construit avec le jeu C1 synthétique (archives décompressées ou non, trois parties), suivie d'un entraînement `--smoke` sur le dossier reconstitué. Elle n'a pas encore tourné sur Kaggle même.

## DANN (Ganin et al., *JMLR* 2016)

**Réglage RQ3 (v2, run `rq3-dann-v2-pli0`)** : la première run Colab divergeait (perte totale 10 508 à l'itération 100, 33 M à 400). La perte du centre (poids 1, ~ln 39 = 3,7 au départ) dominait la perte totale et le clipping, et le classifieur lisait les caractéristiques du bottleneck sans normalisation : l'encodeur pouvait en faire exploser l'échelle. Correctif : LayerNorm (sans paramètres) après le pooling global, `dann_weight` = 0,1, lambda nul sur les 5 premiers % puis montée de Ganin avec gamma = 5, `perte_segmentation`, `perte_centre` et lambda à chaque ligne de progression, arrêt si la perte totale dépasse 100. Essai local (RTX 3050, 300 itérations, lot complet) : perte de segmentation 1,63 → 1,07 comme la baseline, perte du centre entre 3,5 et 3,9, y compris avec lambda montant à 0,78.

```python
from monai.networks.nets import UNet
from adaptation.dann import DANNWrapper, center_labels, dann_loss

unet = UNet(spatial_dims=3, in_channels=1, out_channels=1, channels=(32, 64, 128, 256), strides=(2, 2, 2))
centres = sorted(centres_d_entrainement_du_pli)          # jamais un centre de validation ou de test
model = DANNWrapper(unet, n_centers=len(centres))

for epoch in range(n_epochs):
    model.set_progress(epoch / n_epochs)                  # lambda de 0 vers 1
    for batch in loader:
        segmentation, logits = model(batch["image"])
        loss, parts = dann_loss(segmentation, batch["label"], logits, center_labels(batch["site"], centres))
        loss.backward(); optimizer.step(); optimizer.zero_grad()
```

- **Couche d'inversion de gradient** : identité à l'aller ; au retour, le gradient est multiplié par −λ. Le classifieur apprend à reconnaître le centre, l'encodeur reçoit le gradient opposé et apprend à effacer l'empreinte du centre.
- **Planification de Ganin** : λ = 2 / (1 + exp(−10 p)) − 1, p = avancement de 0 à 1. La montée progressive est la solution de repli prévue si DANN est instable.
- **Classifieur de centre** : moyenne spatiale des caractéristiques du bottleneck, puis MLP (64 neurones, dropout 0,25). Le bottleneck est capté par un hook (`find_bottleneck` trouve le module le plus profond d'un UNet MONAI), sans modifier le réseau ; pour un autre réseau, passer `bottleneck=` et `feature_channels=`.
- **Perte** : Dice + CE (`DiceCELoss(sigmoid=True)` de MONAI, sortie binaire à un canal) + `weight` × CE du centre, avec `weight = 1`. Le λ de Ganin agit dans la couche d'inversion : l'encodeur reçoit bien −λ × ∇CE_centre, et le classifieur apprend à pleine vitesse. Mettre λ aussi dans la perte le ralentirait au début (λ ≈ 0) ; `weight` reste réglable.
- **Centres** : `center_labels` indexe les centres d'entraînement du pli ; un centre absent de la liste est refusé.

## TENT (Wang et al., *ICLR* 2021)

```python
from adaptation.tent import Tent, collect_params, configure_tent

model = configure_tent(modele_entraine)                       # seules les normalisations apprennent
tent = Tent(model, torch.optim.Adam(collect_params(model), lr=1e-3), steps=1)
for batch in nouveau_centre:
    logits = tent(batch["image"], brain=batch["brain"])       # sans étiquettes
tent.reset()                                                  # retour exact à l'état d'origine
```

- **`configure_tent`** : mode entraînement ; tout est gelé sauf les paramètres affines (gamma, beta) des BatchNorm, InstanceNorm et GroupNorm. Les BatchNorm utilisent les statistiques du lot courant (`track_running_stats=False`).
- **Perte** : entropie moyenne par voxel **dans le cerveau** (`brain`), sigmoïde pour une sortie à un canal, softmax sinon. Le fond, prédit avec certitude, écraserait sinon la moyenne.
- **`Tent`** : chaque appel évalue le lot puis fait `steps` pas de minimisation ; il renvoie les logits avant la mise à jour, comme l'implémentation de référence (évaluer d'abord, apprendre ensuite). `history` garde l'entropie de chaque pas.
- **`reset()`** restaure exactement les poids et l'état de l'optimiseur pris à la création de `Tent`.

> **Hypothèse forte : le réseau doit avoir des normalisations affines.** Le `UNet` MONAI par défaut utilise `InstanceNorm3d` **sans** paramètres affines : TENT n'aurait rien à adapter, et `configure_tent` lève une erreur claire. La boucle C4 doit utiliser `norm=("instance", {"affine": True})` (comme nnU-Net), BatchNorm ou GroupNorm. À convenir avec Anas.

## Démo (contrat C8)

- Brancher le modèle sur la démo (`demo/app.py`) : `AVC_PREDICT_ADAPTER=adaptation.demo_adapter:predict`, lancé depuis la racine du dépôt (le module `adaptation` doit être importable).
- Checkpoint : `data/modeles/unet3d_intensity_pli0_best.pt` (ignoré par Git), ou un autre chemin dans `PROJET_AVC_MODELE`.
- Le masque est écrit en NIfTI uint8 dans la grille de l'IRM reçue, dans `AVC_MASK_DIR` s'il est défini, sinon dans un dossier temporaire.

## Vérification

24 tests dans `adaptation/tests`, environ 25 secondes sur CPU (petits UNet MONAI 3D, volumes de 32³) :

- **Boucle** (`test_train.py`, jeu C1 synthétique) : entraînement puis évaluation pour la baseline et DANN (fichiers de sortie, colonnes du CSV, journal, perte du centre), rechargement du checkpoint et évaluation TENT, reprise depuis `last.pt` avec les mêmes poids qu'une run sans coupure (baseline et intensité, 0 et 2 workers, CPU), suite d'indices identique à `RandomSampler`, sauvegarde atomique, `predict()` binaire et nul hors du cerveau (volume plus petit que le patch compris), patch centré sur la lésion, technique inconnue refusée.

- **DANN** : gradient inversé et multiplié par λ ; planification de Ganin (0 à p = 0, croissante, bornée) ; formes de la segmentation et des logits ; le classifieur lit bien la sortie du bottleneck ; sur l'encodeur, le gradient de la perte du centre avec inversion est l'opposé de celui sans inversion, et nul pour λ = 0, alors que celui du classifieur ne change pas ; un pas d'optimisation met à jour segmentation, encodeur et classifieur ; poids de la perte du centre.
- **TENT** : avec BatchNorm, InstanceNorm affine et GroupNorm, seuls les paramètres de normalisation sont entraînables et seuls eux changent ; BatchNorm sur les statistiques du lot ; l'entropie baisse sur un lot synthétique ; `reset()` restaure exactement les poids, l'optimiseur et la sortie ; refus du UNet MONAI par défaut ; entropie (0 pour une prédiction certaine, ln 2 au maximum, restreinte au cerveau).
- **Vrais volumes C1** (ignorés si `data/processed` est absent) : 3 sujets d'entraînement de centres différents, patches de 64³ centrés sur la lésion. DANN : cinq pas d'optimisation, pertes finies, la perte de segmentation baisse. TENT : l'entropie dans le cerveau baisse en 8 pas, seules les normalisations bougent, `reset()` restaure tout.

Ces tests vérifient la mécanique, pas l'efficacité : un réseau non entraîné sur trois patches ne dit rien du gain de DANN ou de TENT sur les centres OOD.

## Lien avec la boucle d'Anas (contrat C4)

`train.py` est une version de travail du contrat C4, faite pour ne pas bloquer la RQ3 : il ne remplace pas nnU-Net ni les Transformers d'Anas (RQ1, RQ2). Si sa boucle MONAI arrive, il suffira d'y reprendre les points d'accroche :

1. **DANN** : envelopper le réseau dans `DANNWrapper`, passer le centre de chaque sujet (clé `site` de `load_processed`) et appeler `set_progress` à chaque itération ou époque. Le classifieur ne doit connaître que les centres d'entraînement du pli.
2. **Normalisations affines** dans le réseau, pour TENT.
3. **Modèles « avant / après DANN »** pour le test d'empreinte de Yosra (§5.5) : `best.pt` des runs baseline et DANN.
4. **TENT sur les centres OOD** : un `Tent` par centre arrivant, `reset()` entre deux centres pour la stratégie « TENT seul » de l'adaptation continue (§5.1) ; déjà fait par `--evaluate-only --tent`.
5. **Réglages** (λ maximal, γ de Ganin, taux d'apprentissage et nombre de pas de TENT, nombre d'itérations) : sur la validation des plis seulement, jamais sur le test OOD.
6. **Comparabilité avec Anas** : les CSV par sujet utilisent ses métriques (`baselines/common.py`) et les découpages de `splits.json` ; sa baseline 2.5D utilise encore son propre découpage.
