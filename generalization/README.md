# Généralisation inter-centres — bibliothèque de transformations (contrat C5)

Techniques de généralisation de la question RQ3 (§3.5 du document de projet) : augmentation d'intensité, randomisation du style, méthode de Fourier et harmonisation. Tout fonctionne sur les **données prétraitées** du contrat C1 (`load_processed`) et, en repli, sur les **données brutes** ATLAS v3 (`data_raw.py`).

État au 4 novembre 2026 :

- **données C1 produites et vérifiées sur cette machine** : 1 415 sujets prétraités par le code de Yosra (un échec, `R065_sub-r065s019`) ; `load_processed` les charge sur tous les découpages sans erreur de format ([détail](#données-c1-sur-cette-machine)) ;
- **calibrage refait sur C1**, la source de l'entraînement : gamma 0,68 – 1,79, contraste 0,77 – 1,53, référence R059 (par pli : R059, R071, R059, R004, R071), `beta` de FDA et gamma de BigAug re-balayés ([détail](#recalibrage-sur-c1-du-4-novembre-2026)). Un test échoue si un centre de test sert à un réglage ;
- chargeur C1 (`data_processed.py`) ; les scripts acceptent `--source raw|processed` ;
- les 187 tests passent sur des volumes synthétiques (3 de plus se lancent avec les données réelles) ;
- le chargement brut a été vérifié sur les 1 416 sujets réels (14 octobre) ;
- les transformations ont été regardées sur un sujet d'entraînement de six centres (R015, R029, R014, R030, R039, R040), en C1 ;
- deux défauts du chargement ont été corrigés le 14 octobre (lecture de deux masques de R040, rééchantillonnage des volumes déjà à 1 mm).

Ce qui reste à faire est [en fin de document](#reste-à-faire).

| Fichier | Rôle |
|---|---|
| `data_processed.py` | Charge les données prétraitées du contrat C1 en `torchio.Subject`, par découpage et par pli de `splits.json` |
| `data_raw.py` | Repli : liste et charge les sujets bruts en `torchio.Subject` (RAS, 1 mm, z-score dans le cerveau) |
| `preprocess_c1_local.py` | Produit les données C1 sous Windows avec le code de Yosra tel quel, sans Spark |
| `transforms/` | Les transformations et `build_transforms(config)` |
| `configs/*.yaml` | Une configuration par famille, plus `baseline.yaml` (aucune transformation) |
| `contrast.py` | Rapport d'intensité lésion / tissu voisin |
| `synthetic.py` | Volumes synthétiques qui imitent les pièges des données réelles |
| `calibration.py`, `calibration.json` | Calibrage sur les seuls sujets d'entraînement de `splits.json` ; garde-fou contre les centres de test |
| `visual_check.py` | Figures avant / après, mesure du contraste de la lésion, balayages de FDA et du gamma |
| `benchmark.py` | Temps CPU par transformation |
| `tests/` | Tests pytest, sans aucune donnée patient |

## Installation sous Windows

Python 3.11 suffit (le 3.12 n'est requis que pour l'environnement Spark). L'environnement est séparé de `.venv-spark` :

```powershell
python -m venv .venv-dl
.venv-dl\Scripts\python -m pip install --upgrade pip
.venv-dl\Scripts\python -m pip install -r generalization\requirements-dl.txt
.venv-dl\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.venv-dl\Scripts\python -m pytest generalization\tests -q
```

PyTorch est installé dans sa version CUDA 12.6 (téléchargement d'environ 2,5 Go). Sans GPU NVIDIA, retirer la ligne `--extra-index-url` et le suffixe `+cu126` de `requirements-dl.txt`. Toutes les commandes se lancent depuis la racine du dépôt.

## Utilisation

```python
from generalization.data_processed import load_processed
from generalization.transforms import build_transforms

subjects = load_processed("data/processed", split="train", fold=0)   # torchio.Subject : t1, mask, brain, site…
transform = build_transforms("generalization/configs/intensity.yaml")
augmented = transform(subjects[0])                                   # le sujet d'entrée n'est pas modifié
```

Repli sur les données brutes, tant que C1 n'est pas sur la machine :

```python
from generalization.data_raw import list_subjects, load_subject

records = list_subjects("data/ATLAS3_Training_Raw")   # table pandas, centre R032 exclu
subject = load_subject(records.iloc[0])                # mêmes clés : t1, mask, brain, site, subject…
```

Une configuration est une liste de transformations, chacune avec un nom, une probabilité `p` et ses plages `[min, max]` :

```yaml
transforms:
  - name: gamma
    p: 0.3
    gamma: [0.71, 1.77]
  - name: noise
    enabled: false        # désactivée sans supprimer ses paramètres
    p: 0.2
    std: [0.0, 0.1]
```

Pour le tableau d'ablation, plusieurs configurations se combinent : `build_transforms(["…/intensity.yaml", "…/fda.yaml"])`. En validation croisée, passer le pli : `build_transforms(config, fold=k)` prend la référence de FDA et de l'histogram matching choisie hors des centres de validation du pli k (voir [Référence pli par pli](#référence-pli-par-pli)).

**Boucle MONAI (contrat C4).** `build_transforms(config, monai=True)` renvoie un objet qui prend et rend un dictionnaire `{"image", "label"}` en canal d'abord, utilisable dans un `monai.transforms.Compose`. Trois points à connaître :

- **Masque du cerveau.** Une clé `"brain"` est utilisée si elle existe. Sinon le cerveau est estimé par les voxels différents de la valeur du fond.
- **Graine.** L'aléa vient du générateur de PyTorch : `torch.manual_seed` ou `monai.utils.set_determinism` suffisent.
- **Images de référence.** FDA et l'histogram matching lisent leurs images de référence dans `data_root` (argument de `build_transforms`, sinon clé `data_root` de la configuration). Avec la clé `splits: splits.json` de leur entrée, un centre de référence de test est refusé et seuls les sujets d'entraînement servent de référence.

## Données prétraitées (contrat C1)

`load_processed(root, split="train", fold=None)` lit la sortie de `preprocessing/week4.py` (Yosra) : `<root>/<sujet>/{t1,mask,brain}.nii.gz`, `qc.json` et la table `<root>/subjects.parquet` (statut `ok` seulement). Elle renvoie une liste de `torchio.Subject` avec les clés `t1`, `mask`, `brain`, `site`, `subject`, `case_id`, `empty_mask`, `norm_mean` et `norm_std`.

- **Découpages** : `split` vaut `"train"`, `"id_test"` ou `"ood_test"` ; avec `fold=k`, `"train"` et `"val"` sont ceux du pli k et `"id_test"` le test ID du pli. `split="val"` sans pli est refusé.
- **Lecture** : nibabel, à la demande (les volumes ne sont lus qu'à `subject["t1"].data`) ; les sujets sont sérialisables pour les `DataLoader`. Le z-score stocké en int16 avec `scl_slope = 0,001` est relu en float32.
- **Masques** : lésion seuillée à `> 0.5` ; cerveau lu dans `brain.nii.gz`, jamais déduit de « > 0 » (après z-score, la moitié du cerveau est négative).
- **Contrôles dès la liste** (en-têtes seulement) : orientation RAS, voxels de 1 mm, même grille pour les trois volumes. Sinon `ProcessedFormatError` avec le sujet et le fichier en cause.
- **Masques vides** : signalés par un avertissement et gardés comme cas négatifs (`empty_mask`), jamais une erreur.
- **Sujets manquants** : un sujet du découpage sans volumes est une erreur ; `on_missing="warn"` l'écarte avec un avertissement (cas de `R065_sub-r065s019`, en échec côté C1).
- **R032** est toujours exclu.
- **Normalisation réversible** : `norm_mean` et `norm_std` viennent de `qc.json` (`raw_mean`, `raw_std`) : brut = z × `norm_std` + `norm_mean`, comme avec `data_raw`.

`build_transforms(config, source="processed")` prend les images de référence de FDA et de l'histogram matching dans les données C1 (sujets d'entraînement, du pli avec `fold=k`). Les scripts acceptent `--source raw|processed` :

```powershell
.venv-dl\Scripts\python -m generalization.calibration --source processed   # espacement natif lu dans les données brutes
.venv-dl\Scripts\python -m generalization.visual_check --source processed --sites R015 R029 R014
.venv-dl\Scripts\python -m generalization.benchmark --source processed --data-root data\processed   # vrais volumes C1
```

Avec `--source processed`, la calibration reconstruit les intensités brutes dans le cerveau (brut = z × `raw_std` + `raw_mean`), après rééchantillonnage à 1 mm : les grandeurs sont proches de celles des données brutes, pas identiques. Le critère « 1 mm isotrope » du centre de référence porte sur l'espacement natif, lu dans les en-têtes bruts ; sans données brutes, il ne peut pas être évalué. `calibration.json` note la source (`"source"`) ; les entrées `fda` et `histogram_matching` portent `source: processed`, et un test vérifie que les deux concordent.

### Données C1 sur cette machine

Les données C1 ont été produites le 4 novembre 2026 sur ce poste Windows, avec le code de Yosra tel quel. Son lanceur (`preprocessing/run_week4.sh`) vise macOS et Spark exige `winutils.exe` sous Windows (`addPyFile` échoue sans `HADOOP_HOME`) : `preprocess_c1_local.py` appelle donc ses fonctions `pipeline.inventory` et `week4.work` sans les modifier, avec des processus Python au lieu de Spark et la même règle pour les volumes de plus de 40 M voxels (un par un, à la fin).

```powershell
uv venv --python 3.12 .venv-spark                                        # environnement du pipeline, ignoré par Git
uv pip install --python .venv-spark\Scripts\python.exe -r requirements.txt   # requirements.txt racine, sans modification
.venv-spark\Scripts\python generalization\preprocess_c1_local.py --workers 2   # ~50 min, 3,6 Go dans data\processed
```

Attention : `subjects.parquet` ne décrit que le dernier lot traité. Un essai avec `--cases` doit écrire ailleurs (`--output`), sinon il remplace la table complète.

Résultat (1 453 sujets, 50 min avec 2 processus) :

- **1 415 sujets `ok`**, 37 exclus (R032), **1 échec : `R065_sub-r065s019`** (`lesion_outside_image_based_crop` : plus de 1 % de la lésion hors du recadrage calculé sur l'image). Ce sujet appartient au pool d'entraînement (validation du pli 0) ; `on_missing="warn"` l'écarte. À traiter côté C1.
- **`load_processed` sur tous les découpages** : entraînement 916/917, test ID 231/231, test OOD 268/268, plis 732 à 734 sujets ; aucune erreur de format (RAS, 1 mm, grilles), aucun sujet orphelin, identifiants conformes à `splits.json`.
- **Masques vides** : les 5 connus (R041 s108 et R065 s002 en entraînement ; R053 s016, SOOP 0910 et 0961 en test OOD), gardés comme cas négatifs.
- **Validateur de Yosra** (`preprocessing/validate_s4.py`) : sur un essai de 92 sujets, `sub-r002s001` est rejeté de justesse (moyenne du cerveau 1,24e-4 et écart-type 0,99987 pour une tolérance de 1e-4) ; c'est l'effet du z-score stocké en int16 (pas de 0,001), pas une erreur. Signalé à Yosra.

## Ce que fait le chargement provisoire (données brutes)

`load_subject` lit les fichiers avec nibabel, comme l'exploration et le pipeline Spark, puis applique dans l'ordre :

1. **Vérification de la grille** : erreur si l'image et le masque n'ont pas la même grille.
2. **Masque du cerveau** : voxels strictement au-dessus de la valeur du fond (la valeur la plus fréquente), et non `> 0`. À R029 le fond vaut environ 0,0007 et il existe des valeurs négatives.
3. **Masque de lésion** : `mask > 0.5`, jamais `== 1` (certains masques valent 0,99999998).
4. **Réorientation RAS** puis **rééchantillonnage à 1 mm** : interpolation linéaire pour l'image, plus proche voisin pour les masques. Un volume déjà à 1 mm garde sa grille ; les autres passent sur une grille qui couvre tout le champ de vue.
5. **Z-score dans le cerveau**, fond mis à 0. Les valeurs `norm_mean` et `norm_std` sont gardées dans le sujet pour revenir aux intensités brutes.

> **Deux masques de R040 mal lus par SimpleITK.** Les masques de `sub-r040s053` et `sub-r040s087` n'ont pas de qform. SimpleITK, le lecteur par défaut de TorchIO et de nnU-Net, rejette leur sform et les lit retournés gauche-droite par rapport à leur image. Avec nibabel, image et masque sont sur la même grille. Tout outil qui lit les données brutes avec SimpleITK doit traiter ces deux sujets à part.

`list_subjects` exclut le centre R032 par défaut (`exclude_sites=()` pour le garder). Les 5 masques vides connus sont marqués `known_empty_mask` ; tout autre masque vide déclenche un avertissement au chargement.

## Transformations

Les transformations d'intensité ne modifient jamais le masque, gardent la forme et l'affine, et laissent le fond intact. Elles n'agissent que dans le cerveau.

| Nom YAML | Ce qu'elle fait | Défaut | Référence |
|---|---|---|---|
| `gamma` | Gamma sur le cerveau ramené à [0, 1], puis retour à l'échelle d'origine | `gamma: [0.71, 1.77]` | Isensee et al., nnU-Net, *Nature Methods* 2021 |
| `contrast` | Étire les intensités autour de la moyenne du cerveau | `factor: [0.78, 1.52]` | idem |
| `noise` | Bruit gaussien, en fraction de l'écart-type du cerveau | `std: [0, 0.1]` | idem |
| `blur` | Flou gaussien | `sigma_mm: [0.25, 1.0]` | idem |
| `bias_field` | Champ multiplicatif lisse (exponentielle d'un polynôme d'ordre 3) | `coefficients: 0.2` | Van Leemput et al., *IEEE TMI* 1999 ; Pérez-García et al., TorchIO, 2021 |
| `low_resolution` | Moyenne vers une grille grossière puis retour ; image seule | `isotropic_mm: [1, 1.5]`, `axis_mm: [1, 3]` | Isensee et al. 2021 |
| `randconv` | Convolution 3D aléatoire à une couche, mélangée à l'original | `kernel_sizes: [1, 3, 5, 7]`, `mix: [0, 0.5]` | Xu et al., RandConv, ICLR 2021 |
| `gin` | Petit réseau convolutif aléatoire (4 couches), mélangé à l'original | `mix: [0, 0.5]` | Ouyang et al., GIN, *IEEE TMI* 2022 |
| `fda` | Échange l'amplitude basse fréquence avec une image d'un autre centre, phase conservée | `beta: [0.015, 0.03]` | Yang & Soatto, FDA, CVPR 2020 |
| `histogram_matching` | Donne au cerveau l'histogramme moyen du centre de référence ; déterministe | `p: 1` | Nyúl, Udupa & Zhang, *IEEE TMI* 2000 (variante par quantiles) |
| `sharpen`, `brightness` | Masque flou ; décalage et échelle d'intensité (preset BigAug) | voir `bigaug.yaml` | Zhang et al., BigAug, *IEEE TMI* 2020 |
| `affine`, `elastic`, `flip` | Transformations **spatiales** de TorchIO (preset BigAug) : elles déplacent aussi le masque | voir `bigaug.yaml` | Pérez-García et al. 2021 |

Le preset `bigaug.yaml` empile flou, netteté, bruit, luminosité, gamma et transformation affine avec `p = 0.5` chacune. La déformation élastique y figure mais est désactivée (lente sur CPU).

## Calibrage

**Règle : aucun réglage ne voit un centre de test.** Tous les réglages sont faits sur les seuls **sujets d'entraînement** de `splits.json` (917 sujets, 48 centres ; 916 en C1), sans les sujets du test ID, sans les six centres OOD (R008, R028, R031, R034, R053, SOOP) ni le centre exclu R032. Le module `calibration.py` lit `splits.json`, refuse tout centre tenu à l'écart et écrit le résultat agrégé par centre dans `calibration.json` (versionné). Les mesures par sujet restent en local dans `generalization\figures\calibration\`. `visual_check.py` se limite aussi aux sujets d'entraînement et refuse un centre de test, sauf avec `--include-held-out` (inspection seulement).

```powershell
.venv-dl\Scripts\python -m generalization.calibration --source processed   # gamma, contraste, références (~1 min sur C1)
```

Les tests de `tests/test_calibration.py` échouent si un centre de test apparaît dans un réglage ou une configuration, si les configurations ne reprennent pas `calibration.json` (plages, références, source), ou si `splits.json` a changé depuis le calibrage.

### Recalibrage sur C1 du 4 novembre 2026

L'entraînement se fera sur les données C1 : le calibrage est refait sur elles (916 sujets d'entraînement, `R065_sub-r065s019` manquant), avec les mêmes méthodes. Les valeurs bougent peu ; les volumes C1 étant recadrés, les balayages de FDA et du gamma de BigAug sont refaits aussi.

| Paramètre | Données brutes (3 novembre) | **C1 (actuel)** | Fixé par (C1) |
|---|---|---|---|
| `gamma` de `intensity.yaml` | 0,71 – 1,77 | **0,68 – 1,79** | R040 – R030 |
| `factor` de `contrast` | 0,78 – 1,52 | **0,77 – 1,53** | R020 – R030 |
| `reference_sites` (entraînement final) | R059 | **R059** | distance au pool |
| `fold_reference_sites` (plis 0 à 4) | R059, R071, R059, R010, R071 | **R059, R071, R059, R004, R071** | distance au pool du pli |
| `beta` de FDA | 0,015 – 0,03 | **0,015 – 0,03** | balayage C1 |
| `gamma` du preset BigAug | 0,71 – 1,77 | **0,68 – 1,79** | balayage C1 |
| Source des images de référence | brutes | **C1** (`source: processed`) | — |

- **Gamma, 0,68 à 1,79.** Position de la médiane du cerveau sur [0, p99] : 0,41 (R030) à 0,71 (R040), médiane 0,604 ; gamma brut 0,686 à 1,783. R040 remplace R014 comme centre le plus clair : sur C1, R014 vaut 0,698 et R040 0,708 (0,69 sur données brutes).
- **Contraste, 0,77 à 1,53.** Coefficient de variation 0,31 (R020) à 0,62 (R030), médiane 0,406 ; facteur brut 0,772 à 1,527.
- **Pli 3 : R004 au lieu de R010.** Les deux sont proches (0,0154 contre 0,0181) ; sur C1, R004 passe devant.

### Recalibrage du 3 novembre 2026 (données brutes)

Le premier réglage (14 octobre) avait été fait avant `splits.json`, sur tous les centres : il utilisait SOOP et R008, devenus des centres de test, et la borne basse du gamma venait de R032.

| Paramètre | Avant (14 octobre, tous centres) | Après (sujets d'entraînement) | Fixé par |
|---|---|---|---|
| `gamma` de `intensity.yaml` | 0,7 – 1,85 | **0,71 – 1,77** | R014 – R030 |
| `factor` de `contrast` | 0,78 – 1,56 | **0,78 – 1,52** | R020 – R030 |
| `reference_sites` (FDA, histogram matching) | R009 (provisoire) | **R059** ; par pli : R059, R071, R059, R010, R071 | distance au pool |
| `beta` de FDA | 0,015 – 0,03 | **0,015 – 0,03** (inchangé, re-balayé) | balayage |
| `gamma` du preset BigAug | 0,7 – 1,85 | **0,71 – 1,77** | balayage |

Les méthodes sont celles du premier réglage ; seules les données changent :

- **Gamma, 0,71 à 1,77.** La position de la médiane du cerveau sur [0, p99] va de 0,41 (R030) à 0,69 (R014) selon le centre d'entraînement, pour une médiane de 0,601. Le gamma qui amène le centre médian sur chaque centre est `ln(position) / ln(0,601)`, soit 0,719 à 1,765, arrondi vers l'extérieur.
- **Contraste, 0,78 à 1,52.** Le coefficient de variation dans le cerveau va de 0,32 (R020) à 0,62 (R030), pour une médiane de 0,409. Le facteur est le rapport à cette médiane (0,787 à 1,519).
- **Les extrêmes viennent de petits centres** (R020 : 1 sujet d'entraînement ; R014 : 7 ; R030 : 12), comme dans le premier réglage qui gardait tous les centres. C'est voulu : ce sont des styles réels que le modèle doit couvrir.
- **Basse résolution.** Les résolutions natives vont de 0,47 mm à 1,5 mm isotrope (R049), avec des coupes jusqu'à 3 mm sur un seul axe. Sur une grille de 1 mm, seules les résolutions plus grossières que 1 mm sont simulables : 1 à 1,5 mm isotrope, ou 1 à 3 mm sur un axe.
- **Bruit, flou, biais de champ : non calibrés.** `site_summary.csv` ne contient aucune mesure de ces grandeurs. Les valeurs sont des ordres de grandeur usuels. Le biais par défaut donne un champ d'environ 0,82 à 1,26 dans le cerveau (mesuré sur volume synthétique).

### `beta` de FDA : 0,015 à 0,03, re-balayé sur les sujets d'entraînement

Le demi-côté du cube de fréquences échangé vaut `b = floor(beta × plus petit côté)`.

**Balayage sur C1 (4 novembre 2026, plage actuelle).** Mêmes six sujets d'entraînement (R015, R029, R014, R030, R039, R040), 4 images de référence C1 de R059, toutes d'entraînement, 24 paires par valeur :

```powershell
.venv-dl\Scripts\python -m generalization.visual_check --source processed --sites R015 R029 R014 R030 R039 R040 --fda-betas 0.01 0.015 0.02 0.03 0.04 0.05 0.06 0.08 0.12 --fda-references 4 --out generalization\figures\c1_fda
```

| `beta` | `b` | Changement médian (écarts-types du cerveau) | Contraste conservé, minimum |
|---|---|---|---|
| 0,01 | 1 | 0,17 | 0,84 |
| 0,015 | 2 | 0,23 | 0,79 |
| 0,02 | 2 – 3 | 0,24 | 0,79 |
| 0,03 | 4 | 0,28 | 0,68 |
| 0,04 | 5 – 6 | 0,30 | 0,66 |
| 0,05 | 7 | 0,32 | 0,64 |
| 0,06 | 8 – 9 | 0,33 | 0,63 |
| 0,08 | 11 – 12 | 0,36 | 0,62 |
| 0,12 | 17 – 18 | 0,40 | 0,58 |

Les minima viennent tous de R030. Les volumes C1 recadrés ont un plus petit côté plus court (le `b` maximal à 0,03 passe de 5 à 4) ; la plage reste valable : `b` = 2 partout à 0,015, `b` sous le trentième du plus petit côté à 0,03, et le pire cas à 0,03 (0,68) reste au-dessus de celui du flou sur les mêmes sujets (0,59, à R014).

**Balayage sur données brutes (3 novembre 2026).** Mêmes sujets, références brutes de R059 :

```powershell
.venv-dl\Scripts\python -m generalization.visual_check --sites R015 R029 R014 R030 R039 R040 --fda-betas 0.01 0.015 0.02 0.03 0.04 0.05 0.06 0.08 0.12 --fda-references 4 --out generalization\figures\recal_fda
```

| `beta` | `b` | Changement médian de l'image (écarts-types du cerveau) | Contraste conservé, minimum |
|---|---|---|---|
| 0,01 | 1 | 0,10 | 0,89 |
| 0,015 | 2 | 0,18 | 0,76 |
| 0,02 | 2 – 3 | 0,21 | 0,74 |
| 0,03 | 4 – 5 | 0,23 | 0,73 |
| 0,04 | 5 – 7 | 0,28 | 0,66 |
| 0,05 | 7 – 9 | 0,29 | 0,62 |
| 0,06 | 8 – 11 | 0,31 | 0,56 |
| 0,08 | 11 – 15 | 0,33 | 0,55 |
| 0,12 | 17 – 23 | 0,38 | 0,51 |

Les minima viennent de R029 jusqu'à 0,04, puis de R015.

- **Borne basse 0,015 (inchangée).** En dessous, `b` vaut 0 ou 1 et l'image change peu ; quand `b` vaut 0, elle ne change pas du tout. À 0,015, `b` vaut 2 sur tous les volumes.
- **Borne haute 0,03 (inchangée).** Jusque-là, la carte de différence est un ombrage lisse à l'échelle de 3 à 5 cm : c'est bien le « style » du scanner. À partir de 0,05, elle montre des taches de 1 à 2 cm, la taille des lésions. Ce critère d'échelle ne dépend pas des centres : `b` reste sous le trentième du plus petit côté.
- **Ce qui change par rapport au 14 octobre.** Sans R008, aucune valeur ne fait passer le contraste conservé sous la moitié (0,51 au pire, à 0,12) : c'est le critère d'échelle qui fixe seul la borne haute. À 0,03, le pire cas vaut 0,73 au lieu de 0,63.
- **Marge.** À 0,03, le pire cas (0,73) reste au-dessus de celui du flou sur les mêmes sujets (0,55, à R014), un effet déjà accepté.

Limites : un seul centre de référence, six sujets, volumes entiers dont le plus petit côté fait 144 à 192 voxels. Sur des patchs de 128³ ou des volumes recadrés (C1), la même plage donne des `b` plus petits (1 à 3). Le réglage est à refaire si le centre de référence ou la taille des volumes change.

Ancien balayage (14 octobre, avec SOOP et R008 et des références de R009) : même plage retenue, contraste conservé minimal de 0,63 à 0,03 et 0,27 à 0,08 (R008).

### Gamma du preset BigAug : 0,68 à 1,79 au lieu de 0,5 à 4,5

**Sur C1 (4 novembre 2026, plage actuelle)**, mêmes six sujets d'entraînement (`--source processed --gammas …`, sorties dans `generalization\figures\c1_gamma`) : au gamma de 1,79, 1 à 5 % du cerveau passe au noir (R030 : 33 %, 8 % sans transformation) ; à 2,5, 9 à 13 % (R030 : 48 %) ; un gamma de 0,5 garde 0,51 à 0,59 du contraste de la lésion, 0,68 en garde 0,69 à 0,76. Mêmes conclusions que sur les données brutes : le preset reprend la plage calibrée de `intensity.yaml`, soit 0,68 à 1,79.

**Sur données brutes (3 novembre 2026)**, sur un sujet d'entraînement de six centres : R015 (0,5 mm anisotrope, à la place de SOOP), R029, R014, R030 (le centre le plus sombre, à la place de R008), R039 et R040.

```powershell
.venv-dl\Scripts\python -m generalization.visual_check --sites R015 R029 R014 R030 R039 R040 --gammas 0.5 0.6 0.71 0.85 1.0 1.4 1.77 1.85 2.5 3.5 4.5 --out generalization\figures\recal_gamma
```

Part du cerveau écrasée au noir (sous 5 % de l'échelle [minimum, p99] du cerveau) :

| Gamma | 1 | 1,77 | 2,5 | 3,5 | 4,5 |
|---|---|---|---|---|---|
| R015, R029, R014, R039, R040 | 0 à 0,6 % | 1 à 5 % | 10 à 13 % | 18 à 22 % | 26 à 32 % |
| R030 | 8 % | 32 % | 48 % | 61 % | 70 % |

Contraste de la lésion conservé : 0,51 à 0,59 pour un gamma de 0,5, 0,72 à 0,78 pour 0,71 ; au-dessus de 1, la lésion sombre s'assombrit encore et le contraste augmente.

Au-delà de 2,5, la substance grise disparaît. R030 est déjà sombre (8 % au noir sans transformation, surtout la grande lésion et les ventricules) ; à 1,77, sa substance grise reste visible, à 3,5 il ne reste que la substance blanche. Un gamma de 0,5 divise par deux le contraste de toutes les lésions. Le preset reprend donc la plage calibrée de `intensity.yaml`, comme au premier réglage.

Ancien balayage (14 octobre, avec SOOP et R008, centres de test) : 2 à 7 % au noir à 1,85, 26 à 40 % à 4,5, et 82 % à R008.

### Centre de référence

Le centre de référence de FDA et de l'histogram matching est le paramètre `reference_sites` des configurations, jamais une valeur du code. Il vaut **`[R059]`**, choisi par `calibration.py` parmi les centres d'entraînement :

- **Critères** : au moins 20 sujets d'entraînement, 1 mm isotrope (pas de rééchantillonnage) et au moins 200 niveaux de gris dans le cerveau. Ce dernier critère écarte R029 et son histogramme en peigne (73 niveaux) ; le centre suivant en a 242.
- **Choix** : la plus petite distance W1 entre l'histogramme moyen du centre (intensité / p99) et celui du reste du pool d'entraînement. Un centre « typique » déforme le moins les images qu'on harmonise vers lui.
- **Classement sur C1** : R059 (0,009), R071 (0,013), R004 (0,015), R011 (0,016), R010 (0,018). Sur données brutes : R059 (0,009), R071 (0,010), R010 (0,016), R004 (0,018), R011 (0,021), R052 (0,022), R001 (0,032), R009 (0,036) ; R009, la référence provisoire, n'arrivait qu'en 8e position.
- **Seuls les sujets d'entraînement de R059 servent de référence**, jamais ceux du test ID : la clé `splits: splits.json` des entrées `fda` et `histogram_matching` le garantit et refuse un centre de test à la construction. Les images de référence sont lues dans C1 (`source: processed`), avec le même recadrage et la même normalisation que les images entraînées.

#### Référence pli par pli

Chaque centre d'entraînement est un centre de validation dans un des 5 plis ; R059 l'est dans le pli 3. Pour qu'aucun pli ne voie sa validation, la référence est choisie **pli par pli**, avec le même critère, en ne gardant que les sujets d'entraînement du pli : ses centres de validation ne sont ni candidats ni comptés dans le pool qui sert à mesurer la distance.

| Pli (C1) | 0 | 1 | 2 | 3 | 4 | Entraînement final |
|---|---|---|---|---|---|---|
| Référence | R059 | **R071** | R059 | **R004** | **R071** | R059 |
| Distance W1 | 0,010 | 0,005 | 0,009 | 0,015 | 0,007 | 0,009 |
| Deuxième du classement | R071 | R004 | R010 | R010 | R011 | R071 |

R059 est exclu du pli 3, où il sert à la validation. Aux plis 1 et 4, R059 reste autorisé, mais une fois les centres de validation retirés du pool, R071 devient le plus proche. Le choix est refait dans chaque pli au lieu de reprendre le classement global : c'est la même procédure que celle qu'on évalue, appliquée sans voir la validation. Sur données brutes, le pli 3 donnait R010 ; sur C1, R004 passe devant de peu.

```python
transform = build_transforms("generalization/configs/fda.yaml", fold=3)   # référence R004, sujets d'entraînement C1 du pli 3
final = build_transforms("generalization/configs/fda.yaml")               # référence R059, tous les sujets d'entraînement
```

Les références par pli sont dans la clé `fold_reference_sites` des entrées `fda` et `histogram_matching`, et dans la section `fold_references` de `calibration.json`. Avec `fold=k`, `build_transforms` refuse un centre de validation du pli k ou un centre de test, et ne prend comme images de référence que des sujets d'entraînement du pli k. Le cache `reference_file` de l'histogram matching reçoit le suffixe `_pliK`. **Cette variabilité entre plis est voulue.** Une référence différente selon le pli fait un peu varier les résultats d'un pli à l'autre, mais chaque pli applique la procédure complète sans voir sa validation : l'écart entre plis mesure aussi la sensibilité au choix de la référence, ce qu'une référence fixe cacherait. Ne pas la remplacer par une référence unique pour « stabiliser » les plis.

## Vérification

### Tests

`.venv-dl\Scripts\python -m pytest generalization\tests -q` : 187 tests, environ 30 secondes, sans donnée patient (3 tests de plus se lancent si les données réelles, brutes et C1, sont présentes). Ils couvrent :

- **Invariants de chaque transformation d'intensité**, sur un volume isotrope, un volume anisotrope 0,5 × 0,47 × 0,47 mm et un volume de type R029 : masque et cerveau identiques, forme et affine conservées, pas de NaN ni d'infini, fond intact, entrée non modifiée.
- **Reproductibilité** avec une graine fixée, et identité avec `p = 0`.
- **Comportement propre** : gamma monotone avec des valeurs négatives, FDA à `beta = 0` égal à l'original, phase conservée par FDA, histogrammes rapprochés par l'histogram matching.
- **Chargement** : LAS et PSR réorientés en RAS, 1 mm, volume du cerveau conservé, grille gardée pour un volume déjà à 1 mm, masque 0,99999998, fond de R029, masque vide signalé, exclusion de R032, z-score.
- **Configurations et adaptateur MONAI** : chaque YAML se construit et s'applique, les transformations sont sérialisables pour les `DataLoader`.
- **Chargeur C1** (`test_data_processed.py`), sur un jeu prétraité synthétique de moins de 64³ écrit dans un dossier temporaire : clés et z-score, chargement paresseux et sérialisable, masque 0,99999998 seuillé, cerveau lu dans `brain.nii.gz`, masque vide signalé, découpages et plis, R032 exclu, erreurs claires (orientation, espacement, grille, sujet manquant, table absente), calibration et `visual_check` avec `--source processed`, et un test de bout en bout `load_processed` → `build_transforms(monai=True)` dans un `Compose` MONAI, avec références FDA et histogram matching prises dans C1.
- **Aucun centre de test dans les réglages** (`test_calibration.py`) : centres de référence des YAML, centres utilisés par `calibration.json` et ses balayages, empreinte de `splits.json`, accord entre les YAML et `calibration.json`, refus d'un centre OOD par `check_training_sites`, `build_transforms` et `visual_check`, références prises hors du test ID, et référence de chaque pli hors de ses centres de validation.

### Vérification visuelle

```powershell
.venv-dl\Scripts\python -m generalization.visual_check --synthetic
.venv-dl\Scripts\python -m generalization.visual_check --source processed --sites R015 R029 R014 R030 R039 R040 --out generalization\figures\c1_familles
```

Sur données réelles, seuls les sujets d'entraînement de `splits.json` sont candidats (le premier du centre qui a une lésion) ; un centre de test est refusé, sauf avec `--include-held-out`.

Le script écrit une figure par famille et par sujet : coupe axiale où la lésion est la plus grande, contour de la lésion, original puis 4 tirages. Il écrit aussi `contraste_lesion.csv` (une ligne par tirage), `resume.csv` et `conserve_min_par_centre.csv`. Tout va dans `generalization\figures\`, ignoré par Git parce que les figures réelles montrent des coupes de patients.

Le **rapport L/V** est la médiane de l'intensité dans la lésion divisée par celle d'une couronne de 3 voxels autour, comme `lesion_vs_shell` de l'exploration. Le **contraste conservé** vaut `(1 − L/V après) / (1 − L/V avant)` : 1 s'il est inchangé, 0 si la lésion devient invisible, négatif si elle devient plus claire que son voisinage.

#### Résultat sur C1 (4 novembre 2026, réglages actuels)

Mêmes sujets d'entraînement, lus dans C1, 10 tirages par transformation, références C1 de R059. Contraste conservé **minimal** :

| Transformation | R015 | R029 | R014 | R030 | R039 | R040 |
|---|---|---|---|---|---|---|
| *L/V d'origine* | *0,57* | *0,84* | *0,93* | *0,50* | *0,91* | *0,73* |
| `gamma` (0,68 – 1,79) | 0,79 | 0,76 | 0,75 | 0,81 | 0,75 | 0,77 |
| `contrast` (0,77 – 1,53) | 0,82 | 0,85 | 0,86 | 0,83 | 0,87 | 0,85 |
| `noise` | 0,98 | 0,85 | 0,89 | 0,99 | 0,99 | 0,99 |
| `blur` | 0,87 | 0,74 | 0,59 | 0,79 | 0,72 | 0,64 |
| `bias_field` | 0,94 | 0,77 | 0,76 | 0,97 | 1,00 | 0,99 |
| `low_resolution` | 0,88 | 0,72 | 0,50 | 0,82 | 0,71 | 0,65 |
| `randconv` | 0,85 | 0,74 | 0,68 | 0,82 | 0,76 | 0,68 |
| `gin` | 0,79 | 0,70 | 0,75 | 0,75 | 0,87 | 0,76 |
| `fda` (`beta` 0,015 – 0,03) | 0,86 | 0,88 | 0,84 | 0,69 | 1,00 | 0,92 |
| `histogram_matching` | 0,91 | 1,20 | 1,12 | 0,86 | 0,96 | 1,23 |
| preset `bigaug` (gamma 0,68 – 1,79) | 0,72 | 0,75 | 0,49 | 0,71 | 0,65 | 0,54 |

Aucune transformation n'inverse la lésion ; aucune transformation d'intensité n'a modifié un masque (606 tirages ; les 22 masques modifiés viennent de la transformation affine de BigAug). Les cas les plus durs sont à R014, dont la lésion fait moins de 0,1 mL et se voit à peine au départ (L/V 0,93) : basse résolution à 0,50, deux tirages BigAug à 0,49 et 0,50, tous deux avec transformation affine. Sur la figure BigAug de R014, la rotation change le plan de coupe affiché et la lésion reste à sa place. Les sujets C1 ont un L/V un peu différent des sujets bruts (R014 0,93 au lieu de 0,91, R040 0,73 au lieu de 0,69) : rééchantillonnage et recadrage du pipeline.

#### Résultat sur les sujets d'entraînement (3 novembre 2026, données brutes)

Un sujet d'entraînement par centre (le premier qui a une lésion), 10 tirages par transformation, référence R059. Contraste conservé **minimal** :

| Transformation | R015 | R029 | R014 | R030 | R039 | R040 |
|---|---|---|---|---|---|---|
| *L/V d'origine* | *0,57* | *0,84* | *0,91* | *0,50* | *0,90* | *0,69* |
| `gamma` (0,71 – 1,77) | 0,82 | 0,79 | 0,78 | 0,83 | 0,78 | 0,80 |
| `contrast` (0,78 – 1,52) | 0,83 | 0,86 | 0,87 | 0,84 | 0,87 | 0,86 |
| `noise` | 0,98 | 0,83 | 0,95 | 0,99 | 0,99 | 0,97 |
| `blur` | 0,87 | 0,74 | 0,55 | 0,79 | 0,74 | 0,59 |
| `bias_field` | 0,96 | 0,78 | 0,85 | 0,97 | 1,00 | 0,97 |
| `low_resolution` | 0,88 | 0,74 | 0,58 | 0,80 | 0,72 | 0,63 |
| `randconv` | 0,86 | 0,74 | 0,77 | 0,82 | 0,71 | 0,66 |
| `gin` | 0,78 | 0,70 | 0,74 | 0,75 | 0,89 | 0,73 |
| `fda` (`beta` 0,015 – 0,03) | 0,71 | 0,72 | 0,88 | 0,79 | 1,01 | 0,91 |
| `histogram_matching` | 0,91 | 1,19 | 1,10 | 0,84 | 0,95 | 1,18 |
| preset `bigaug` (gamma 0,71 – 1,77) | 0,77 | 0,64 | 0,55 | 0,65 | 0,68 | 0,49 |

Aucune transformation d'intensité n'efface ni n'inverse la lésion : le minimum est 0,55, atteint par le flou à R014, sur une lésion de moins de 0,1 mL, comme au premier réglage. Sur les 606 tirages, aucune transformation d'intensité n'a modifié un masque ; les 24 masques modifiés viennent de la transformation affine de BigAug, qui déplace aussi le masque. Le seul tirage sous 0,5 est un tirage BigAug à R040 (0,49), qui combine transformation affine et flou sur une petite lésion : elle reste visible sur la figure. Lecture des figures regardées :

- **Style (R029)** : anatomie et lésion intactes ; un tirage RandConv laisse un liseré clair au bord du cerveau (effet de bord des convolutions), comme au premier réglage.
- **Harmonisation (R030 vers R059)** : l'histogramme après traitement se superpose à la référence et la lésion reste visible. L'histogramme de R059 est discret (environ 300 niveaux de gris dans le cerveau), d'où des dents de scie sur la courbe ; l'image n'en montre aucune trace.
- **BigAug (R040)** : anatomie lisible sur les 4 tirages affichés, lésion en place.
- **Gamma et FDA** : voir les balayages ci-dessus.

Trois limites :

- **Un seul sujet par centre.** Ces minima ne sont pas des statistiques de centre.
- **Figures regardées une par une** pour les cas extrêmes seulement (R030, R029, R040, R015) ; les autres figures sont dans `generalization\figures\recal_familles\`.
- **Le rapport L/V suppose un zéro d'intensité fixe.** Il n'a plus de sens quand une transformation décale les intensités (luminosité de BigAug) ou écrase l'image : c'est ce qui donnait des valeurs aberrantes à R008 avant le réglage. Pour BigAug, les figures priment.

### Temps CPU

`.venv-dl\Scripts\python -m generalization.benchmark`, sur un volume synthétique de 176 × 208 × 176 voxels (6,4 millions), 4 threads, médiane de 3 essais :

| Transformation | Temps (s) | Transformation | Temps (s) |
|---|---|---|---|
| `gamma` | 0,16 | `sharpen` | 0,22 |
| `contrast` | 0,11 | `brightness` | 0,12 |
| `noise` | 0,15 | `randconv` | 0,95 |
| `blur` | 0,19 | `gin` | 1,37 |
| `bias_field` | 0,21 | `fda` | 0,60 |
| `low_resolution` | 0,31 | `histogram_matching` | 0,37 |

Sur les volumes réels (7 à 12,5 millions de voxels), les temps médians mesurés par `visual_check` sont du même ordre : 0,1 à 0,35 s pour l'intensité, 0,7 à 0,9 s pour FDA et RandConv, 1,5 s pour GIN.

En moyenne par volume synthétique, avec leurs probabilités : 0,26 s pour `intensity.yaml` et 0,52 s pour `bigaug.yaml`. Un patch de 128³ contient trois fois moins de voxels (2,1 millions) : les temps y seront plus courts, mais ce n'est pas mesuré.

**Sur un vrai volume C1** (`--source processed --data-root data\processed`, 4 novembre 2026, médiane de 3 essais), `R001_sub-r001s001` recadré à 144 × 191 × 160 voxels (4,4 millions) : 0,07 à 0,16 s pour l'intensité, 0,36 s pour l'histogram matching, 0,40 s pour FDA, 0,62 s pour RandConv et 1,11 s pour GIN ; 0,19 s en moyenne pour `intensity.yaml` et 0,50 s pour `bigaug.yaml`.

Avec `--source processed` sans données (4 novembre 2026, un essai), le même cerveau synthétique écrit au format C1 et recadré fait 168 × 198 × 168 voxels (5,6 millions) : 0,05 à 0,18 s pour l'intensité, 0,33 s pour FDA, 0,40 s pour l'histogram matching, 0,60 s pour RandConv et 0,95 s pour GIN ; 0,13 s en moyenne pour `intensity.yaml` et 0,45 s pour `bigaug.yaml`. Ces temps ne sont pas directement comparables au tableau ci-dessus (autre jour, un seul essai, volume plus petit).

## Choix faits

- **TorchIO comme format natif**, avec un adaptateur MONAI. Les transformations sont écrites pour ce projet plutôt que reprises de TorchIO, parce qu'elles doivent agir dans le cerveau seulement et sur des images en z-score.
- **Fond à 0 après le z-score** (convention nnU-Net). Le fond se confond alors avec la moyenne du cerveau : c'est pour cela que le sujet porte un masque `brain`. C'est aussi la convention du contrat C1 (week4.py), qui fournit `brain.nii.gz`.
- **Gamma et biais sur une échelle positive.** L'image est ramenée entre son minimum et son 99e percentile dans le cerveau avant le gamma ; le biais multiplie l'intensité mesurée depuis le minimum du cerveau.
- **Filtres spatiaux avec un fond sombre.** Avant un flou ou une basse résolution, le fond est mis au niveau le plus sombre du cerveau, pour que les bords se comportent comme sur une image brute.
- **RandConv et GIN : deux écarts aux articles**, pour garder la lésion visible :
  - le mélange avec l'original est plafonné à 0,5 au lieu de 1. Avec le plafond des articles, 9 % (RandConv) et 12 % (GIN) des tirages synthétiques perdaient plus de la moitié du contraste ;
  - une image stylisée anti-corrélée à l'original est retournée (`allow_inversion: false`), donc le contraste ne s'inverse pas.
  - De plus, le résultat retrouve la moyenne et l'écart-type du cerveau d'origine, au lieu de la normalisation de Frobenius de GIN.
- **FDA à `beta = 0` est l'identité stricte** : aucun coefficient n'est échangé, pas même la composante continue.
- **Histogram matching par quantiles.** La référence est la moyenne des fonctions quantiles des sujets de référence. À R029 (une trentaine de niveaux de gris), les voxels de même valeur restent égaux : l'histogramme ne se rapproche qu'en partie de la référence.
- **Preset BigAug adouci** : bruit, échelle, déformation et gamma sont plus faibles que dans l'article, pour ne pas détruire l'anatomie ni les petites lésions (voir le calibrage). Les valeurs de l'article ont été reportées de mémoire et sont à vérifier pendant l'état de l'art.
- **Bruit, flou et biais** gardent leurs valeurs de départ : les figures réelles ne montrent aucun défaut, mais rien ne permet de les calibrer.
- **Lecture par nibabel** plutôt que par le lecteur de TorchIO, pour lire les en-têtes comme le reste du projet.
- **Grille de rééchantillonnage calculée ici** plutôt que par `tio.Resample(1.0)`, qui arrondit la taille vers le bas : sur un volume à 0,99999997 mm, il retirait une rangée de voxels par axe, soit jusqu'à 3 % du cerveau sur les volumes recadrés serré.

## Reste à faire

Vérifié le 14 octobre 2026 sur les données réelles :

- `list_subjects()` rend 1 416 sujets et 54 centres, R032 exclu ;
- les 1 416 sujets se chargent sans erreur ni avertissement. Les seuls masques vides sont les 5 connus, et les 48 lésions de moins de 0,1 mL de l'exploration sont retrouvées. Aucune lésion n'est perdue au rééchantillonnage : 914 volumes déjà à 1 mm gardent leur grille, 502 sont rééchantillonnés. Pour ceux-ci, le volume du cerveau change de 1,5 % au plus et celui des lésions de plus de 1 mL de 0,7 % en médiane ; les très petites lésions sont les plus touchées (0,071 → 0,053 mL pour `sub-r050s012`) ;
- les six centres ci-dessus ont été regardés, et le `beta` de FDA réglé.

Fait le 3 novembre 2026 : recalibrage sur les seuls sujets d'entraînement de `splits.json`. Fait le 4 novembre 2026 : données C1 produites sur ce poste, `load_processed` vérifié sur tous les découpages, calibrage et balayages refaits sur C1, vérification visuelle et temps CPU sur C1 (voir [Calibrage](#calibrage)).

Reste :

1. **Étendre la vérification visuelle** à plusieurs sujets par centre et aux centres d'entraînement non regardés, avant les premiers entraînements.
2. **Refaire le calibrage si `splits.json` ou C1 change** : `test_calibration.py` le signale par l'empreinte du fichier. Relancer `python -m generalization.calibration --source processed`, puis les deux balayages, et reporter les plages dans les YAML et dans les sections `fda_beta` et `bigaug_gamma` de `calibration.json`.
3. **`R065_sub-r065s019`** : en échec côté C1 (`lesion_outside_image_based_crop`), écarté par `on_missing="warn"`. Si Yosra le corrige, relancer `preprocess_c1_local.py` (reprise automatique des autres sujets).
4. **Chargement des données brutes lent** (repli seulement) : `load_subject` prend 1,2 s en médiane, mais 6,7 s pour SOOP et jusqu'à 28 s (volumes SOOP natifs à 0,25 mm). Les données C1, déjà à 1 mm et recadrées, règlent ce point pour l'entraînement.
5. **Vérifier les valeurs de BigAug** dans l'article.

## Suite du bloc

Ne sont pas dans ce livrable : la normalisation de type Fan-Net, le tableau d'ablation RQ3 et l'adaptation continue contrôlée (§5.1). DANN et TENT sont dans `adaptation/` (voir son README) ; leur branchement dans la boucle d'entraînement dépend du contrat C4.
