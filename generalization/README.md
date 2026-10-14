# Généralisation inter-centres — bibliothèque de transformations (contrat C5)

Techniques de généralisation de la question RQ3 (§3.5 du document de projet) : augmentation d'intensité, randomisation du style, méthode de Fourier et harmonisation. Tout fonctionne sur les **données brutes** ATLAS v3, en attendant les données prétraitées du contrat C1.

État au 14 octobre 2026 :

- les 138 tests passent sur des volumes synthétiques ;
- le chargement a été vérifié sur les 1 416 sujets réels ;
- les transformations ont été regardées sur un sujet de six centres (SOOP, R029, R009, R014, R008, R039) ;
- deux défauts du chargement ont été corrigés (lecture de deux masques de R040, rééchantillonnage des volumes déjà à 1 mm) ;
- deux réglages ont changé : le `beta` de FDA et le gamma du preset BigAug.

Ce qui reste à faire est [en fin de document](#reste-à-faire).

| Fichier | Rôle |
|---|---|
| `data_raw.py` | Liste et charge les sujets bruts en `torchio.Subject` (RAS, 1 mm, z-score dans le cerveau) |
| `transforms/` | Les transformations et `build_transforms(config)` |
| `configs/*.yaml` | Une configuration par famille, plus `baseline.yaml` (aucune transformation) |
| `contrast.py` | Rapport d'intensité lésion / tissu voisin |
| `synthetic.py` | Volumes synthétiques qui imitent les pièges des données réelles |
| `visual_check.py` | Figures avant / après et mesure du contraste de la lésion |
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
from generalization.data_raw import list_subjects, load_subject
from generalization.transforms import build_transforms

records = list_subjects("data/ATLAS3_Training_Raw")   # table pandas, centre R032 exclu
subject = load_subject(records.iloc[0])                # clés : t1, mask, brain, site, subject…
transform = build_transforms("generalization/configs/intensity.yaml")
augmented = transform(subject)                         # le sujet d'entrée n'est pas modifié
```

Une configuration est une liste de transformations, chacune avec un nom, une probabilité `p` et ses plages `[min, max]` :

```yaml
transforms:
  - name: gamma
    p: 0.3
    gamma: [0.7, 1.85]
  - name: noise
    enabled: false        # désactivée sans supprimer ses paramètres
    p: 0.2
    std: [0.0, 0.1]
```

Pour le tableau d'ablation, plusieurs configurations se combinent : `build_transforms(["…/intensity.yaml", "…/fda.yaml"])`.

**Boucle MONAI (contrat C4).** `build_transforms(config, monai=True)` renvoie un objet qui prend et rend un dictionnaire `{"image", "label"}` en canal d'abord, utilisable dans un `monai.transforms.Compose`. Trois points à connaître :

- **Masque du cerveau.** Une clé `"brain"` est utilisée si elle existe. Sinon le cerveau est estimé par les voxels différents de la valeur du fond.
- **Graine.** L'aléa vient du générateur de PyTorch : `torch.manual_seed` ou `monai.utils.set_determinism` suffisent.
- **Images de référence.** FDA et l'histogram matching lisent leurs images de référence dans `data_root` (argument de `build_transforms`, sinon clé `data_root` de la configuration).

**Passage aux données prétraitées (C1).** Les transformations ne dépendent que du `torchio.Subject` (clés `t1`, `mask`, `brain`). Il suffira d'écrire un chargeur équivalent à `load_subject` pour `data/processed/` ; le reste ne change pas.

## Ce que fait le chargement provisoire

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
| `gamma` | Gamma sur le cerveau ramené à [0, 1], puis retour à l'échelle d'origine | `gamma: [0.7, 1.85]` | Isensee et al., nnU-Net, *Nature Methods* 2021 |
| `contrast` | Étire les intensités autour de la moyenne du cerveau | `factor: [0.78, 1.56]` | idem |
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

Les plages viennent de `exploration/outputs/site_summary.csv`.

- **Gamma, 0,7 à 1,85.** La position de la médiane du cerveau sur [0, p99] va de 0,40 (R030) à 0,70 (R032) selon le centre, pour une médiane de 0,60. Le gamma qui amène le centre médian sur chaque centre est `ln(position) / ln(0,60)`, soit 0,70 à 1,84.
- **Contraste, 0,78 à 1,56.** Le coefficient de variation dans le cerveau va de 0,32 à 0,64, pour une médiane de 0,41. Le facteur est le rapport à cette médiane.
- **Basse résolution.** Les résolutions natives vont de 0,47 mm à 1,5 mm isotrope (R049), avec des coupes jusqu'à 3 mm sur un seul axe. Sur une grille de 1 mm, seules les résolutions plus grossières que 1 mm sont simulables : 1 à 1,5 mm isotrope, ou 1 à 3 mm sur un axe.
- **Bruit, flou, biais de champ : non calibrés.** `site_summary.csv` ne contient aucune mesure de ces grandeurs. Les valeurs sont des ordres de grandeur usuels. Le biais par défaut donne un champ d'environ 0,82 à 1,26 dans le cerveau (mesuré sur volume synthétique).

> **À recalibrer.** Ces plages sont calculées sur tous les centres, y compris les futurs centres de test. Dès que `splits.json` est fixé, il faut les recalculer sur les seuls centres d'entraînement, sinon le test OOD n'est plus honnête. La borne basse du gamma vient d'ailleurs de R032, un centre aujourd'hui exclu ; sans lui elle vaut 0,74 (R040).

### `beta` de FDA : 0,015 à 0,03, réglé sur les données réelles

Le demi-côté du cube de fréquences échangé vaut `b = floor(beta × plus petit côté)`. Mesure du 14 octobre 2026 sur 6 sujets (SOOP, R029, R014, R008, R039, R040) et 4 images de référence de R009, soit 24 paires par valeur :

| `beta` | `b` | Changement médian de l'image (écarts-types du cerveau) | Contraste conservé, minimum |
|---|---|---|---|
| 0,01 | 1 | 0,04 | 0,97 |
| 0,015 | 2 | 0,13 | 0,77 |
| 0,02 | 2 – 3 | 0,16 | 0,69 |
| 0,03 | 4 – 5 | 0,19 | 0,63 |
| 0,04 | 5 – 7 | 0,23 | 0,54 |
| 0,05 | 7 – 9 | 0,26 | 0,63 |
| 0,06 | 8 – 11 | 0,28 | 0,42 |
| 0,08 | 11 – 15 | 0,32 | 0,27 |
| 0,12 | 17 – 23 | 0,37 | 0,35 |

- **Borne basse 0,015.** En dessous, `b` vaut 0 ou 1 : l'image ne change presque pas (0,04 écart-type), et quand `b` vaut 0 elle ne change pas du tout. Avec l'ancienne plage de 0 à 0,03, 35 à 46 % des tirages tombaient dans ce cas, selon la taille du volume.
- **Borne haute 0,03.** Jusque-là, la carte de différence est un ombrage lisse à l'échelle de plusieurs centimètres : c'est bien le « style » du scanner. À partir de 0,05, elle montre des taches de 1 à 2 cm, la taille des lésions. Au-delà, le contraste conservé passe sous la moitié : 0,42 pour `beta` = 0,06 et 0,27 pour 0,08, les deux à R008.
- **Marge.** À 0,03, le pire cas (0,63, à R008) reste au-dessus de celui du flou (0,55, à R014), un effet déjà accepté.

Limites : un seul centre de référence (R009), six sujets, volumes entiers dont le plus petit côté fait 144 à 192 voxels. Sur des patchs de 128³, la même plage donne `b` de 1 à 3. Le réglage est à refaire si le centre de référence change :

```powershell
.venv-dl\Scripts\python -m generalization.visual_check --sites SOOP R029 R014 R008 R039 R040 --fda-betas 0.01 0.015 0.02 0.03 0.04 0.05 0.06 0.08 0.12 --fda-references 4 --out generalization\figures\raw_fda
```

### Gamma du preset BigAug : 0,7 à 1,85 au lieu de 0,5 à 4,5

Part du cerveau écrasée au noir (sous 5 % de l'échelle d'intensité), mesurée sur un sujet de six centres :

| Gamma | 1,85 | 2,5 | 3,5 | 4,5 |
|---|---|---|---|---|
| SOOP, R029, R009, R014, R039 | 2 à 7 % | 10 à 18 % | 18 à 31 % | 26 à 40 % |
| R008 | 8 % | 22 % | 55 % | 82 % |

Au-delà de 2,5, la substance grise disparaît ; à R008, tout le cerveau devient noir et la lésion avec lui. À l'autre bout, un gamma de 0,5 divise par deux le contraste des lésions peu visibles (R008 : L/V 0,94 → 0,97). Le preset reprend donc la plage calibrée de `intensity.yaml`.

### Centre de référence

Le centre de référence de FDA et de l'histogram matching est le paramètre `reference_sites` des configurations, jamais une valeur du code. Il vaut `[R009]` à titre provisoire : il devra être pris parmi les centres d'entraînement.

## Vérification

### Tests

`.venv-dl\Scripts\python -m pytest generalization\tests -q` : 138 tests, environ 20 secondes, sans donnée patient (2 tests de plus se lancent si les données réelles sont présentes). Ils couvrent :

- **Invariants de chaque transformation d'intensité**, sur un volume isotrope, un volume anisotrope 0,5 × 0,47 × 0,47 mm et un volume de type R029 : masque et cerveau identiques, forme et affine conservées, pas de NaN ni d'infini, fond intact, entrée non modifiée.
- **Reproductibilité** avec une graine fixée, et identité avec `p = 0`.
- **Comportement propre** : gamma monotone avec des valeurs négatives, FDA à `beta = 0` égal à l'original, phase conservée par FDA, histogrammes rapprochés par l'histogram matching.
- **Chargement** : LAS et PSR réorientés en RAS, 1 mm, volume du cerveau conservé, grille gardée pour un volume déjà à 1 mm, masque 0,99999998, fond de R029, masque vide signalé, exclusion de R032, z-score.
- **Configurations et adaptateur MONAI** : chaque YAML se construit et s'applique, les transformations sont sérialisables pour les `DataLoader`.

### Vérification visuelle

```powershell
.venv-dl\Scripts\python -m generalization.visual_check --synthetic
.venv-dl\Scripts\python -m generalization.visual_check --sites SOOP R029 R009 R014
.venv-dl\Scripts\python -m generalization.visual_check --sites R008 R039 --out generalization\figures\raw_r008_r039
```

Le script écrit une figure par famille et par sujet : coupe axiale où la lésion est la plus grande, contour de la lésion, original puis 4 tirages. Il écrit aussi `contraste_lesion.csv` (une ligne par tirage), `resume.csv` et `conserve_min_par_centre.csv`. Tout va dans `generalization\figures\`, ignoré par Git parce que les figures réelles montrent des coupes de patients.

Le **rapport L/V** est la médiane de l'intensité dans la lésion divisée par celle d'une couronne de 3 voxels autour, comme `lesion_vs_shell` de l'exploration. Le **contraste conservé** vaut `(1 − L/V après) / (1 − L/V avant)` : 1 s'il est inchangé, 0 si la lésion devient invisible, négatif si elle devient plus claire que son voisinage.

#### Résultat sur les données réelles (14 octobre 2026)

Un sujet par centre (le premier qui a une lésion), 10 tirages par transformation. Contraste conservé **minimal** :

| Transformation | SOOP | R029 | R009 | R014 | R008 | R039 |
|---|---|---|---|---|---|---|
| *L/V d'origine* | *0,82* | *0,84* | *0,79* | *0,91* | *0,94* | *0,90* |
| `gamma` | 0,78 | 0,78 | 0,78 | 0,77 | 0,77 | 0,77 |
| `contrast` | 0,85 | 0,86 | 0,86 | 0,87 | 0,84 | 0,87 |
| `noise` | 0,99 | 0,83 | 1,00 | 0,95 | 0,96 | 0,99 |
| `blur` | 0,92 | 0,74 | 0,81 | 0,55 | 0,87 | 0,74 |
| `bias_field` | 0,91 | 0,78 | 0,91 | 0,85 | 0,94 | 1,00 |
| `low_resolution` | 0,90 | 0,74 | 0,84 | 0,58 | 0,86 | 0,72 |
| `randconv` | 0,76 | 0,74 | 0,82 | 0,77 | 0,74 | 0,71 |
| `gin` | 0,78 | 0,70 | 0,69 | 0,74 | 0,74 | 0,89 |
| `fda` (`beta` 0,015 – 0,03) | 0,97 | 0,72 | 0,93 | 0,81 | 0,75 | 1,10 |
| `histogram_matching` | 0,77 | 1,07 | 0,98 | 1,09 | 1,20 | 0,89 |
| preset `bigaug` (gamma 0,7 – 1,85) | 0,77 | 0,64 | 0,71 | 0,55 | 0,67 | 0,68 |

Avec les réglages actuels, aucune transformation n'efface ni n'inverse la lésion : le minimum est 0,55, atteint par le flou et par le preset BigAug à R014, sur une lésion de moins de 0,1 mL. Sur les 606 tirages, aucune transformation d'intensité n'a modifié un masque. Lecture des figures, famille par famille :

- **Intensité** : anatomie et lésion intactes sur les six centres. Le flou et la basse résolution atténuent les petites lésions, ce qui est leur effet physique.
- **Style** : anatomie et lésion intactes. Certains tirages laissent un liseré clair au bord du cerveau (effet de bord des convolutions).
- **Fourier** : l'ombrage change à grande échelle, l'anatomie et la lésion restent en place.
- **Harmonisation** : l'histogramme après traitement se superpose à la référence. À R008, dont l'histogramme est très différent, le contraste de l'image change nettement. À R029, l'histogramme reste en peigne.
- **BigAug** : l'anatomie reste lisible après réduction du gamma. Avec le gamma de l'article, les tirages à fort gamma écrasaient la substance grise, et à R008 le cerveau devenait presque entièrement noir.

Trois limites :

- **Un seul sujet par centre.** Ces minima ne sont pas des statistiques de centre.
- **Colonne R009.** R009 est le centre de référence de FDA et de l'histogram matching : cette colonne mesure un échange à l'intérieur du même centre.
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

## Choix faits

- **TorchIO comme format natif**, avec un adaptateur MONAI. Les transformations sont écrites pour ce projet plutôt que reprises de TorchIO, parce qu'elles doivent agir dans le cerveau seulement et sur des images en z-score.
- **Fond à 0 après le z-score** (convention nnU-Net). Le fond se confond alors avec la moyenne du cerveau : c'est pour cela que le sujet porte un masque `brain`. À aligner avec le contrat C1 quand la convention de Yosra sera connue.
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

Reste :

1. **Étendre la vérification visuelle** à plusieurs sujets par centre et aux centres non regardés, avant les premiers entraînements.
2. **Recalibrer après `splits.json`** : plages de gamma et de contraste, `reference_sites` et `beta` de FDA, sur les seuls centres d'entraînement.
3. **Accélérer le chargement.** `load_subject` prend 1,2 s en médiane, mais 6,7 s pour SOOP et jusqu'à 28 s : certains volumes SOOP natifs sont à 0,25 mm (354 millions de voxels pour `sub-soop0430`). Trois chargements en parallèle ont une fois épuisé les 16 Go de mémoire de la machine. Le prétraitement C1 réglera ce point ; d'ici là, un cache local des volumes à 1 mm est à prévoir pour l'entraînement.
4. **Vérifier les valeurs de BigAug** dans l'article.

## Suite du bloc

Ne sont pas dans ce livrable : la normalisation de type Fan-Net, DANN (inversion de gradient et classifieur de centre), TENT (adaptation au test), le tableau d'ablation RQ3 et l'adaptation continue contrôlée (§5.1). DANN et TENT se branchent dans le modèle et la boucle d'entraînement : ils dépendent du contrat C4.
