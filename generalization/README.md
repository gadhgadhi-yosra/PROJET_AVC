# Généralisation inter-centres — bibliothèque de transformations (contrat C5)

Techniques de généralisation de la question RQ3 (§3.5 du document de projet) : augmentation d'intensité, randomisation du style, méthode de Fourier et harmonisation. Tout fonctionne sur les **données brutes** ATLAS v3, en attendant les données prétraitées du contrat C1.

État au 13 octobre 2026 : le code et les 132 tests tournent sur des volumes synthétiques. **Rien n'a encore été exécuté sur les vraies IRM**, qui n'étaient pas disponibles ; la liste de ce qui reste à lancer est [en fin de document](#à-lancer-une-fois-les-données-extraites).

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

`load_subject` applique dans l'ordre :

1. **Vérification de la grille** : erreur si l'image et le masque n'ont pas la même grille.
2. **Masque du cerveau** : voxels strictement au-dessus de la valeur du fond (la valeur la plus fréquente), et non `> 0`. À R029 le fond vaut environ 0,0007 et il existe des valeurs négatives.
3. **Masque de lésion** : `mask > 0.5`, jamais `== 1` (certains masques valent 0,99999998).
4. **Réorientation RAS** puis **rééchantillonnage à 1 mm** : interpolation linéaire pour l'image, plus proche voisin pour les masques.
5. **Z-score dans le cerveau**, fond mis à 0. Les valeurs `norm_mean` et `norm_std` sont gardées dans le sujet pour revenir aux intensités brutes.

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
| `fda` | Échange l'amplitude basse fréquence avec une image d'un autre centre, phase conservée | `beta: [0, 0.03]` | Yang & Soatto, FDA, CVPR 2020 |
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

Le centre de référence de FDA et de l'histogram matching est le paramètre `reference_sites` des configurations, jamais une valeur du code. Il vaut `[R009]` à titre provisoire : il devra être pris parmi les centres d'entraînement.

## Vérification

### Tests

`.venv-dl\Scripts\python -m pytest generalization\tests -q` : 132 tests, environ 20 secondes, sans donnée patient. Ils couvrent :

- **Invariants de chaque transformation d'intensité**, sur un volume isotrope, un volume anisotrope 0,5 × 0,47 × 0,47 mm et un volume de type R029 : masque et cerveau identiques, forme et affine conservées, pas de NaN ni d'infini, fond intact, entrée non modifiée.
- **Reproductibilité** avec une graine fixée, et identité avec `p = 0`.
- **Comportement propre** : gamma monotone avec des valeurs négatives, FDA à `beta = 0` égal à l'original, phase conservée par FDA, histogrammes rapprochés par l'histogram matching.
- **Chargement** : LAS et PSR réorientés en RAS, 1 mm, masque 0,99999998, fond de R029, masque vide signalé, exclusion de R032, z-score.
- **Configurations et adaptateur MONAI** : chaque YAML se construit et s'applique, les transformations sont sérialisables pour les `DataLoader`.

### Vérification visuelle

```powershell
.venv-dl\Scripts\python -m generalization.visual_check --synthetic
.venv-dl\Scripts\python -m generalization.visual_check --sites SOOP R029 R009 R014
```

Le script écrit une figure par famille et par sujet dans `generalization\figures\synthetic\` ou `generalization\figures\raw\` : coupe axiale où la lésion est la plus grande, contour de la lésion, original puis 4 tirages. Il écrit aussi `contraste_lesion.csv` (une ligne par tirage) et `resume.csv`. Ce dossier est ignoré par Git, car les figures réelles montrent des coupes de patients.

Le **rapport L/V** est la médiane de l'intensité dans la lésion divisée par celle d'une couronne de 3 voxels autour, comme `lesion_vs_shell` de l'exploration. Le **contraste conservé** vaut `(1 − L/V après) / (1 − L/V avant)` : 1 s'il est inchangé, 0 si la lésion devient invisible.

Résultat sur 4 sujets **synthétiques** (lésions construites à L/V 0,67 à 0,90), 10 tirages par transformation et par sujet :

| Transformation | L/V après | Contraste conservé, minimum | médiane | Tirages sous 50 % |
|---|---|---|---|---|
| `gamma` | 0,50 – 0,92 | 0,78 | 1,16 | 0 |
| `contrast` | 0,55 – 0,91 | 0,86 | 1,11 | 0 |
| `noise` | 0,67 – 0,91 | 0,91 | 1,00 | 0 |
| `blur` | 0,68 – 0,93 | 0,63 | 0,84 | 0 |
| `bias_field` | 0,67 – 0,90 | 0,94 | 1,00 | 0 |
| `low_resolution` | 0,68 – 0,94 | 0,60 | 0,81 | 0 |
| `randconv` | 0,67 – 0,95 | 0,49 | 0,94 | 1 sur 40 |
| `gin` | 0,66 – 0,92 | 0,78 | 0,95 | 0 |
| `fda` | 0,64 – 0,90 | 0,98 | 1,00 | 0 |
| `histogram_matching` | 0,67 – 0,82 | 1,00 | 1,01 | 0 |
| preset `bigaug` | 0,34 – 0,94 | 0,56 | 1,16 | 0 |

Aucun tirage n'inverse le contraste. Trois lectures :

- **Flou et basse résolution** réduisent le contraste des petites lésions, ce qui est leur effet physique attendu.
- **RandConv** peut tirer un noyau passe-haut qui retire le contraste lisse de la lésion. Avec le mélange plafonné à 0,5, le pire tirage en garde la moitié (0,90 → 0,95 sur le sujet de type R014).
- **FDA** change très peu les volumes synthétiques, parce que les centres synthétiques ont le même contenu basse fréquence. Son effet réel ne se jugera que sur les vraies données.

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

En moyenne par volume, avec leurs probabilités : 0,26 s pour `intensity.yaml` et 0,52 s pour `bigaug.yaml`. Un patch de 128³ contient trois fois moins de voxels (2,1 millions) : les temps y seront plus courts, mais ce n'est pas mesuré.

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
- **Preset BigAug adouci** : bruit, échelle et déformation sont plus faibles que dans l'article, pour ne pas détruire les petites lésions. Le gamma garde la plage de l'article (0,5 à 4,5), très forte : le tirage le plus sombre écrase la substance grise. Les valeurs de l'article ont été reportées de mémoire et sont à vérifier pendant l'état de l'art.
- **`beta` de FDA, bruit, flou et biais** sont des valeurs de départ, à ajuster sur les figures des vraies données.

## À lancer une fois les données extraites

1. Vérifier le chargement : `list_subjects()` doit rendre 1 416 sujets et 54 centres (1 453 moins les 37 de R032).
2. Lancer `visual_check` sur SOOP, R029, R009 et R014, puis sur R008 et R039 (`--sites R008 R039`). Regarder les figures et la colonne « conservé min » : R014, R008 et R039 ont un L/V d'environ 0,91.
3. Ajuster sur ces figures le `beta` de FDA, le plafond `mix` de RandConv et GIN, le bruit, le flou et le biais.
4. Charger les 5 sujets à masque vide et vérifier qu'aucun autre masque vide n'est signalé. Vérifier aussi qu'aucune petite lésion n'est perdue au rééchantillonnage (un avertissement le signale).
5. Mesurer le temps réel de `load_subject` sur un volume SOOP natif (0,5 × 0,47 × 0,47 mm).
6. Dès que `splits.json` existe : recalibrer les plages sur les centres d'entraînement et choisir `reference_sites` parmi eux.

## Suite du bloc

Ne sont pas dans ce livrable : la normalisation de type Fan-Net, DANN (inversion de gradient et classifieur de centre), TENT (adaptation au test), le tableau d'ablation RQ3 et l'adaptation continue contrôlée (§5.1). DANN et TENT se branchent dans le modèle et la boucle d'entraînement : ils dépendent du contrat C4.
