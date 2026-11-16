# RQ3 — Adaptation de domaine pour la segmentation des lésions d'AVC (brouillon)

Question : les techniques de généralisation et d'adaptation de domaine réduisent-elles l'écart de
performance entre les centres vus à l'entraînement (ID) et des centres jamais vus (OOD) ?

Sources des chiffres : `adaptation/outputs/rq3/` (tableau, par centre, par taille, entraînement),
`adaptation/outputs/rq3/echecs/` (analyse des échecs), `adaptation/outputs/rq3/adaptation_continue/`
(adaptation continue). Scripts : `adaptation/aggregate_rq3.py`, `adaptation/analyse_echecs.py`.

## 1. Protocole

**Données.** Images T1 d'ATLAS (déjà débarrassées du crâne) après le prétraitement S4 de l'équipe :
orientation RAS, rééchantillonnage à 1 mm, recadrage sur le cerveau, z-score dans le cerveau
(`preprocessing/week4.py`). Le découpage est fixé dans `splits.json` ; le centre R032 est exclu.

**Découpage par centre.** Six centres sont réservés au test OOD (R008, R028, R031, R034, R053, SOOP). Ils
ont été choisis sur la taille des centres et la distance des histogrammes d'intensité, sans regarder le Dice
(`splits.json`, champ `selection`). Les autres centres sont répartis en 5 plis. Seul le pli 0 a été entraîné :

| Ensemble | Sujets | Centres |
|---|---|---|
| Entraînement | 733 | 39 |
| Validation | 184 | 9 centres non vus à l'entraînement |
| Test ID | 185 | centres d'entraînement, sujets non vus |
| Test OOD | 268 (265 à lésion non vide) | 6 |

**Modèle.**
- U-Net 3D de MONAI : canaux 32 à 320, 4 sous-échantillonnages, 2 unités résiduelles par niveau.
- Normalisation d'instance affine, nécessaire à TENT.
- Patches de 96³, lots de 4 patches (2 volumes × 2 patches), dont la moitié est centrée sur la lésion.
- Miroir gauche-droite aléatoire pour toutes les runs.
- Perte Dice + entropie croisée (`DiceCELoss`).
- AdamW : taux d'apprentissage 1e-3 à décroissance polynomiale (puissance 0,9), décroissance des poids 1e-5.
- Écrêtage de la norme des gradients à 12.
- 40 000 itérations et graine 2026 pour toutes les techniques ; GPU T4 (Kaggle), fp32 (voir section 6).

**Sélection.** Le modèle est validé toutes les 2 000 itérations sur les 184 sujets de validation, en volume
entier (fenêtre glissante 96³, recouvrement 0,25). Le checkpoint retenu (`best.pt`) est celui du meilleur
Dice de validation. Il est ensuite évalué une seule fois sur les tests ID et OOD.

**Métriques.**
- Dice, HD95, écart absolu de volume, F1 et différence du nombre de lésions (`baselines/common.py`, communes à l'équipe).
- Le Dice n'est calculé que sur les sujets à lésion non vide.

**Tests statistiques.**
- Moyennes avec IC à 95 % par bootstrap percentile (10 000 tirages).
- Écart ID − OOD : les deux groupes sont rééchantillonnés indépendamment.
- Comparaison à la baseline : test des rangs signés de Wilcoxon, apparié sur les mêmes sujets.
- Correction de Holm sur l'ensemble des techniques comparées.
- [À COMPLÉTER : BigAug] Les p corrigés changeront avec une cinquième comparaison.

## 2. Techniques évaluées

Toutes partagent le réseau, les plis, les graines et le budget d'itérations. Seule l'étape propre à chaque
technique change (`adaptation/train.py`, `generalization/configs/`).

- **Augmentation d'intensité** (`intensity.yaml`) :
  - gamma (p = 0,3) et contraste (p = 0,3), calibrés sur les seuls sujets d'entraînement ;
  - bruit (p = 0,2) et flou (p = 0,2) ;
  - champ de biais (p = 0,3) ;
  - basse résolution (p = 0,25, jusqu'à 3 mm sur un axe).
- **BigAug** (`bigaug.yaml`), d'après Zhang et al. (2020) :
  - flou, netteté, bruit, luminosité, gamma (plage calibrée) et transformation affine (échelle 0,8–1,2, rotation 20°) ;
  - chaque transformation avec p = 0,5 ;
  - déformation élastique désactivée ;
  - plages adoucies par rapport à l'article pour préserver les petites lésions.
- **FDA** (`fda.yaml`), d'après Yang et Soatto (2020) :
  - l'amplitude basse fréquence est échangée avec une image d'un centre de référence d'entraînement (R059 pour le pli 0) ;
  - β entre 0,015 et 0,03, p = 0,5.
- **DANN** :
  - classifieur des 39 centres d'entraînement sur la sortie du goulot (moyenne spatiale, LayerNorm sans paramètres, MLP de 64 unités) ;
  - couche d'inversion du gradient ;
  - λ nul sur les 2 000 premières itérations, puis montée de Ganin (γ = 5) ;
  - poids de la perte du centre : 0,1.
- **TENT** :
  - adaptation au test, sans étiquettes, du modèle baseline : seuls les paramètres affines des normalisations sont modifiés ;
  - minimisation de l'entropie, par centre, avec Adam (taux 1e-4) et 4 patches par sujet.
- **Histogram matching** : non lancé, faute de quota GPU. La configuration existe
  (`histogram_matching.yaml`), mais aucun résultat n'est rapporté.

## 3. Résultats RQ3

Tableau (`adaptation/outputs/rq3/tableau_rq3.md`) :

| Technique | Dice ID [IC 95 %] | Dice OOD [IC 95 %] | Écart ID − OOD | Δ OOD vs baseline [IC 95 %] | p Wilcoxon OOD (Holm) |
|---|---|---|---|---|---|
| Baseline | 0,646 [0,610 ; 0,680] | 0,553 [0,517 ; 0,589] | 0,093 [0,044 ; 0,143] | – | – |
| Intensité | 0,656 [0,622 ; 0,689] | 0,577 [0,542 ; 0,610] | 0,080 [0,031 ; 0,127] | +0,024 [0,008 ; 0,040] | 0,0014 (0,0041) |
| BigAug | [À COMPLÉTER : BigAug] | | | | |
| FDA | 0,660 [0,627 ; 0,693] | 0,559 [0,524 ; 0,593] | 0,101 [0,052 ; 0,149] | +0,006 [−0,008 ; 0,020] | 0,32 (0,32) |
| DANN | 0,650 [0,616 ; 0,683] | 0,549 [0,514 ; 0,584] | 0,102 [0,054 ; 0,150] | −0,004 [−0,018 ; 0,009] | 0,13 (0,26) |
| Baseline + TENT | 0,645 [0,610 ; 0,679] | 0,542 [0,507 ; 0,577] | 0,104 [0,054 ; 0,152] | −0,011 [−0,016 ; −0,006] | < 1e-4 (< 1e-4) |

n = 185 sujets ID et 265 sujets OOD pour chaque technique.

Figures à insérer :
- `figures/rq3/rq3_dice_id_ood.png` : Dice ID et OOD par technique, IC 95 %.
- `figures/rq3/rq3_delta_ood.png` : différences appariées à la baseline sur le test OOD.
- `figures/rq3/rq3_dice_par_centre.png` : Dice moyen par centre et par technique.

**Lecture.**
- **L'écart demeure pour toutes les techniques.** Il va de 0,080 à 0,104, et aucun IC ne contient 0.
- **L'intensité est la seule technique qui améliore le test OOD** de façon significative après correction :
  +0,024 sur les 265 sujets appariés. Son écart (0,080) est le plus faible, mais son IC chevauche celui de la baseline.
- **FDA** améliore le test ID (Wilcoxon ID p = 0,012, Holm 0,036 ; Δ ID +0,014 [−0,001 ; 0,029]), pas le test OOD.
- **DANN** ne se distingue pas de la baseline, ni en ID (p = 0,45) ni en OOD (p = 0,13).
- **TENT dégrade légèrement le test OOD** (−0,011) et le test ID (−0,0007, p Holm 0,0007). L'effet est petit mais systématique.
- **Les métriques secondaires ne suivent pas toujours le Dice** (`tableau_rq3.csv`). Le F1 des lésions OOD
  baisse avec l'intensité (0,450 contre 0,477) ainsi qu'avec FDA et DANN (0,366 et 0,367). L'écart absolu
  de volume baisse avec l'intensité (10,0 ml contre 11,1 ml).
- **La validation ne départage pas les techniques.** Les meilleurs Dice de validation vont de 0,430 à 0,437
  (`adaptation/outputs/rq3/entrainement.csv`).
- **Pourquoi la validation (≈ 0,43) est plus basse que le test ID (0,646) :** elle porte sur 9 centres non vus,
  et ses lésions sont petites (médiane 1,26 ml, 43 % sous 1 ml, contre 7,56 ml et 18 % au test ID ; `lesion_ml`
  des `qc.json`). Le Dice baisse nettement avec le volume (section 4).

## 4. Analyse des échecs

Fichiers : `adaptation/outputs/rq3/echecs/echecs.md` (tableaux) et `spearman.csv`.

**Par centre OOD** (`par_centre_ood.csv`) :

| Centre | Sujets (lésion) | Vol. médian (ml) | Jours post-AVC (médiane) | Distance W1 | Contraste T1 | Dice baseline | Dice intensité |
|---|---|---|---|---|---|---|---|
| R008 | 7 (7) | 0,95 | 1 | 0,216 | −0,17 | 0,00 | 0,00 |
| R031 | 37 (37) | 6,74 | 138 | 0,164 | −0,98 | 0,62 | 0,67 |
| R028 | 26 (26) | 4,94 | 576 | 0,157 | −0,82 | 0,66 | 0,65 |
| SOOP | 169 (167) | 22,75 | non renseigné | 0,023 | −0,69 | 0,54 | 0,57 |
| R053 | 5 (4) | 0,81 | 25 | 0,020 | −0,50 | 0,20 | 0,20 |
| R034 | 24 (24) | 2,38 | 1 506 | 0,014 | −0,64 | 0,63 | 0,65 |

Le contraste T1 est la différence entre le z-score moyen de la lésion et celui d'un anneau de 1 à 4 mm autour.
Une lésion chronique est hypo-intense (contraste négatif). Figure : `figures/rq3/rq3_echecs_distance_dice.png`.

**Taille des lésions.**
- C'est le facteur le plus net. Sur les 265 sujets OOD, ρ de Spearman entre volume et Dice (modèle intensité)
  = 0,53, p < 1e-4.
- Dice OOD de la baseline : 0,31 sous 1 ml (n = 41), 0,46 de 1 à 10 ml, 0,64 de 10 à 50 ml, 0,74 au-delà de 50 ml.
- Les techniques gardent ce profil ; l'intensité gagne surtout entre 1 et 50 ml (0,50 et 0,67).
- Figures : `figures/rq3/rq3_dice_par_taille.png` (ID et OOD), `figures/rq3/rq3_echecs_dice_par_taille.png` (OOD).

**R008 (Dice 0 pour toutes les techniques).** Fichier `zoom_R008_R053.csv`.
- **Phase aiguë et lésions petites :** 6 sujets sur 7 sont à J1 ; les lésions font 0,1 à 4,1 ml.
- **Lésions peu visibles en T1 :** contraste médian −0,17, contre −0,50 à −0,98 dans les autres centres OOD.
- **Intensités anormales :** 1,1 à 1,7 % des voxels du cerveau dépassent z = 3, contre 0,003 % en médiane
  dans les autres sujets OOD (`contraste_lesions.csv`).
- **Prédictions :** le modèle prédit 0,4 à 33,6 ml répartis en de nombreuses composantes (différence du nombre
  de lésions de 9 à 67).
- **Coupe :** vaisseaux, sinus et dure-mère apparaissent hyperintenses. Figure `figures/rq3/rq3_echecs_coupes_R008.png`,
  à régénérer en local : elle n'est pas versionnée car elle montre des images ATLAS.
- *Hypothèse* : séquence T1 acquise après injection de gadolinium, en phase aiguë. La lésion n'y a pas encore le
  contraste T1 des lésions chroniques qui dominent l'entraînement. Le décalage est donc à la fois d'acquisition
  et de stade clinique.
- *À vérifier* : le protocole d'acquisition de R008, et l'écart entre la date de l'IRM et celle de l'annotation.

**R053 (Dice 0,20).**
- Un sujet sur 5 a un masque vide ; il est exclu du Dice.
- Deux lésions minuscules (0,13 et 0,015 ml) ont un Dice de 0. Les deux autres (1,5 et 2,6 ml) obtiennent 0,27 et 0,52.
- Les intensités sont dans la norme (z-score au 99e centile ≈ 1,8) et le contraste vaut −0,50.
- L'échec s'explique par la taille des lésions, pas par un décalage d'intensité. Avec 4 sujets, la moyenne du centre est très instable.

**Distance de domaine (C6).** La distance de Wasserstein entre histogrammes d'intensité (`analysis/center_distance.py`,
référence = tous les autres centres) ne prédit pas le Dice dans nos runs :
- 6 centres OOD : ρ de −0,43 à −0,20 selon la technique, p de 0,40 à 0,70. Avec la MMD : ρ de −0,37 à −0,14.
- 44 centres de test (ID + OOD) : ρ de 0,05 à 0,12, p ≥ 0,44.
- R053 est parmi les centres les plus proches (0,020) et parmi les pires ; R031 est loin (0,164) et parmi les meilleurs.
- Six centres donnent très peu de puissance : on ne peut pas conclure à l'absence de lien, seulement qu'il n'apparaît pas ici.
- Au niveau des sujets, le contraste de la lésion est faiblement lié au Dice (ρ = −0,19, p = 0,001) :
  `figures/rq3/rq3_echecs_contraste_dice.png`.

## 5. Adaptation continue contrôlée

**Protocole.** Script `adaptation/continual.py`, sorties dans `adaptation/outputs/rq3/adaptation_continue/`.
- Le modèle baseline (`best.pt`) reçoit les 6 centres OOD l'un après l'autre, par lots de 4 sujets : 28 lots,
  108 sujets, au plus 24 par centre. Le modèle n'est jamais réinitialisé.
- Évaluation préquentielle : chaque lot est d'abord noté (Dice), puis utilisé pour l'adaptation, sans étiquettes.
- Quatre stratégies, sur les mêmes lots :
  - `fige` : modèle inchangé ;
  - `auto_apprentissage` : pseudo-étiquettes, tous les poids ;
  - `tent` : TENT continu ;
  - `controlee` : TENT dans un challenger, promu seulement si (1) la distance C6 du lot dépasse un seuil calibré
    sur la source (0,077), (2) l'entropie baisse sur des patches neufs et (3) le Dice sur un jeu de garde étiqueté
    de la source ne baisse pas de plus de 0,01. Elle utilise aussi un replay des centres déjà vus et des patches source.
- Oubli : Dice sur 16 sujets fixes du test ID et sur les premiers sujets des centres OOD déjà vus, après chaque centre.

**Résultats** (`resume.json`, `oubli.csv`, `decisions.csv`) :

| Stratégie | Dice préquentiel | Dice ID départ → fin | Oubli ID | Oubli OOD moyen |
|---|---|---|---|---|
| Figé | 0,528 | 0,635 → 0,635 | 0,000 | 0,000 |
| Auto-apprentissage | 0,486 | 0,635 → 0,583 | 0,052 | 0,033 |
| TENT continu | 0,526 | 0,635 → 0,632 | 0,002 | 0,000 |
| Contrôlée | 0,527 | 0,635 → 0,634 | 0,001 | 0,000 |

**Lecture.**
- **Détection :** un décalage est détecté sur 15 lots sur 28 : tous ceux de R008, R028 et R031, et 1 lot sur 6 de SOOP.
  Il n'est jamais détecté sur R034 ni R053, les centres les plus proches de la source.
- **Promotions :** 12, réparties en 2 pour R008, 6 pour R028, 3 pour R031 et 1 pour SOOP. Trois challengers de R031
  ont été refusés parce que leur entropie montait.
- **Oubli :** la stratégie contrôlée évite l'oubli de l'auto-apprentissage (0,001 contre 0,052 sur le test ID).
- **Gain :** elle n'apporte pas de gain mesurable par rapport au modèle figé (0,527 contre 0,528). R008 reste à 0.
- Les Dice préquentiels portent sur un sous-ensemble des sujets OOD (108 sur 268). Ils ne sont pas directement
  comparables au tableau de la section 3.

## 6. Limites

- **Un seul pli** (pli 0) et une seule graine. Les IC reflètent la variabilité des sujets, pas celle de l'entraînement.
- **DANN reprise après coupure :**
  - La run a été coupée à l'itération 35 889 (Colab) et reprise sur Kaggle jusqu'à 40 000.
  - La reprise restaure les poids, l'optimiseur, le taux d'apprentissage, λ, la position de l'échantillonneur et
    les états aléatoires des workers. Le contrôle automatique de la reprise est passé (première ligne : itération
    35 900, taux 1,29e-04, λ 0,9771).
  - Les masques de Dropout du classifieur de centre ne sont pas restaurés.
  - Le meilleur modèle (itération 24 000) précède de toute façon la coupure.
- **Choix de `dann_weight` = 0,1 :**
  - Avec un poids de 1, la première run DANN a divergé (perte totale de 10 508 à l'itération 100 à 33 M à l'itération 400).
  - Le poids 0,1 et une LayerNorm avant le classifieur ont été choisis pour la stabilité, sans recherche d'hyperparamètres.
  - Un poids plus fort pourrait donner un autre résultat.
- **Validation sur des centres non vus**, aux lésions plus petites que le test. Elle sert à choisir le checkpoint,
  mais son Dice (≈ 0,43) n'est pas une estimation de la performance ID.
- **Diagnostic fp16 :**
  - La première baseline, en fp16, a produit des NaN (`rq3-diag-nan`) : des patchs entiers donnaient des logits
    NaN dès l'itération 160 à 180 environ, sans NaN en fp32 sur les mêmes lots.
  - Toutes les runs ont été refaites en fp32, environ deux fois plus lent.
  - La couche qui déborde n'a pas été localisée.
- **Bug de reprise corrigé :**
  - Dans le code des runs (`7449c49`), la ligne de progression écrasait la file des états aléatoires sauvegardés
    pour la reprise. Le bug est corrigé dans `c29aa22` et couvert par un test.
  - Il ne touche que les reprises, donc uniquement DANN. Le `last.pt` de DANN a été réparé avant la reprise en ne
    gardant que les 4 derniers états, qui sont les bons ; les poids sont inchangés.
- **Quota GPU** (30 h par semaine sur Kaggle) : histogram matching n'a pas été lancé et aucun autre pli n'a été entraîné.
- **TENT et l'adaptation continue** sont évalués sur la seule baseline.

## 7. Conclusion RQ3

Sur le pli 0 :
- Seule l'augmentation d'intensité améliore le Dice OOD de façon significative : +0,024 [0,008 ; 0,040],
  p Holm = 0,0041.
- Aucune technique ne supprime l'écart ID − OOD : 0,080 à 0,104 selon la technique.
- FDA, DANN et TENT n'améliorent pas le test OOD ; TENT le dégrade légèrement.
- L'adaptation continue contrôlée évite l'oubli mais n'améliore pas le Dice.
- Les échecs les plus marqués dépendent davantage des lésions que de la distance d'intensité entre centres :
  les petites lésions, et celles de R008, peu contrastées en T1 (stade aigu, injection supposée).
- [À COMPLÉTER : BigAug] Comparer BigAug à l'intensité avant de choisir le modèle final.
- Le modèle livré à la démo (contrat C8, `adaptation/inference.py`) est aujourd'hui celui de l'intensité.
