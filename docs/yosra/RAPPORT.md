# Sections de rapport — contribution de Yosra

Brouillon scientifique fondé sur les résultats locaux disponibles. Les valeurs finales
doivent être prises dans les JSON/CSV générés après le dernier QC ; les expériences
sans prédictions ou checkpoints ne sont pas présentées comme réalisées.

## Introduction et questions de recherche

Ce projet étudie la segmentation des lésions d’AVC sur des IRM T1 acquises dans plusieurs
centres. Les différences de scanner, protocole et résolution peuvent modifier la distribution
des images. Notre objectif est de mesurer ce décalage, d’évaluer la généralisation des
modèles sur des centres absents de l’entraînement, puis d’étudier la possibilité d’anticiper
et de réduire leur perte de performance.

Trois questions structurent le projet : quel est l’écart entre test ID et test OOD ?
Les architectures Transformer améliorent-elles la généralisation par rapport à nnU-Net ?
Quelles techniques de généralisation ou d’adaptation réduisent cet écart ? La contribution
de Yosra fournit le pipeline de données, les partitions communes, les mesures du décalage,
les analyses explicatives et l’interface de démonstration. Les comparaisons de segmentation
et l’adaptation continue sont produites par Fayfa et Anas.

## Données et contrôle qualité

L’inventaire local contient 1 453 paires IRM/masque dans 55 centres. Le centre R032,
soit 37 cas, est exclu provisoirement pour espacement incohérent. Les métadonnées vides
sont signalées sans supprimer une paire utilisable. Les masques vides sont conservés
comme cas négatifs. Le délai post-AVC reste numérique ; `CHRONICITY`, codé 1/vide,
n’est pas interprété comme une catégorie aigu/subaigu/chronique.

La livraison v2 contient 1 415 volumes prétraités, relus et exportés. Elle exclut les
37 cas R032 et met un cas supplémentaire en quarantaine technique. Le split v1 est
archivé ; v2 retire uniquement ce cas, sans déplacer les autres sujets ni les centres.
La relecture collective de cette quarantaine reste nécessaire ; les annotations brutes sont intactes.

## Pipeline Spark

Chaque sujet constitue une unité de travail. Le pipeline réoriente les images en RAS,
rééchantillonne à 1 mm isotrope (linéaire pour l’image, plus proche voisin pour les masques),
normalise les intensités dans le support cérébral et recadre avec une marge de cinq voxels.
Le support est défini par les intensités supérieures à la valeur de fond estimée ; cette
précaution traite notamment le centre R029 dont le fond n’est pas nul. Cette heuristique
reste une limite et doit être examinée sur les cas atypiques.

Le z-score est stocké en `int16` avec un pas de 0,001. L’erreur d’arrondi est au plus
0,0005 unité de z-score. Le validateur utilise cette borne, avec une petite marge numérique,
pour la moyenne et l’écart-type ; le fond doit rester exactement nul. Une signature des
entrées et paramètres permet la reprise et chaque paire est accompagnée d’un QC.

La boîte de recadrage est déterminée par l’image seule. Le code existant tolère la suppression
de labels hors boîte jusqu’à 1 % du volume lésionnel. Ce choix n’est pas une correction
clinique des annotations : il doit être documenté et validé. Le dernier cas dépasse cette
tolérance (1,76 %) et est mis en quarantaine dans v2. L’option N4 a été ajoutée pour une expérience distincte ;
elle n’a pas été appliquée au jeu de référence et aucun bénéfice n’en est revendiqué.

## Partitions et absence de fuite

Le fichier commun `splits.json` version 2 fixe 916 cas d’entraînement/validation, 231 cas ID et
268 cas OOD. Six centres OOD sont intégralement retirés : R008, R028, R031, R034, R053
et SOOP. Les cinq plis séparent les centres d’entraînement et de validation. Le test ID
de chaque pli est limité aux centres présents dans son entraînement.

La sélection des centres OOD varie taille et distance d’intensité. Comme la distance
participe à cette sélection, un écart de distance ID/OOD n’est pas une découverte indépendante.
Les mesures prédictives doivent utiliser la référence d’entraînement du modèle considéré,
pas le pool global. Aucun hyperparamètre d’alerte ne doit être choisi sur le test OOD.

## Benchmark

Le pilote existant porte sur huit petits volumes et un essai par configuration. Les temps
de calcul sont 56,59 s, 41,46 s, 27,63 s et 42,54 s pour 1, 2, 4 et 8 workers. L’accélération
maximale mesurée est ×2,05 avec quatre workers. La dégradation à huit workers est compatible
avec des coûts de concurrence et de ressources ; ce pilote ne permet pas d’en isoler la cause.
Les workers ne constituent pas des cœurs physiques réservés. Le démarrage Spark n’entre
pas dans `compute_seconds`. Les courbes ne sont pas extrapolées au dataset entier.

## Statistiques et empreinte du centre

Les rapports S5 comparent les distributions avant/après prétraitement et mesurent les
distances Wasserstein et MMD entre centres et référence. Le z-score contraint moyenne et
variance par scan ; les différences restantes portent donc surtout sur la forme des distributions.
Les échelles avant/après ne sont pas identiques : leur comparaison normalisée reste descriptive.

Le test S6 prédit le centre depuis des histogrammes ou statistiques, avec cinq plis
stratifiés, exactitude équilibrée et contrôle à labels mélangés. Comparer des histogrammes
de granularité proche et les mêmes patients permet de mieux isoler l’effet du prétraitement.
Une empreinte reconnaissable n’implique pas à elle seule une baisse du Dice.

Valeurs et figures : `analysis/reports/post_stats_summary.json`,
`analysis/reports/site_fingerprint_s6.csv`, `figures/s5_*.png`, `figures/s6_empreinte_centre.png`.
Les versions historiques sur 1 402 cas ne doivent pas être confondues avec la relance sur 1 415.

La relance du 11 novembre couvre 1 415 sujets dans 54 centres. La distance relative médiane
passe de 0,1315 avant à 0,0376 après, sur la même cohorte disponible. La corrélation entre
Wasserstein et MMD vaut 0,842 pour les centres d’au moins cinq scans. La médiane de
Wasserstein vaut 0,0376 pour ID et 0,0900 pour OOD ; cet écart reste affecté par la sélection
initiale des centres. Le test S6 retient 1 140 scans de 25 centres. Avec les histogrammes
32 classes avant / 30 classes après et le meilleur des deux classifieurs, l’exactitude
équilibrée vaut 69,7 % puis 48,9 %. Le contrôle mélangé vaut 4,0 %, proche du hasard à 4 %.
Ces chiffres ne mesurent pas la performance de segmentation.

## Prédiction d’échec et effet du temps — protocole C1 préparé

S7 met en relation la distance au train du pli et l’erreur `1 − Dice` par centre. Une validation
leave-one-center-out produit des prédictions hors calibration. La régression et le seuil
d’alerte sont ajustés uniquement sur les autres centres de validation. La différence de
Dice par rapport au test ID est rapportée séparément. Sans les résultats C3 de C1, ni corrélation
empirique ni performance d’alerte ne peuvent être annoncées.

S8 compare les associations du centre et du délai post-AVC par modèles linéaires emboîtés
sur les mêmes cas complets. Le délai est transformé par `log(1+jours)`. Le test de permutation
préserve le centre. Les délais manquants, la sélection des cas et les confusions possibles
limitent l’interprétation. Le protocole n’établit pas une relation causale.

## Caractéristiques réseau et DANN — protocole préparé

La version C6 v2 accepte des représentations gelées, obtenues par pooling spatial d’une
couche documentée. La référence seule définit la standardisation et la largeur du noyau MMD.
La comparaison d’empreinte baseline/augmentation/DANN utilise les mêmes patients hors
entraînement du réseau et les mêmes plis du classifieur. Les checkpoints doivent partager
un protocole compatible. Ces expériences n’ont pas de résultat tant que les modèles ne
sont pas fournis.

## Démonstration et limites

L’application Gradio affiche les coupes d’une IRM et un masque, calcule le volume lésionnel
en mL et accepte une carte d’incertitude. Un adaptateur permet de brancher le modèle final.
NiiVue fournit une vue 3D locale sur la route `/viewer/` du même serveur. ANTsPy peut recaler un template MNI et transférer
un atlas de labels ; une inspection visuelle demeure nécessaire avant de nommer les régions.

L’alerte individuelle d’intensité utilise un quantile de validation distinct du seuil par
centre. Elle n’est ni une garantie de performance ni une décision clinique. En l’absence
d’une entrée requise, l’application indique ce qui manque au lieu de créer un résultat.
Le support animé de secours est synthétique et explicitement étiqueté.

## Discussion et travaux à terminer

Les principales limites actuelles sont la quarantaine QC à relire collectivement, le faible nombre de centres
pour calibrer un prédicteur d’échec, la dépendance des distances aux effectifs, les métadonnées
manquantes et l’absence locale des checkpoints et des Dice correspondant à C1 v2. Les résultats de RQ1–RQ3, la vidéo
de vraie inférence et la relecture collective restent nécessaires
avant un rapport final. Une analyse S8 historique distincte est disponible plus bas. Les références bibliographiques du document de projet doivent être
vérifiées dans leurs publications originales avant d’être reprises dans la version remise.

## Compléments vérifiés de la livraison v2

L’export nnU-Net contient 916 images et labels d’entraînement, 499 images de test
(231 ID, 268 OOD) et cinq plis. Le contrôle officiel nnU-Net reste à lancer dans
l’environnement d’entraînement de Fayfa. Les autres appartenances de v1 sont conservées.

L’analyse descriptive S8 trouve 1 073 délais utilisables et 342 manquants ou invalides ;
le délai médian est de 365 jours. Huit centres n’ont aucun délai utilisable. L’association
centre / log-délai (eta carré 0,7475) souligne la nécessité de tenir compte du centre ;
elle ne mesure pas l’effet du délai sur le Dice. Voir `analysis/reports/s8_metadata/`.

Le seuil individuel d’atypie du pli 0 vaut 0,114885 : référence de 733 scans du train,
quantile 95 % sur 183 scans de validation. Aucun test ID/OOD n’a servi à ce réglage.

Chrome a chargé l’application, produit le rapport synthétique de 2,091 mL, puis affiché
les coupes et le rendu 3D NiiVue sans erreur JavaScript. L’image Docker a été construite
et ses routes ainsi que le calcul du rapport ont été testés. La vidéo de secours MP4
utilise ces captures réelles d’interface sur un phantom explicitement synthétique ;
elle ne montre pas une inférence du réseau.

L’atlas Harvard-Oxford et le template compatible MNI152NLin6Asym ont été récupérés
avec leurs sources et empreintes dans `data/demo/atlas/provenance.json`. Les labels
corticaux sont prioritaires, puis les labels sous-corticaux décalés de 100 ; les zones
non couvertes gardent la valeur zéro. Le recalage natif exige une relecture anatomique.
Sources : [Nilearn Harvard-Oxford](https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_atlas_harvard_oxford.html)
et [TemplateFlow MNI152NLin6Asym](https://github.com/templateflow/tpl-MNI152NLin6Asym).

Le recalage SyN sur l’exemple réel a été exécuté. Atlas et IRM ont la même grille,
les labels restent entiers et 97,75 % des voxels atlas non nuls se trouvent dans le
support d’intensité de l’image. La planche locale `data/demo/registration/qc_overlay.png`
a été examinée dans trois plans ; cela ne remplace pas une relecture anatomique experte.
Le rapport régional utilise le masque de référence fourni, sans inférence de réseau.

## S8 — résultat exploratoire sur les baselines récupérées

Le 12 novembre, les scores historiques de `cnn_pretrained_freeze-none` ont été
rattachés à leur propre partition (903 train, 226 ID validation, 287 OOD), sans
les reclasser dans C1 v2. L’analyse utilise 167 cas ID complets de 31 centres et
257 cas OOD de 12 centres. En ID, 58 délais manquent ou sont invalides et un Dice
est absent ; en OOD, 30 délais manquent ou sont invalides. Le Dice absent du cas
à masque vide n’est pas remplacé par un score supposé.

Après ajustement sur le centre, l’ajout de log(1 + délai) augmente le R² de 0,000046
en ID (p exploratoire = 0,911) et de 0,03808 en OOD (p = 0,001 ; 999 permutations
intra-centre pour chaque groupe). Après ajout de log(1 + volume lésionnel), ces
gains sont respectivement 0,000711 et 0,000217. L’association OOD dépend donc
fortement des variables prises en compte ; elle ne permet pas d’attribuer un effet
causal au temps. Le volume est une analyse de sensibilité, pas un ajustement causal validé.

Ces deux tests sont exploratoires, sans correction de multiplicité. L’ID a participé
au choix historique du modèle et n’est pas un test indépendant. La partition et le
prétraitement diffèrent de C1 v2. Les résultats des autres baselines sont résumés
séparément sur leurs effectifs disponibles, sans comparaison appariée implicite.

Livrables reproductibles : `analysis/historical_baselines.py`,
`analysis/reports/s8_historical/summary.json`, `README.md`, `model_summary.csv`,
`missing_by_center.csv` et `time_dice.png`. Les empreintes des fichiers sources sont conservées.

## S6 — complément sur l’incertitude (12 novembre)

Une évaluation complémentaire utilise 912 sujets d’entraînement et 228 sujets réservés,
stratifiés parmi les mêmes 25 centres. Deux classifieurs fixés, régression logistique
et forêt aléatoire, sont évalués sur les mêmes sujets. Les sujets réservés sont ensuite
rééchantillonnés 10 000 fois à l’intérieur de chaque centre, avec les mêmes indices
pour toutes les représentations ; l’exactitude équilibrée donne le même poids aux centres.

Pour la forêt aléatoire, le score avant traitement (32 classes) est de 70,1 %, avec un
intervalle percentile conditionnel à 95 % de [64,6 ; 75,5] %. Après traitement (30 classes),
il est de 51,1 % [45,3 ; 57,0] %. La différence appariée après moins avant est de
−19,0 points [−25,5 ; −12,6]. Pour la régression logistique, elle est de −21,5 points
[−27,4 ; −15,5]. La variante après traitement à 150 classes est aussi rapportée
pour montrer la sensibilité à la représentation, sans sélectionner uniquement le meilleur score.

Ces intervalles conditionnent sur les modèles et la partition fixés : ils n’incluent
pas la variabilité d’un nouvel entraînement. La cohorte avait déjà été explorée en S6,
et la granularité avant/après reste différente. Il s’agit d’un complément exploratoire,
pas d’une validation externe ni d’un effet causal isolé du prétraitement. Les centres
sont présents dans le train et le test : l’expérience mesure leur reconnaissance,
pas la segmentation dans un centre inconnu. Les moyennes des cinq plis précédentes
restent inchangées et ne doivent pas être confondues avec ces nouveaux scores.

Livrables : `analysis/reports/s6_uncertainty/README.md`, `intervals.csv`,
`paired_differences.csv`, `summary.json` et `confidence_intervals.png`.
Les prédictions individuelles restent dans `data/analysis/s6_uncertainty/`.

## Synthèse du contrôle qualité par centre — 13 novembre

La synthèse couvre les 1 453 sujets inventoriés dans 55 centres. Elle retrouve 1 415
sujets retenus dans 54 centres, 38 exclus et aucune erreur active. Les exclusions
correspondent aux 37 cas R032 pour espacement non fiable et au cas R065 mis en
quarantaine pour annotation hors du support et du recadrage. Cette synthèse n’ajoute
aucune exclusion et ne remplace pas la relecture collective des décisions.

Parmi les 1 415 sujets retenus, cinq masques sont vides et conservés comme cas négatifs ;
1 410 sont non vides et aucun statut de masque n’est inconnu. Le délai post-AVC est
disponible pour 1 073 sujets ; 342 délais sont manquants ou invalides (24,2 %).
Huit centres retenus n’ont aucun délai utilisable. Le drapeau CHRONICITY est présent
pour 296 sujets et non renseigné pour 1 119 ; cette absence ne permet pas de déduire
un stade clinique. Deux sujets SOOP portent l’avertissement de fichier de métadonnées
vide : cet avertissement de fichier ne recouvre pas toutes les valeurs de délai absentes.

Les effectifs d’exclusion utilisent tout l’inventaire ; les pourcentages de métadonnées
utilisent seulement les sujets retenus de chaque centre. Pour R032, sans sujet retenu,
les taux sont non définis plutôt que nuls. Les colonnes inconnues ne sont pas assimilées
à des masques non vides. Le tableau détaillé, la figure, les motifs d’exclusion et
les avertissements sont disponibles dans `analysis/reports/qc_by_center/`.
