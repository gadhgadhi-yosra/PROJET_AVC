# Script oral du pipeline — objectif 7 minutes

Durée cible, **pas une répétition déjà effectuée**. Les huit séquences correspondent
à `Soutenance_Projet_AVC.pptx`. Les pauses et manipulations sont incluses dans le budget.
Lire à voix haute et noter le temps réellement mesuré dans la fiche en fin de document.

## 1 — Rôle et question · 00:00 à 00:40

« Bonjour. Ma contribution porte sur la préparation des données, l’étude des différences
entre centres et la démonstration du projet. Notre question est la suivante : un modèle
qui segmente correctement les lésions dans des centres connus reste-t-il performant
quand les images viennent d’un autre centre ? Pour étudier cela, il faut d’abord disposer
de données contrôlées et d’un protocole traçable. Je vais présenter les résultats déjà
obtenus, puis distinguer les expériences qui attendent encore les modèles entraînés. »

**Transition :** « La première étape est donc le contrôle des données. »

## 2 — Données et QC · 00:40 à 01:25

« L’inventaire contient 1 453 paires d’images et de masques provenant de 55 centres.
La livraison actuelle comporte 1 415 volumes validés et exportés. Les 37 cas du centre
R032 sont exclus pour un problème d’espacement. Un autre cas est en quarantaine car
son annotation dépasse le recadrage autorisé. Le split a été versionné : les autres
sujets gardent leurs affectations et les annotations brutes restent intactes.
Cette décision technique doit être relue avec l’équipe ; elle n’est pas une correction
médicale automatique. »

**Montrer :** les effectifs ; ne pas détailler le numéro du sujet à l’oral.

## 3 — Prétraitement et Spark · 01:25 à 02:15

« Le pipeline réoriente les images, les rééchantillonne à un millimètre, normalise les
intensités dans le cerveau puis effectue un recadrage et un contrôle qualité. Les masques
utilisent une interpolation adaptée aux labels. Les sorties sont réutilisables pour
l’entraînement et pour les analyses. Sur un pilote de huit volumes, quatre workers
ont donné le meilleur temps, environ 28 secondes, soit une accélération de 2,05.
C’est une mesure sur ce pilote, pas une promesse d’accélération identique sur toutes
les images. L’option N4 est disponible mais n’a pas été appliquée au jeu de référence. »

## 4 — Partition par centre · 02:15 à 03:00

« Le split C1 version 2 contient 916 cas pour l’entraînement et la validation, 231 pour
le test ID et 268 pour le test OOD. Les cinq plis séparent les centres du train et de
la validation. Six centres sont entièrement réservés au test OOD. Les réglages d’alerte
utilisent la validation, jamais le test OOD. Les anciens résultats CNN reçus de l’équipe
proviennent d’un autre split : je les conserve comme expérience historique, sans les
présenter comme des résultats de ce protocole. »

## 5 — Distributions d’intensité · 03:00 à 03:55

« La normalisation rapproche les échelles d’intensité, mais les distributions peuvent
encore avoir des formes différentes. Nous mesurons ces écarts avec des distances entre
centres. Cela décrit les images ; une distance élevée ne prouve pas à elle seule que
la segmentation échouera. Pour tester cette hypothèse, le code relie la distance aux
scores Dice avec une référence d’entraînement propre à chaque pli. L’expérience finale
sur C1 attend encore ces scores. De plus, la distance a participé au choix des centres
OOD : l’écart observé entre ID et OOD doit donc être interprété avec cette limite. »

## 6 — Empreinte du centre · 03:55 à 04:40

« Pour préciser l’incertitude de S6, nous réservons 228 sujets et entraînons les
classifieurs sur les 912 autres, avec les mêmes 25 centres. Pour la forêt aléatoire,
l’exactitude passe de 70,1 % avant traitement à 51,1 % après. Avec dix mille
rééchantillonnages appariés, la baisse estimée est de 19 points, avec un intervalle
de 12,6 à 25,5 points. Cet intervalle concerne les modèles fixés ; il ne couvre pas
un nouvel entraînement. La cohorte avait déjà été explorée. Il s’agit de reconnaître
le centre, pas de mesurer la qualité de segmentation. »

## 7 — Délai et Dice historiques · 04:40 à 05:40

« Les CSV récupérés permettent déjà une analyse exploratoire du délai post-AVC.
Pour la baseline CNN, nous avons 167 cas ID et 257 OOD avec les données nécessaires.
L’analyse tient compte du centre. En OOD, l’ajout du délai explique environ 3,8 points
supplémentaires de variance du Dice. Mais cet apport devient très faible lorsqu’on
prend aussi en compte le volume lésionnel. Nous ne pouvons donc pas affirmer que le
temps cause une amélioration de la segmentation. Les données manquantes sont comptées,
sans inventer de scores. Cette analyse utilise le protocole historique et ne remplace
pas les expériences finales C1. »

## 8 — Démonstration et fin · 05:40 à 07:00

**00–15 s :** ouvrir la démo locale et charger le phantom et son masque.

« Ici, le masque est fourni et le volume est synthétique. Cette séquence démontre
l’interface ; elle ne montre pas une prédiction du modèle. »

**15–40 s :** cliquer sur Analyser, montrer les coupes, les **2,091 mL** et le rapport JSON.

« L’application calcule le volume à partir de la géométrie de l’image. Elle peut aussi
recevoir un atlas déjà recalé et une carte d’incertitude si elles sont disponibles. »

**40–60 s :** ouvrir NiiVue, charger les mêmes fichiers et passer en 3D.

**60–80 s :** conclure et transmettre la parole.

« Les données, le raccordement des modules et la démonstration sont prêts. Il reste à
recevoir les poids et les scores compatibles pour terminer les expériences C1 et montrer
une inférence réelle. Je passe la parole pour la partie modèles. »

**Si la démo ne répond pas en 15 secondes :** lancer `demo_secours.mp4` et expliquer
qu’il s’agit des captures du même phantom. Ne pas multiplier les tentatives pendant l’oral.

## Fiche de répétition à remplir

| Mesure | Répétition 1 | Répétition 2 |
|---|---|---|
| Date et participants | À renseigner | À renseigner |
| Temps des diapositives 1–4 | À mesurer | À mesurer |
| Temps des diapositives 5–7 | À mesurer | À mesurer |
| Temps de démonstration | À mesurer | À mesurer |
| Temps total | À mesurer | À mesurer |
| Questions auxquelles je n’ai pas su répondre | À renseigner | À renseigner |
| Corrections décidées | À renseigner | À renseigner |

Pour raccourcir : supprimer les détails N4 et le second exemple sur les distances.
Conserver les distinctions **C1 / historique**, **masque fourni / inférence** et les limites scientifiques.
