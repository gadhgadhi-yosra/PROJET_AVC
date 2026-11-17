# Soutenance — partie Yosra

Trame de huit diapositives, à intégrer au support commun. Ne présenter comme résultats
que les valeurs présentes dans les rapports produits ; garder les analyses dépendantes
de modèles dans une section « protocole préparé » tant que leurs expériences sont absentes.

Le fichier éditable est [Soutenance_Yosra.pptx](Soutenance_Yosra.pptx).
Pour le régénérer après mise à jour des figures : installer `python-pptx>=1,<2` dans
`.venv-demo`, puis exécuter `.venv-demo/bin/python docs/yosra/build_slides.py`.

## 1 — Pourquoi un nouvel hôpital peut changer le problème (40 s)

**À afficher :** même tâche de segmentation, distributions d’intensité différentes selon le centre.
**À dire :** « Mon travail prépare les données, mesure ce décalage et fournit une démo.
Nous voulons distinguer une bonne performance sur les centres connus d’une bonne généralisation. »

## 2 — Des données contrôlées et traçables (45 s)

**À afficher :** 1 453 paires, 55 centres ; 37 exclusions R032 + 1 quarantaine QC ; 1 415 validés et exportés, split v2.
**À dire :** « Une sortie existante n’est pas suffisante : je vérifie sa géométrie, ses valeurs
et son masque. Je conserve les masques vides et je signale les métadonnées manquantes. »

## 3 — Le pipeline Spark (50 s)

**À afficher :** inventaire → RAS → 1 mm → z-score → recadrage → QC → Parquet.
**Figure :** `preprocessing/reports/benchmark_speedup.png`.
**À dire :** « Sur le pilote de huit volumes, quatre workers donnent ×2,05. Ce résultat
ne prédit pas à lui seul le temps du lot complet. N4 est une option séparée. »

## 4 — Simuler un centre jamais vu (40 s)

**À afficher :** train/validation 916 ; test ID 231 ; test OOD 268 ; cinq plis par centre.
**À dire :** « Les expériences finales doivent partager le split C1 v2 ; les baselines historiques utilisent une autre partition. Pour un modèle donné, un centre
de validation n’appartient pas à son entraînement ; les OOD ne servent pas à régler les seuils. »

## 5 — Le z-score ne répond pas à toute la question (55 s)

**Figures :** `figures/s5_avant_apres_par_centre.png`, `figures/s5_distances_centres.png`.
**À dire :** « Les échelles sont alignées, mais la forme des distributions peut encore varier.
La distance doit ensuite être confrontée au Dice, sans conclure trop tôt à un échec du modèle. »

## 6 — Empreinte et incertitude (45 s)

**Figure :** `analysis/reports/s6_uncertainty/confidence_intervals.png`.
**À dire :** « Sur un lot réservé de 228 sujets, la forêt reconnaît moins bien le centre
après traitement : baisse estimée de 19 points, intervalle conditionnel de 12,6 à
25,5 points. Les modèles sont fixés ; il ne s’agit ni d’une validation externe ni
d’une preuve d’amélioration de la segmentation. »

## 7 — Délai et Dice sur la baseline historique (45 s)

**À afficher :** 167 cas ID et 257 OOD complets, protocole historique distinct de C1 v2.
**Figure :** `analysis/reports/s8_historical/time_dice.png`.
**À dire :** « L’association observée en OOD après prise en compte du centre s’atténue
fortement lorsqu’on ajoute le volume lésionnel. Nous ne pouvons pas conclure à un
effet causal du délai. Cette analyse exploratoire utilise les anciens scores ; les
expériences finales sur notre split v2 restent à produire. »

## 8 — Démonstration (60–90 s)

1. Ouvrir Gradio localement ; charger le phantom synthétique et son masque.
2. Dire explicitement : « Ce masque est fourni ; ce passage démontre la visualisation. »
3. Afficher les coupes, le volume et télécharger le JSON.
4. Ouvrir NiiVue et basculer en 3D si le navigateur WebGL2 est disponible.
5. Montrer les régions/alertes uniquement si leurs véritables entrées ont été préparées.
6. Après réception de C8, remplacer cette séquence par l’inférence réelle et enregistrer une vidéo locale.

Secours immédiat : [vidéo MP4](demo_secours.mp4) et `demo_secours_synthetique.gif`. Il n’est pas présenté comme une vidéo
de segmentation automatique ni comme une preuve d’efficacité du modèle.

## Questions du jury

| Question | Réponse |
|---|---|
| Pourquoi séparer par centre ? | Pour mesurer la généralisation à un environnement absent du train. |
| Pourquoi ne pas corriger silencieusement le dernier masque ? | Cela changerait l’annotation et pourrait biaiser l’évaluation ; la décision doit être traçable. |
| Pourquoi tolérer l’arrondi du z-score ? | Le pas de stockage vaut 0,001 ; l’erreur maximale attendue vaut 0,0005. |
| Une grande distance prédit-elle toujours un mauvais Dice ? | Non : c’est une hypothèse à tester, pas une conséquence mathématique. |
| Le temps est-il un effet causal ? | Non, l’analyse est descriptive, conditionnée au centre et limitée aux délais disponibles. |
| Pourquoi un seuil différent pour un scan ? | Un histogramme individuel et une moyenne de centre n’ont pas la même variabilité. |
| Une faible empreinte DANN prouve-t-elle un meilleur modèle ? | Non : il faut également examiner ses scores de segmentation hors centre. |
| Docker garantit-il la reproductibilité ? | Il aide à figer l’environnement ; les données, poids, paramètres et seeds restent nécessaires. |

## Deux répétitions à réaliser

- Répétition 1 : environ 7 minutes pour ce bloc, vérifier les transitions avec Fayfa/Anas,
  noter le temps réel de chaque diapositive et de l’inférence. Couper les explications secondaires si nécessaire.
- Répétition 2 : simuler une panne de réseau ou de modèle, utiliser le support local,
  poser les questions ci-dessus et vérifier que les conclusions correspondent aux preuves.

Ces répétitions et la relecture croisée sont des actions humaines prévues, pas des tâches
que le dépôt permet de déclarer effectuées automatiquement.

Le [script oral chronométré](SCRIPT_ORAL.md) propose une cible de sept minutes.
Les [points de validation collective](VALIDATION_EQUIPE.md) disposent de pièces et de champs de décision.
