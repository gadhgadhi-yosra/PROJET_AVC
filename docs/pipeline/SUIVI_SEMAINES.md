# Suivi du pipeline — toutes les semaines

État local au **13 novembre 2026**. Périmètre : volet données, prétraitement et analyses de
`docs/Repartition_Taches_Equipe.pdf`. Les semaines futures sont préparées dès maintenant.

| Semaines | Livrables et état vérifiable |
|---|---|
| S1–S2 | Exploration, QC, statistiques et présentation existants dans `exploration/` et `docs/`. |
| S3 | Environnement Python/Java/Spark, inventaire, rangement et géométrie documentés. |
| S4 | **1 415 volumes validés et exportés**, split v2 : 916 train/val, 231 ID, 268 OOD. Benchmark et option N4 testée ; N4 inactive sur le jeu de référence. |
| S5 | Statistiques et distances recalculées sur 1 415 sujets / 54 centres. |
| S6 | Empreinte relancée sur 1 140 sujets / 25 centres : 69,7 % avant, 48,9 % après ; contrôle mélangé 4,0 %. |
| S7 | Analyse distance/Dice, Spearman, LOCO et seuil codés et testés. Résultats réels en attente du CSV C3 de Fayfa. |
| S8 | Analyse délai/Dice exécutée sur la baseline historique : 167 cas ID et 257 OOD complets, 999 permutations par groupe et sensibilité au volume. Rapport dans `analysis/reports/s8_historical/`. Protocole distinct de C1 v2. Extraction réseau et analyse finale C1 encore dépendantes des modèles/résultats. Démo opérationnelle. |
| S9–S10 | Rapports JSON, volume, alerte individuelle calibrée sans test, NiiVue coupes/3D testés dans Chrome. Atlas Harvard-Oxford et template MNI compatibles disponibles ; recalage réel effectué et contrôlé techniquement, relecture anatomique experte à prévoir. Comparaison DANN attend les checkpoints d’Anas. |
| S11 | Image Docker construite et testée ; adaptateur du modèle final préparé, poids C8 attendus. |
| S12 | README, rapport, provenance, dépendances et tests disponibles. Conclusions scientifiques dépendantes des modèles encore ouvertes. |
| S13–S14 | Huit diapositives éditables avec notes, vidéo MP4 de secours synthétique, captures et trame de répétitions produites. Répétitions et relecture collective à effectuer par l’équipe. |

## Quarantaine QC et version du split

Le cas `R065_sub-r065s019_ses-1` perdrait 77 / 4 376 voxels (1,76 %) au recadrage,
au-dessus de la tolérance existante de 1 %. Il est mis en quarantaine technique par
`preprocessing/qc_exclusions.json`. Les 37 cas R032 restent exclus : total 38 exclusions.
Les IRM et annotations brutes ne sont pas modifiées. La version 1 est archivée dans
`releases/splits_v1.json`. Tous les autres sujets gardent leurs rôles et leurs plis.
La relecture collective de cette décision et de la règle existante de perte ≤ 1 % reste nécessaire.
Ne pas mélanger des résultats entraînés sur v1 avec ceux de v2 sans préciser leur provenance.

## Vérifications

- 42 tests unitaires vérifiés (34 précédents, 4 S8 historiques et 4 contrôles de livraison) et cinq workflows CLI sur données synthétiques isolées.
- 1 415 sorties relues : `preprocessing/reports/validation_s4_complete_available.json`.
- Couverture exacte du split v2 : `preprocessing/reports/delivery_audit.json`.
- Chrome : volume 2,091 mL, NiiVue coupes et 3D, aucune erreur JavaScript ; captures dans `demo_verification/`.
- Docker : construction réussie et test des trois routes et du calcul de volume.
- Vidéo `demo_secours.mp4` clairement étiquetée synthétique ; aucune inférence simulée.
- Rapport de vérification machine : `verification.json`.

## Entrées encore nécessaires

Fayfa : CSV de Dice C3, checkpoint, architecture/prétraitement, adaptateurs de prédiction
et d’extraction, incertitude C7. Anas : modèles comparables baseline/augmentation/DANN
et modèle final C8. Voir [la remise technique](REMISE_EQUIPE.md).
Les branches de Fayfa ont depuis été récupérées et fusionnées : chargeur C1,
généralisation, DANN et TENT. Les CSV de baselines historiques sont disponibles, mais
leur partition diffère de C1 v2 ; les checkpoints entraînés restent absents.
Voir `integration/README.md` pour le raccordement et les vérifications.
Le code prêt ne constitue pas un résultat scientifique exécuté.

Aucun fichier privé n’a été publié ni envoyé. Le PDF historique
`docs/Explication_S4_S5_S6.pdf` reste inchangé.

## Lancement des analyses finales

La procédure `integration/run_remaining.py` contrôle le split et la livraison C3 avant
d’exécuter S7/S8. Audit du 12 novembre : scores C1 v2 absents pour les cinq plis.
Voir [les commandes et la livraison attendue](PROCHAINE_ETAPE.md).

## Relecture et préparation de l’oral — 12 novembre

Rapport et diaporama relus et harmonisés : voir [le compte rendu](RELECTURE.md).
Le [script oral](SCRIPT_ORAL.md) vise sept minutes et prévoit le secours de démonstration.
La [fiche de validation collective](VALIDATION_EQUIPE.md) réunit les pièces QC, atlas
et protocole. La répétition avec l’équipe et les décisions de l’équipe restent à effectuer.

## Complément S6 ajouté

Analyse d’incertitude exécutée : 912 sujets train, 228 réservés, 25 centres,
10 000 bootstrap appariés. Rapport, figure et oral mis à jour.
Voir `analysis/reports/s6_uncertainty/README.md` pour les intervalles conditionnels
et leurs limites. Quatre tests supplémentaires vérifient le rééchantillonnage.

## Synthèse QC ajoutée le 13 novembre

Tableau et figure pour les 55 centres : 1 415 retenus, 38 exclus, zéro erreur active,
5 masques vides et 342 délais manquants/invalides parmi les sujets retenus.
Les motifs d’exclusion, avertissements et dénominateurs sont explicités dans
`analysis/reports/qc_by_center/README.md`. Quatre tests supplémentaires réussissent.
