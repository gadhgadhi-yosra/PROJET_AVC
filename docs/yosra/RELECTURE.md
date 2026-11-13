# Relecture des supports — 12 novembre 2026

La relecture documentaire est effectuée. Elle ne vaut ni validation clinique du QC,
ni approbation collective, ni répétition réellement chronométrée avec Yosra.

| Vérification | Résultat |
|---|---|
| Cohorte et export C1 v2 | 1 415 = 916 train/validation + 231 ID + 268 OOD ; concordance avec `export_v2.json` |
| Cas complets de l’analyse S8 historique | 167 ID et 257 OOD ; concordance avec `summary.json` |
| Distinction entre les protocoles | La phrase « tous les modèles partagent le même split » est corrigée : C1 v2 est le protocole cible, les anciens CNN utilisent un autre split |
| Disponibilité des Dice | L’absence concerne les scores C1 v2 ; les scores historiques sont bien disponibles |
| Calibration dans les diapositives | La validation est précisée pour les seuils d’alerte ; cette phrase ne désigne plus indistinctement toutes les calibrations |
| Diaporama | Régénéré : 8 diapositives, 8 notes orales, 4 figures, aucun objet hors des limites de page |
| Date du diaporama | Mise à jour au 12 novembre 2026 |
| Script oral | 706 mots prononcés environ, budget cible de 7 minutes incluant pauses et manipulation ; durée réelle à mesurer |
| Démonstration | Masque fourni et phantom synthétique annoncés explicitement ; vidéo de secours en cas d’incident |
| Validation collective | Pièces et décisions attendues regroupées dans `VALIDATION_EQUIPE.md` ; aucune décision inventée |

Le rendu typographique final dans PowerPoint n’a pas été vérifié avec cette application.
Le contrôle du fichier porte sur sa structure, la présence des figures et les limites des
objets ; il ne détecte pas toutes les substitutions de polices ou tous les débordements de texte.

## À ouvrir pour la répétition

1. `Soutenance_Yosra.pptx` — support affiché.
2. `SCRIPT_ORAL.md` — texte et repères de temps.
3. `demo_secours.mp4` — secours, clairement synthétique.
4. `VALIDATION_EQUIPE.md` — points à discuter après la répétition.

Les expériences C1 dépendantes des scores et checkpoints restent en attente ; leur état
est suivi séparément dans `PROCHAINE_ETAPE.md`.
