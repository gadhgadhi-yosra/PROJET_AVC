# Remise technique à Fayfa et Anas — version 2

Le split actif est désormais **`splits.json`, version 2**. La version 1 est conservée
dans `docs/yosra/releases/splits_v1.json`. Aucun centre n’a changé de rôle et aucun cas
restant n’a été déplacé d’un pli à un autre. Un seul cas est mis en quarantaine technique :
`R065_sub-r065s019_ses-1`, en raison d’une annotation isolée hors du support de l’image.
Les fichiers bruts et leurs annotations restent intacts.

| Groupe | Cas v2 |
|---|---:|
| Entraînement / validation | 916 |
| Test ID | 231 |
| Test OOD | 268 |
| Total livré | 1 415 |

Décision et preuves : `preprocessing/qc_exclusions.json` et `preprocessing/reports/release_v2.json`.
La relecture collective de cette décision reste à effectuer. Si des modèles ont déjà été
entraînés avec v1, ne pas présenter leurs résultats comme une expérience v2 : conserver
leur provenance et convenir du protocole commun avant les comparaisons finales.

## Pour Fayfa

Le dataset local est `data/nnUNet_raw/Dataset001_ATLAS/` exporté et contrôlé
(`preprocessing/reports/delivery_audit.json` confirme sa présence). Les cinq plis sont
dans `splits_final.json` : les copier dans le dossier prétraité nnU-Net, comme décrit dans
`preprocessing/SEMAINE4.md`. Exécuter le contrôle officiel nnU-Net dans l’environnement
d’entraînement. Les vérités terrain ID/OOD sont séparées des labels d’entraînement.

À remettre pour les analyses de Yosra : CSV `model,technique,fold,case_id,site,dice`,
checkpoint, architecture, preprocessing, version/empreinte du split, fonction de prédiction
retournant un masque dans la grille native et fonction d’extraction des caractéristiques.
Placer les fichiers privés sous `data/predictions/` et `data/features/`.

## Pour Anas

`analysis/center_distance.py` fournit C6 v1 sur histogrammes ;
`analysis/network_features.py` fournit C6 v2 sur représentations. La référence doit être
le train du pli, sans le centre mesuré. La démo dispose d’un seuil individuel descriptif
calibré sur validation ; ce seuil n’est pas une probabilité d’échec ni le seuil par centre de S7.

À remettre : checkpoints baseline/augmentation/DANN comparables, métadonnées de leur
entraînement et modèle final C8. L’empreinte utilise une cohorte commune hors entraînement.

Ce document prépare la remise ; aucun fichier privé n’a été partagé ou envoyé automatiquement.
