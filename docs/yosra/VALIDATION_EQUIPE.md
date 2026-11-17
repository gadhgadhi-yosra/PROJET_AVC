# Points de validation collective

Dossier préparé le 12 novembre 2026. **Aucune validation humaine n’est déclarée acquise.**
Les pièces sont locales ; aucune donnée ni demande n’a été envoyée automatiquement.

| Point | Preuves à ouvrir | Décision attendue | État |
|---|---|---|---|
| Quarantaine du cas R065_sub-r065s019_ses-1 | `data/processed/pending_case_qc.png`, `pending_case_review.json`, `preprocessing/qc_exclusions.json` | Confirmer l’exclusion technique v2 ou définir une correction/procédure documentée ; ne pas modifier silencieusement les labels | À relire |
| Tolérance de perte au recadrage ≤ 1 % | `preprocessing/week4.py`, QC par sujet et `docs/yosra/RAPPORT.md` | Accepter ou réviser la règle commune, avec impact sur versions et réentraînement | À relire |
| Adoption de C1 v2 | `splits.json`, `preprocessing/reports/release_v2.json`, `export_v2.json` | Confirmer le split de chaque entraînement et conserver la provenance des modèles v1/historiques | À confirmer |
| Atlas sur l’exemple réel | `data/demo/registration/qc_overlay.png`, `qc.json`, `registration.json` | Relecture anatomique compétente avant interprétation des régions ; l’accord de grille ne suffit pas | À relire |
| Conclusions S8 | `analysis/reports/s8_historical/README.md`, `summary.json` | Vérifier l’interprétation exploratoire, le rôle du volume et les données manquantes | À relire |
| Rapport et oral commun | `RAPPORT.md`, `Soutenance_Yosra.pptx`, `SCRIPT_ORAL.md` | Harmoniser les conclusions, les rôles et les transitions avec les autres parties | À relire |

Les chemins abrégés de la première ligne sont dans `data/processed/` ; ceux de l’atlas
dans `data/demo/registration/`. Les documents de rapport et d’oral sont dans `docs/yosra/`.

## Trace des décisions

Pour chaque point, renseigner : nom des personnes ayant relu, date, pièces examinées,
décision motivée, modifications demandées et version du livrable concerné.
Si le split ou le prétraitement change, recalculer les résultats affectés avant de les
réutiliser. Une décision ne doit pas être antidatée ni déduite d’un simple test technique réussi.

| Point | Personnes / date | Décision et justification | Actions / version |
|---|---|---|---|
| Quarantaine et recadrage | À renseigner | À renseigner | À renseigner |
| Split commun | À renseigner | À renseigner | À renseigner |
| Recalage atlas | À renseigner | À renseigner | À renseigner |
| Rapport et oral | À renseigner | À renseigner | À renseigner |
