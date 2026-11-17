# Frontend web — Projet AVC

Application web unique qui présente tout le projet : exploration et QC, prétraitement Spark,
domain shift, modèles et Dice, adaptation (DANN / TENT), démo IRM avec vue 3D, et galerie des figures.

## Lancement

```sh
# environnement de la démo (FastAPI, nibabel, matplotlib déjà installés)
.venv-demo/bin/python demo/setup_viewer.py        # une fois : NiiVue local pour la vue 3D
GRADIO_ANALYTICS_ENABLED=False .venv-demo/bin/python frontend/server.py
```

Ouvrir http://127.0.0.1:8000. L'API JSON est documentée sur `/api/docs`.
Variables facultatives : `AVC_PORT`, `AVC_HOST`, `AVC_PREDICT_ADAPTER` (modèle de l'équipe).

Pour segmenter une IRM sans masque avec le modèle final C8 (checkpoint dans `data/modeles/`,
voir `demo/README.md`) :

```sh
AVC_PREDICT_ADAPTER=c8_bridge:predict GRADIO_ANALYTICS_ENABLED=False .venv-demo/bin/python frontend/server.py
```

## Principes

- Les chiffres viennent **uniquement** des rapports versionnés (`analysis/reports`,
  `preprocessing/reports`, `baselines/outputs`, `integration/reports`, `generalization/calibration.json`).
  Rien n'est simulé : quand une expérience manque (plis C1 v2, DANN/TENT entraînés), la page le dit.
- La démo réutilise `demo/core.py` ; les fichiers envoyés sont supprimés après l'analyse.
- Aucune étape de build et aucun CDN : HTML/CSS/JS et graphiques SVG faits maison, utilisables hors ligne.
- Seules les figures PNG des dossiers de résultats sont servies (`/files/...`), jamais les données brutes.

## Structure

| Fichier | Rôle |
|---|---|
| `server.py` | API FastAPI : `/api/overview`, `/qc`, `/preprocessing`, `/shift`, `/models`, `/adaptation`, `/figures`, `POST /api/analyze` |
| `static/index.html` | Squelette, navigation latérale |
| `static/app.js` | Une fonction de rendu par page (routage par `#ancre`) |
| `static/charts.js` | Barres, barres empilées, intervalles, nuage de points, courbe (SVG) |
| `static/styles.css` | Thème clair / sombre, mise en page responsive |

## Tests

```sh
MPLCONFIGDIR=/tmp/avc-mpl .venv-demo/bin/python -m unittest discover -s frontend/tests -v
```
