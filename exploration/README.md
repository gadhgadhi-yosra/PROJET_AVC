# Exploration du jeu de données (semaines 1–2)

Inventaire, statistiques par centre et contrôle qualité des 1 453 IRM T1 d'ISLES'26 / ATLAS v3.
Présentation : [`docs/Presentation_Exploration_Donnees.pptx`](../docs/Presentation_Exploration_Donnees.pptx).

## Données

Les scripts lisent `data/ATLAS3_Training_Raw/<SITE>/sub-*/ses-1/anat/` (ignoré par git).
L'archive est chiffrée ; avec le mot de passe reçu de NITRC :

```bash
mkdir -p data
openssl aes-256-cbc -md sha256 -d -a -in data/raw/atlas3_training_raw.tar.gz | tar -xzf - -C data
```

Environ 6 Go une fois extrait. Ne jamais committer les données ni le mot de passe.

## Scripts

Dépendances : `pip install nibabel numpy pandas scipy scikit-learn matplotlib pyarrow`

| Script | Rôle | Durée |
|---|---|---|
| `01_compute_stats.py` | Lit chaque patient (métadonnées, géométrie, volumes cerveau/lésion, histogramme d'intensité) → `outputs/subjects.parquet`, `outputs/histograms.npz` | ~14 min, 8 cœurs |
| `04_qc_report.py` | Règles de contrôle qualité → `outputs/qc_report.txt`, `outputs/qc_flags.csv` | secondes |
| `02_figures.py` | Tableau par centre `outputs/site_summary.csv` + figures dans `figures/` | ~1 min |
| `03_site_fingerprint_test.py` | Peut-on deviner le centre d'un scan ? → `outputs/site_fingerprint_test.txt` | secondes |

Lancer dans l'ordre 01 → 04 → 02 → 03. `01` accepte `--sites R029 R032` pour ne recalculer que certains centres.
`subjects.parquet` et `histograms.npz` contiennent des données par patient : ils ne sont pas versionnés, il faut lancer `01` en local.

## Points d'attention pour le pipeline

- La colonne du centre s'appelle `SITE` (pas `CENTER`) ; `CHRONICITY` est un drapeau 1/vide, pas un stade.
- Délai post-AVC manquant pour 341 patients ; 5 masques vides ; 2 fichiers metadata vides (SOOP).
- **R032** : espacement des voxels incohérent dans l'en-tête (cerveau de 31 cm, 2,7 L) → corriger ou exclure.
- **R029** : le fond vaut ~0,0007, pas 0 → masque du cerveau = voxels > valeur de fond, pas `> 0`.
- 41 masques codent la lésion 0,99999998 → toujours seuiller `mask > 0.5`.
- Le centre se devine à 68 % depuis la seule forme de l'histogramme d'intensité normalisé (hasard : 3,8 %).
