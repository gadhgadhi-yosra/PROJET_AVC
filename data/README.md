# Dataset AVC

Les données ne sont pas versionnées (accord d'utilisation ATLAS : ne pas redistribuer).

## Organisation

```
data/
├── ATLAS3_Training_Raw/          # données extraites, lues par les scripts
│   └── <SITE>/sub-*/ses-1/anat/  # 55 centres, 1 453 sujets
│       ├── *_desc-brain_T1w.nii.gz
│       ├── *_label-lesion_desc-T1lesion_mask.nii.gz
│       └── *_metadata.csv
└── raw/
    └── ATLAS_R3.0_raw.tar.gz     # archive d'origine (5,7 Go), sauvegarde
```

## Réextraire depuis l'archive

```bash
tar -xzf data/raw/ATLAS_R3.0_raw.tar.gz -C data
```

L'archive contient le dossier `ATLAS3_Training_Raw/` (1 453 IRM T1w). Prévoir environ 6 Go.

## Chemin depuis un notebook à la racine du projet

```python
from pathlib import Path
root = Path("data/ATLAS3_Training_Raw")
print(len(list(root.glob("*/sub-*/ses-*/anat/*_desc-brain_T1w.nii.gz"))))  # 1453
```

## Préparation de la semaine 3

Inventaire et géométrie RAS/1 mm : voir [`preprocessing/README.md`](../preprocessing/README.md). Les sorties intermédiaires sont dans `data/preprocessing_s3*/`, hors Git. Le dossier brut contient 1 453 paires IRM/masque ; les doublons de l’extraction partielle et le lien cassé ont été nettoyés après vérification le 13 octobre 2026.
