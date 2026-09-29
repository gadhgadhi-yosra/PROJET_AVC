# Dataset AVC

`raw/atlas3_training_raw.tar.gz` est un lien symbolique vers le fichier original :
`/Users/yosragadhgadhi/Downloads/atlas3_training_raw.tar.gz`.

Le fichier original doit rester à cet emplacement pour que le lien fonctionne.
Le lien évite une copie de 7,7 Go.

## Accès aux images

Le fichier est chiffré avec OpenSSL et encodé en Base64, malgré son extension `.tar.gz`.
Il faut la procédure et les informations de déchiffrement fournies avec le téléchargement avant de pouvoir extraire les IRM.
Prévoir suffisamment d’espace disque pour le déchiffrement et l’extraction.

## Chemin depuis un notebook à la racine du projet

```python
from pathlib import Path
dataset = Path("data/raw/atlas3_training_raw.tar.gz")
print("Dataset accessible :", dataset.is_file())
```
