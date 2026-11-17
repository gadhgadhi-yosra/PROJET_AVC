# Démo locale — Pipeline

## Lancement

```sh
python3.12 -m venv .venv-demo
.venv-demo/bin/pip install -r demo/requirements.lock.txt
GRADIO_ANALYTICS_ENABLED=False .venv-demo/bin/python demo/app.py
```

Ouvrir http://127.0.0.1:7860. Charger une IRM T1 et son masque lésionnel sur la même
grille, ou configurer `AVC_PREDICT_ADAPTER=equipe.modele:predict` avant le lancement.
Le modèle doit renvoyer `mask_path` et éventuellement `uncertainty_path`, dans la grille
native de l’IRM reçue. Le serveur ne télécharge ni n’exécute un checkpoint fourni via l’interface.

### Modèle final C8 de l’équipe

Le modèle de Fayfa (`adaptation/demo_adapter.py`, U-Net 3D « intensité », pli 0) demande
PyTorch et MONAI, absents de `.venv-demo`. Le pont `demo/c8_bridge.py` le lance dans
`.venv-integration` et récupère le masque, sans mélanger les environnements :

```sh
# checkpoint partagé par l’équipe (Drive), jamais versionné : data/modeles/ est ignoré par Git
ls data/modeles/unet3d_intensity_pli0_best.pt
AVC_PREDICT_ADAPTER=c8_bridge:predict GRADIO_ANALYTICS_ENABLED=False .venv-demo/bin/python demo/app.py
```

Vérifié le 17 novembre 2026 sur `sub-r031s006` (centre OOD R031) : Dice 0,745, volume prédit
33,8 mL pour 27,7 mL annotés, environ 35 s sur le CPU du Mac. Checkpoint SHA-256
`ee6b93af5d463cd9cc85fe80ae1b290b0b7ef9e0a41f3ea63ca346a75287abb6`. Un seul cas : ce n’est
pas une évaluation. Le modèle ne produit pas de carte d’incertitude.
Variables facultatives : `AVC_C8_PYTHON`, `PROJET_AVC_MODELE`, `AVC_C8_TIMEOUT`.

Le rapport précise s’il s’agit d’un masque fourni ou d’une prédiction. Il donne le volume
en mL (déterminant de l’affine / 1000), les régions si un atlas est disponible, et les
moyennes d’incertitude si une carte est disponible. Un masque vide est accepté. Les
probabilités non seuillées, grilles différentes et valeurs non finies sont refusées.

Pour l’alerte d’atypie, utiliser `analysis/calibrate_scan_ood.py`, définir
`AVC_REFERENCE_HISTOGRAMS=data/demo/reference_ood.npz`, puis charger un T1 **prétraité**
et son `brain.nii.gz`. Sans référence calibrée, l’application ne fabrique pas d’alerte.
Ne pas appliquer le seuil d’un centre à un scan individuel.

## NiiVue en local

La distribution navigateur officielle 0.69.0 est téléchargée sous `demo/vendor/`
(non versionnée) :

```sh
mkdir -p demo/vendor
curl --fail --location https://unpkg.com/@niivue/niivue@0.69.0/build/index.min.js \
  -o demo/vendor/niivue.es.js
python3 -m http.server 7861 --bind 127.0.0.1 --directory demo
```

Ouvrir http://127.0.0.1:7861/viewer.html, choisir l’IRM et le masque, puis basculer entre
coupes et rendu 3D. Les fichiers sont lus dans le navigateur via des URL de blobs.
Le visualiseur est une page locale séparée de Gradio ; un navigateur avec WebGL2 est requis.
Le bundle `index.min.js` expose le module encodé `esm`, chargé localement par la page.

## Atlas MNI

Fournir un template T1 MNI, son atlas de labels et les noms des régions sous forme JSON
(`{"1":"Nom de région", ...}`). Aucun atlas n’est deviné ou téléchargé implicitement.

```sh
ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 .venv-spark/bin/python demo/register_atlas.py \
  --t1 data/demo/t1.nii.gz --template /chemin/MNI_T1.nii.gz \
  --atlas /chemin/labels_MNI.nii.gz --output data/demo/registration
```

ANTsPy recale le template vers l’IRM, puis applique les mêmes transformations aux labels
avec interpolation `genericLabel`. Examiner la superposition de `template_native.nii.gz`
et du T1. Charger `atlas_native.nii.gz` dans Gradio ; définir `AVC_ATLAS_LABELS` pour les noms.
Un résultat anatomique n’est utilisable qu’après vérification de ce recalage.

## Docker

```sh
docker build -f demo/Dockerfile -t avc-demo .
docker run --rm -p 127.0.0.1:7860:7860 avc-demo
```

L’image ne contient ni données ATLAS ni checkpoints. Le branchement d’un vrai modèle
nécessite son module et ses dépendances, montés explicitement dans le conteneur. ANTsPy
reste dans l’environnement de prétraitement ; les atlas recalés sont chargés dans l’UI.
Le Dockerfile lance le serveur intégré Gradio et NiiVue.

## Secours de soutenance

```sh
.venv-spark/bin/python demo/make_backup.py
```

Cela crée un phantom synthétique sous `data/demo/synthetic/` et l’animation
`docs/pipeline/demo_secours_synthetique.gif`. C’est un support de visualisation clairement
étiqueté, pas une inférence ou un résultat scientifique. Une vidéo de la vraie inférence
reste à enregistrer après réception du modèle final.

Références d’implémentation : [Gradio Blocks](https://www.gradio.app/docs/gradio/blocks),
[chargement NiiVue](https://niivue.com/docs/loading/),
[recalage ANTsPy](https://antspyx.readthedocs.io/en/stable/registration.html).

## Livraison vérifiée du 11 novembre 2026

Le serveur intégré expose désormais Gradio **et** NiiVue sur le même port :
http://127.0.0.1:7860/viewer/ (le serveur 7861 ci-dessus est seulement une option autonome).
Préparer le bundle épinglé et vérifier son SHA-256 avec :

```sh
.venv-demo/bin/python demo/setup_viewer.py
.venv-spark/bin/python demo/fetch_atlas.py
```

Le second script télécharge explicitement Harvard-Oxford et le template correspondant
MNI152NLin6Asym. URLs, empreintes et règles de combinaison sont enregistrées dans
`data/demo/atlas/provenance.json`. Utiliser `template.nii.gz` et `atlas_labels.nii.gz`
de ce dossier dans la commande de recalage. Les noms sont dans `labels.json`.
L’exemple réel local est décrit dans `data/demo/real_example.json` ; son masque est
une annotation fournie, pas une prédiction. Contrôler son recalage avec :

```sh
.venv-spark/bin/python demo/check_registration.py
```

Le Dockerfile construit aussi le visualiseur ; aucune seconde commande serveur n’est
nécessaire dans le conteneur. Image locale vérifiée : `avc-demo:pipeline-v2`.
Sur ce Mac, si Docker n’est pas dans PATH :
`export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"`.

Les captures Chrome et la vidéo de secours proviennent d’un test réel de l’interface :

```sh
.venv-demo/bin/pip install -r demo/requirements-qa.txt
.venv-demo/bin/python demo/check_browser.py
.venv-demo/bin/python demo/make_video.py
```

Le test navigateur utilise Google Chrome installé localement et le serveur déjà lancé.
Le fichier `docs/pipeline/demo_secours.mp4` montre un phantom synthétique : volume 2,091 mL,
coupes et vue 3D. Il ne valide pas les performances d’un modèle de segmentation.
