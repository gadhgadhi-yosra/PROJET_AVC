# Adaptation de domaine — DANN et TENT

Deux modules autonomes de la question RQ3 (§3.5) et de l'adaptation continue contrôlée (§5.1), écrits pour se brancher dans la boucle d'entraînement 3D (contrat C4, Anas) dès qu'elle existera. Chacun est développé sur sa branche : `adaptation-dann` pour `dann.py`, `adaptation-tent` pour `tent.py` ; ce README et `tests/conftest.py` sont communs aux deux.

| Fichier | Rôle |
|---|---|
| `dann.py` | Couche d'inversion de gradient, planification de Ganin, classifieur de centre, `DANNWrapper`, `dann_loss` |
| `tent.py` | `configure_tent`, `collect_params`, entropie par voxel dans le cerveau, `Tent` avec `reset()` |
| `tests/` | Tests pytest sur de petits UNet MONAI 3D (CPU), plus un test par module sur de vrais patches C1 de 64³ |

```powershell
.venv-dl\Scripts\python -m pytest generalization\tests adaptation\tests -q
```

## DANN (Ganin et al., *JMLR* 2016)

```python
from monai.networks.nets import UNet
from adaptation.dann import DANNWrapper, center_labels, dann_loss

unet = UNet(spatial_dims=3, in_channels=1, out_channels=1, channels=(32, 64, 128, 256), strides=(2, 2, 2))
centres = sorted(centres_d_entrainement_du_pli)          # jamais un centre de validation ou de test
model = DANNWrapper(unet, n_centers=len(centres))

for epoch in range(n_epochs):
    model.set_progress(epoch / n_epochs)                  # lambda de 0 vers 1
    for batch in loader:
        segmentation, logits = model(batch["image"])
        loss, parts = dann_loss(segmentation, batch["label"], logits, center_labels(batch["site"], centres))
        loss.backward(); optimizer.step(); optimizer.zero_grad()
```

- **Couche d'inversion de gradient** : identité à l'aller ; au retour, le gradient est multiplié par −λ. Le classifieur apprend à reconnaître le centre, l'encodeur reçoit le gradient opposé et apprend à effacer l'empreinte du centre.
- **Planification de Ganin** : λ = 2 / (1 + exp(−10 p)) − 1, p = avancement de 0 à 1. La montée progressive est la solution de repli prévue si DANN est instable.
- **Classifieur de centre** : moyenne spatiale des caractéristiques du bottleneck, puis MLP (64 neurones, dropout 0,25). Le bottleneck est capté par un hook (`find_bottleneck` trouve le module le plus profond d'un UNet MONAI), sans modifier le réseau ; pour un autre réseau, passer `bottleneck=` et `feature_channels=`.
- **Perte** : Dice + CE (`DiceCELoss(sigmoid=True)` de MONAI, sortie binaire à un canal) + `weight` × CE du centre, avec `weight = 1`. Le λ de Ganin agit dans la couche d'inversion : l'encodeur reçoit bien −λ × ∇CE_centre, et le classifieur apprend à pleine vitesse. Mettre λ aussi dans la perte le ralentirait au début (λ ≈ 0) ; `weight` reste réglable.
- **Centres** : `center_labels` indexe les centres d'entraînement du pli ; un centre absent de la liste est refusé.

## TENT (Wang et al., *ICLR* 2021)

```python
from adaptation.tent import Tent, collect_params, configure_tent

model = configure_tent(modele_entraine)                       # seules les normalisations apprennent
tent = Tent(model, torch.optim.Adam(collect_params(model), lr=1e-3), steps=1)
for batch in nouveau_centre:
    logits = tent(batch["image"], brain=batch["brain"])       # sans étiquettes
tent.reset()                                                  # retour exact à l'état d'origine
```

- **`configure_tent`** : mode entraînement ; tout est gelé sauf les paramètres affines (gamma, beta) des BatchNorm, InstanceNorm et GroupNorm. Les BatchNorm utilisent les statistiques du lot courant (`track_running_stats=False`).
- **Perte** : entropie moyenne par voxel **dans le cerveau** (`brain`), sigmoïde pour une sortie à un canal, softmax sinon. Le fond, prédit avec certitude, écraserait sinon la moyenne.
- **`Tent`** : chaque appel évalue le lot puis fait `steps` pas de minimisation ; il renvoie les logits avant la mise à jour, comme l'implémentation de référence (évaluer d'abord, apprendre ensuite). `history` garde l'entropie de chaque pas.
- **`reset()`** restaure exactement les poids et l'état de l'optimiseur pris à la création de `Tent`.

> **Hypothèse forte : le réseau doit avoir des normalisations affines.** Le `UNet` MONAI par défaut utilise `InstanceNorm3d` **sans** paramètres affines : TENT n'aurait rien à adapter, et `configure_tent` lève une erreur claire. La boucle C4 doit utiliser `norm=("instance", {"affine": True})` (comme nnU-Net), BatchNorm ou GroupNorm. À convenir avec Anas.

## Vérification

18 tests, environ 15 secondes sur CPU (UNet MONAI 3D de 34 000 paramètres, volumes de 32³) :

- **DANN** : gradient inversé et multiplié par λ ; planification de Ganin (0 à p = 0, croissante, bornée) ; formes de la segmentation et des logits ; le classifieur lit bien la sortie du bottleneck ; sur l'encodeur, le gradient de la perte du centre avec inversion est l'opposé de celui sans inversion, et nul pour λ = 0, alors que celui du classifieur ne change pas ; un pas d'optimisation met à jour segmentation, encodeur et classifieur ; poids de la perte du centre.
- **TENT** : avec BatchNorm, InstanceNorm affine et GroupNorm, seuls les paramètres de normalisation sont entraînables et seuls eux changent ; BatchNorm sur les statistiques du lot ; l'entropie baisse sur un lot synthétique ; `reset()` restaure exactement les poids, l'optimiseur et la sortie ; refus du UNet MONAI par défaut ; entropie (0 pour une prédiction certaine, ln 2 au maximum, restreinte au cerveau).
- **Vrais volumes C1** (ignorés si `data/processed` est absent) : 3 sujets d'entraînement de centres différents, patches de 64³ centrés sur la lésion. DANN : cinq pas d'optimisation, pertes finies, la perte de segmentation baisse. TENT : l'entropie dans le cerveau baisse en 8 pas, seules les normalisations bougent, `reset()` restaure tout.

Ces tests vérifient la mécanique, pas l'efficacité : un réseau non entraîné sur trois patches ne dit rien du gain de DANN ou de TENT sur les centres OOD.

## Ce qui attend la boucle 3D d'Anas (contrat C4)

1. **Brancher DANN** : envelopper le réseau de la boucle dans `DANNWrapper`, passer le centre de chaque sujet (clé `site` de `load_processed`) et appeler `set_progress` à chaque époque. Le classifieur ne doit connaître que les centres d'entraînement du pli.
2. **Normalisations affines** dans le réseau, pour TENT.
3. **Entraînements avec et sans DANN** sur les mêmes plis, pour le tableau d'ablation RQ3, puis les modèles « avant / après DANN » pour le test d'empreinte de Yosra (§5.5).
4. **TENT sur les centres OOD** : un `Tent` par centre arrivant, `reset()` entre deux centres pour la stratégie « TENT seul » de l'adaptation continue (§5.1).
5. **Réglages** (λ maximal, γ de Ganin, taux d'apprentissage et nombre de pas de TENT) : sur la validation des plis seulement, jamais sur le test OOD.
