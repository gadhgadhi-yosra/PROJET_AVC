# Projet AVC — travail de Yosra

Segmentation des lésions sur IRM T1 et analyse du domain shift inter-centres.
Les responsabilités sont décrites dans `docs/Repartition_Taches_Equipe.pdf`.

Commencer par [le suivi des 14 semaines](docs/yosra/SUIVI_SEMAINES.md), puis
[les sections de rapport](docs/yosra/RAPPORT.md) et [les supports d’oral](docs/yosra/SOUTENANCE.md).
Le [diaporama éditable](docs/yosra/Soutenance_Yosra.pptx) contient huit diapositives avec notes.

## Environnements

Le pipeline existant utilise Python 3.12, Java 17 et `.venv-spark` :

```sh
python3.12 -m venv .venv-spark
.venv-spark/bin/pip install -r requirements.txt
preprocessing/run_week4.sh run --workers 2
.venv-spark/bin/python preprocessing/validate_s4.py data/processed
.venv-spark/bin/python preprocessing/audit_delivery.py
.venv-spark/bin/python -m unittest discover -s tests -v
```

Les données brutes restent privées. Les résultats individuels, caractéristiques réseau et
prédictions doivent rester sous `data/` dans les dossiers ignorés par Git.

## Livrables

| Période | Code et documentation |
|---|---|
| S1–S2 | [Exploration](exploration/README.md) |
| S3–S4 | [Pipeline, QC, splits et export](preprocessing/SEMAINE4.md) |
| S5–S6 | [Statistiques et empreinte](analysis/README.md) |
| S7–S10 | [Analyses Dice, temps et représentations](analysis/ANALYSES_S7_S10.md) |
| S8–S12 | [Démo Gradio, NiiVue, atlas et Docker](demo/README.md) |
| Tout le projet | [Frontend web : tableau de bord + démo IRM](frontend/README.md) |
| S12–S14 | [Rapport](docs/yosra/RAPPORT.md), [oral et répétitions](docs/yosra/SOUTENANCE.md) |

Le code S7–S10 peut être testé sans modèle, mais ses résultats scientifiques exigent
les véritables prédictions et checkpoints de Fayfa/Anas. Aucun Dice n’est simulé dans
les livrables scientifiques. La démo distingue un masque fourni d’une inférence.

État C1/C2 vérifiable : `preprocessing/reports/delivery_audit.json`. L’export complet
refuse les cas manquants ; ne pas modifier silencieusement `splits.json` pour contourner le QC.

Livraison actuelle : **split v2, 1 415 cas validés et exportés**, 38 exclusions documentées.
Démo et image Docker testées ; [vidéo de secours synthétique](docs/yosra/demo_secours.mp4).
Voir [la remise technique et les entrées modèles encore nécessaires](docs/yosra/REMISE_EQUIPE.md).

## Raccordement avec les branches de Fayfa

Le chargeur C1, la généralisation et les modules DANN/TENT sont intégrés localement.
Consulter [le guide de raccordement](integration/README.md) pour les commandes,
les contrôles du split et la distinction avec les résultats historiques de baselines.

Analyse S8 désormais exécutée sur les anciens scores CNN :
[rapport délai / Dice](analysis/reports/s8_historical/README.md).
Ce résultat exploratoire conserve la partition historique et ne remplace pas C1 v2.

[La synthèse QC par centre](analysis/reports/qc_by_center/README.md) rassemble
les effectifs, exclusions, masques vides et métadonnées manquantes avec leurs dénominateurs.
