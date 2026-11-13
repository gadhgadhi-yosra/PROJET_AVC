# S6 — incertitude de la reconnaissance du centre

Complément exploratoire à S6 ; ce lot réservé ne constitue pas un nouvel échantillon externe.

Lot réservé : 228 sujets ; entraînement : 912 ; 25 centres. 10 000 rééchantillonnages appariés.

| Classifieur | Représentation | Exactitude équilibrée | IC conditionnel 95 % |
|---|---|---:|---:|
| logreg | avant_32 | 66.9% | [61.1% ; 72.7%] |
| logreg | apres_30 | 45.4% | [39.7% ; 51.3%] |
| logreg | apres_150 | 49.9% | [43.5% ; 56.4%] |
| random forest | avant_32 | 70.1% | [64.6% ; 75.5%] |
| random forest | apres_30 | 51.1% | [45.3% ; 57.0%] |
| random forest | apres_150 | 59.1% | [53.0% ; 65.4%] |
| random forest | controle | 2.6% | [1.7% ; 3.6%] |

## Différence après − avant (30 / 32 classes)

- logreg : -21.5 points, intervalle [-27.4 ; -15.5] points.
- random forest : -19.0 points, intervalle [-25.5 ; -12.6] points.

## Interprétation et limites

Ces résultats complètent les moyennes de cinq plis S6 existantes ; ils ne les remplacent pas. Un intervalle de différence s’obtient par rééchantillonnage apparié, pas par soustraction des bornes des deux intervalles.

- Intervalles conditionnels aux modèles et à la partition fixés ; variabilité de réentraînement non incluse.
- Même cohorte déjà explorée en S6 ; aucune validation confirmatoire indépendante.
- Centres présents dans train et test : reconnaissance du centre, pas généralisation à un centre inconnu.
- Comparaison 32 classes avant / 30 après : différence de granularité restante.
- Bootstrap suppose des sujets indépendants au sein du centre ; couverture non garantie sur ce petit échantillon.
- Intervalles marginaux à 95 %, sans correction pour comparaisons multiples ; aucun Dice mesuré.

- Le contrôle utilise une seule permutation des labels du train ; son intervalle ne constitue ni une distribution nulle ni un test contre le hasard.

## Reproduction

```sh
.venv-spark/bin/python analysis/site_fingerprint_uncertainty.py
```

Sources méthodologiques : [bootstrap apparié, SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) ; [précautions sur les scores de validation croisée, scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html).
Prédictions individuelles et partition conservées localement dans `data/analysis/s6_uncertainty/`, ignorées par Git.
