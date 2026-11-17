"""Validation des résultats C3 et rattachement au split exact du modèle."""
import json
from pathlib import Path
import numpy as np
import pandas as pd


def load_results(csv, splits_path, model, technique, fold):
    splits = json.loads(Path(splits_path).read_text())
    data = pd.read_csv(csv)
    required = {'model', 'technique', 'fold', 'case_id', 'dice'}
    if not required <= set(data):
        raise ValueError(f'Colonnes C3 manquantes : {sorted(required - set(data))}')
    data = data[(data.model == model) & (data.technique == technique) & (data.fold == fold)].copy()
    if data.empty or data.case_id.duplicated().any():
        raise ValueError('Résultats absents ou plusieurs prédictions pour un même cas')
    if not np.isfinite(data.dice).all() or not data.dice.between(0, 1).all():
        raise ValueError('Dice doit être fini et compris entre 0 et 1')
    if not 0 <= fold < len(splits['folds']): raise ValueError('Pli inconnu')
    partition = splits['folds'][fold]
    lookup = {c['case_id']: c for c in splits['cases']}
    roles = {case: role for role, cases in (
        ('validation', partition['val']), ('id_test', partition['id_test']),
        ('ood_test', splits['ood_test'])) for case in cases}
    if not set(data.case_id) <= set(roles):
        raise ValueError('Cas inconnus, entraînement ou test ID hors des centres du pli')
    sites = data.case_id.map(lambda c: lookup[c]['site'])
    if 'site' in data and not (data.site == sites).all():
        raise ValueError('Centre C3 incompatible avec splits.json')
    data['site'], data['role'] = sites, data.case_id.map(roles)
    return data, splits, partition
