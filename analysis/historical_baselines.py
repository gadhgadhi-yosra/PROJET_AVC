"""S8 exploratoire sur le protocole historique des baselines, jamais renommé C1 v2."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/avc-mpl')
import numpy as np
import pandas as pd
from temporal_shift import fit_effects

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = 'cnn_pretrained_freeze-none'


def render_report(report):
    lines = ['# S8 — délai post-AVC et Dice des baselines historiques', '',
             'Analyse exploratoire des scores fournis, sans nouvelle inférence. **Ce résultat ne concerne pas C1 v2.**', '',
             'Modèle : `' + report['primary_model'] + '`. Partition historique : 903 train, 226 ID validation, 287 OOD.', '',
             '| Groupe | Cas complets / lignes fournies | Centres | Gain de R² du délai après centre | p permutation exploratoire |',
             '|---|---:|---:|---:|---:|']
    for role, result in report['temporal_by_role'].items():
        lines.append(f"| {role} | {result['n']} / {result['total']} | {result['centers']} | {result['incremental_r2_time']:.4f} | {result['permutation_p_time_given_center']:.3f} |")
    lines += ['', 'Le gain de R² est la part supplémentaire de variance du Dice expliquée par log(1 + délai), après prise en compte du centre. '
              'Les permutations mélangent les délais à l’intérieur de chaque centre (nombre dans summary.json). Les groupes ID et OOD sont analysés séparément.', '',
              '## Sensibilité au volume lésionnel', '']
    for role, result in report['temporal_by_role'].items():
        sensitivity = result['volume_sensitivity']
        if sensitivity is not None:
            lines.append(f"- {role} : après ajustement supplémentaire sur log(1 + volume), gain de R² du délai = {sensitivity['incremental_r2_time_given_center_and_volume']:.4f}.")
    lines += ['', '## Limites', ''] + ['- ' + value for value in report['limitations']]
    lines += ['', 'La relation observée ne prouve pas un effet causal du temps. Les résultats finaux sur C1 v2, les représentations réseau et la comparaison DANN restent à produire.', '',
              'Provenance, empreintes et valeurs détaillées : `summary.json`. Autres modèles : `model_summary.csv`. Métadonnées manquantes par centre : `missing_by_center.csv`.', '']
    return '\n'.join(lines)


def load_historical(metrics, partition):
    if partition.case.duplicated().any() or metrics.case.duplicated().any():
        raise ValueError('Identifiants historiques dupliqués')
    required = {'case', 'split', 'dice', 'model', 'gt_ml'}
    if not required <= set(metrics):
        raise ValueError('Colonnes de résultats manquantes')
    data = metrics.merge(partition[['case', 'site', 'set', 'days_post_stroke']],
                         on='case', how='left', validate='one_to_one')
    expected = data['set'].map({'id_val': 'ID', 'ood_test': 'OOD'})
    if expected.isna().any() or not expected.eq(data['split']).all():
        raise ValueError('Cas inconnu, entraînement ou rôle historique incompatible')
    present = data.dice.notna()
    if not np.isfinite(data.loc[present,'dice']).all() or not data.loc[present,'dice'].between(0, 1).all():
        raise ValueError('Dice invalide')
    data['days_post_stroke'] = pd.to_numeric(data.days_post_stroke, errors='coerce')
    data['valid_delay'] = np.isfinite(data.days_post_stroke) & data.days_post_stroke.ge(0)
    return data


def temporal_association(data, permutations, seed=2026):
    complete = data.loc[data.valid_delay & data.dice.notna()].copy()
    result = fit_effects(complete)
    rng = np.random.default_rng(seed)
    observed = result['incremental_r2_time']
    greater = 0
    for _ in range(permutations):
        shuffled = complete.copy()
        shuffled['days_post_stroke'] = shuffled.groupby('site').days_post_stroke.transform(
            lambda x: rng.permutation(x.to_numpy()))
        greater += fit_effects(shuffled)['incremental_r2_time'] >= observed - 1e-12
    # Sensibilité au volume lésionnel, mesuré par le masque de référence historique.
    if not np.isfinite(complete.gt_ml).all() or (complete.gt_ml < 0).any():
        raise ValueError('Volumes lésionnels invalides')
    centers = pd.get_dummies(complete.site, drop_first=True).to_numpy(float)
    base = np.c_[np.ones(len(complete)), centers, np.log1p(complete.gt_ml.to_numpy(float))]
    both = np.c_[base, np.log1p(complete.days_post_stroke.to_numpy(float))]
    y = complete.dice.to_numpy(float)
    sse = lambda x: float(np.sum((y-x@np.linalg.lstsq(x,y,rcond=None)[0])**2))
    sensitivity = None
    if np.linalg.matrix_rank(both) == both.shape[1] and len(y) > both.shape[1] + 2:
        sensitivity = {'incremental_r2_time_given_center_and_volume': (sse(base)-sse(both))/float(np.sum((y-y.mean())**2)),
                       'coefficient_log1p_days': float(np.linalg.lstsq(both,y,rcond=None)[0][-1])}
    result.update(total=len(data), missing_or_invalid_delay=int((~data.valid_delay).sum()),
                  missing_dice=int(data.dice.isna().sum()),
                  permutation_p_time_given_center=(greater+1)/(permutations+1),
                  permutations=permutations, seed=seed, volume_sensitivity=sensitivity)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--permutations', type=int, default=999)
    parser.add_argument('--output', type=Path, default=ROOT/'analysis/reports/s8_historical')
    args = parser.parse_args()
    if args.permutations < 1:
        raise ValueError('permutations >= 1')
    source = ROOT/'baselines/outputs'
    split_path = source/'split.csv'
    partition = pd.read_csv(split_path)
    counts = partition['set'].value_counts().to_dict()
    primary = None
    rows, hashes = [], {}
    for file in sorted((source/'fast_results').glob('*_cases.csv')):
        data = load_historical(pd.read_csv(file), partition)
        if data.model.nunique() != 1:
            raise ValueError('Un fichier doit décrire un seul modèle')
        model = data.model.iloc[0]
        hashes[str(file.relative_to(ROOT))] = hashlib.sha256(file.read_bytes()).hexdigest()
        for role, group in data.groupby('split'):
            rows.append(dict(model=model, role=role, n=len(group), centers=group.site.nunique(),
                             valid_dice=int(group.dice.notna().sum()), missing_dice=int(group.dice.isna().sum()),
                             mean_dice=group.dice.mean(), median_dice=group.dice.median(),
                             empty_masks=int((group.gt_ml==0).sum()),
                             valid_delay=int(group.valid_delay.sum())))
        if model == PRIMARY:
            primary = data
    if primary is None:
        raise ValueError('Baseline principale absente')
    results = {}
    for role, group in primary.groupby('split'):
        results[role] = temporal_association(group, args.permutations)
    report = dict(protocol='historical_baselines_00e2692', split_counts=counts,
                  split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
                  sources_sha256=hashes, primary_model=PRIMARY, temporal_by_role=results,
                  scope='Association exploratoire sur scores fournis ; aucune nouvelle inférence. Pas C1 v2.',
                  limitations=['ID est la validation historique ayant participé au choix du modèle.',
                    'Analyses ID/OOD séparées, cas complets ; les Dice manquants ne sont pas imputés.',
                    'Le cas à masque vide r065s002 a un Dice absent dans tous les CSV : compté dans les données manquantes, exclu des ajustements.',
                    'P-values exploratoires non corrigées pour les deux groupes ; pas de conclusion causale.',
                    'Prétraitement et partition historiques distincts de C1 v2.',
                    'La sensibilité au volume ne supprime pas tous les facteurs de confusion.'])
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output/'model_summary.csv', index=False)
    missing = primary.groupby(['split','site']).agg(n=('case','size'), valid_delay=('valid_delay','sum'))
    missing['missing_or_invalid_delay'] = missing.n-missing.valid_delay
    missing.to_csv(args.output/'missing_by_center.csv')
    (args.output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (args.output/'README.md').write_text(render_report(report))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,(role,group) in zip(axes,primary.groupby('split')):
        valid = group[group.valid_delay & group.dice.notna()]
        ax.scatter(valid.days_post_stroke,valid.dice,s=14,alpha=.5)
        ax.set(xscale='symlog',xlabel='Délai post-AVC (jours)',ylabel='Dice',
               title=f'{role} historique · {len(valid)} cas complets')
    fig.suptitle('Baseline CNN · association descriptive, protocole distinct de C1 v2')
    fig.tight_layout();fig.savefig(args.output/'time_dice.png',dpi=160);plt.close(fig)
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
