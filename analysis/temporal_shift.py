"""S8 : associations du centre et du délai post-AVC avec le Dice, cas complets."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from result_contract import load_results


def fit_effects(data):
    y = data.dice.to_numpy(float)
    center = pd.get_dummies(data.site, drop_first=True).to_numpy(float)
    intercept = np.ones((len(data), 1))
    days = np.log1p(data.days_post_stroke.to_numpy(float))[:, None]
    designs = {'null': intercept, 'center': np.c_[intercept, center],
               'time': np.c_[intercept, days], 'both': np.c_[intercept, center, days]}
    if len(data) <= designs['both'].shape[1] + 2:
        raise ValueError('Trop peu de cas complets pour le nombre de centres')
    if np.linalg.matrix_rank(designs['both']) < designs['both'].shape[1]:
        raise ValueError('Effets centre et délai non identifiables (colinéarité)')
    sse, coefficients = {}, {}
    for name, x in designs.items():
        coef = np.linalg.lstsq(x, y, rcond=None)[0]
        sse[name] = float(np.sum((y-x@coef)**2)); coefficients[name] = coef
    if sse['null'] <= 1e-15: raise ValueError('Dice constant')
    return dict(n=len(data), centers=int(data.site.nunique()),
                r2={name:1-value/sse['null'] for name,value in sse.items()},
                incremental_r2_time=(sse['center']-sse['both'])/sse['null'],
                incremental_r2_center=(sse['time']-sse['both'])/sse['null'],
                coefficient_log1p_days=float(coefficients['both'][-1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', required=True, type=Path)
    p.add_argument('--model', required=True); p.add_argument('--technique', required=True)
    p.add_argument('--fold', type=int, required=True)
    p.add_argument('--splits', type=Path, default=Path('splits.json'))
    p.add_argument('--subjects', type=Path, default=Path('data/qc_s4/subjects.parquet'))
    p.add_argument('--output', type=Path, default=Path('analysis/reports/s8_time'))
    p.add_argument('--permutations', type=int, default=999)
    args = p.parse_args()
    if args.permutations < 1: raise ValueError('permutations >= 1')
    data, _, _ = load_results(args.results, args.splits, args.model, args.technique, args.fold)
    metadata = pd.read_parquet(args.subjects)[['case_id', 'days_post_stroke']]
    data = data.merge(metadata, on='case_id', how='left', validate='one_to_one')
    data['days_post_stroke'] = pd.to_numeric(data.days_post_stroke, errors='coerce')
    valid = np.isfinite(data.days_post_stroke) & (data.days_post_stroke >= 0)
    missing = data.assign(missing_delay=~valid).groupby('site').agg(n=('case_id','size'), missing_delay=('missing_delay','sum'))
    complete = data.loc[valid].copy()
    result = fit_effects(complete)
    rng = np.random.default_rng(2026); permuted = []
    for _ in range(args.permutations):
        shuffled = complete.copy()
        shuffled['days_post_stroke'] = shuffled.groupby('site').days_post_stroke.transform(lambda x: rng.permutation(x))
        permuted.append(fit_effects(shuffled)['incremental_r2_time'])
    result.update(total=len(data), missing_or_invalid_delay=int((~valid).sum()),
                  permutation_p_time_given_center=float((1+np.sum(np.asarray(permuted)>=result['incremental_r2_time']))/(args.permutations+1)),
                  permutations=args.permutations, seed=2026,
                  interpretation='Association descriptive, pas causalité ; CHRONICITY non utilisé comme stade ; cas complets uniquement')
    args.output.mkdir(parents=True, exist_ok=True)
    missing.to_csv(args.output/'missing_by_center.csv')
    (args.output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    import os
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/avc-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7,4))
    for role, g in complete.groupby('role'): ax.scatter(g.days_post_stroke, g.dice, alpha=.5, s=12, label=role)
    ax.set(xscale='symlog', xlabel='Délai post-AVC (jours)', ylabel='Dice'); ax.legend()
    fig.tight_layout(); fig.savefig(args.output/'time_dice.png', dpi=160); plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
