"""S7 : distance / erreur Dice, LOCO sur validation et test OOD indépendant."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/avc-mpl')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error
from center_distance import center_distance
from result_contract import load_results


def fit_alert(distances, failures):
    """Seuil appris exclusivement sur centres de validation, jamais sur OOD."""
    x, y = np.asarray(distances, float), np.asarray(failures, bool)
    if len(x) < 3 or len(np.unique(y)) < 2:
        return None
    values = np.unique(x)
    candidates = np.r_[np.nextafter(values[0], -np.inf),
                        (values[:-1] + values[1:]) / 2, values[-1]]
    scores = [balanced_accuracy_score(y, x > t) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def analyze(table, dice_failure=0.5):
    validation = table[table.role == 'validation'].copy()
    if len(validation) < 4:
        raise ValueError('Au moins quatre centres de validation sont nécessaires pour LOCO')
    predictions = []
    for index, held in validation.iterrows():
        fit = validation.drop(index)
        model = LinearRegression().fit(fit[['distance']], 1 - fit.dice)
        cutoff = fit_alert(fit.distance, fit.dice < dice_failure)
        predictions.append(dict(site=held.site, observed_loss=1-held.dice,
                                predicted_loss=float(np.clip(model.predict(pd.DataFrame({'distance':[held.distance]}))[0], 0, 1)),
                                alert=None if cutoff is None else bool(held.distance > cutoff),
                                threshold=cutoff))
    cutoff = fit_alert(validation.distance, validation.dice < dice_failure)
    fitted = LinearRegression().fit(validation[['distance']], 1-validation.dice)
    out = table.copy()
    out['predicted_loss'] = np.clip(fitted.predict(out[['distance']]), 0, 1)
    out['alert'] = None if cutoff is None else out.distance > cutoff
    correlation = spearmanr(validation.distance, 1-validation.dice)
    rho = float(correlation.statistic)
    loco = pd.DataFrame(predictions)
    summary = dict(validation_centers=len(validation), dice_failure=dice_failure,
                   spearman_validation=None if not np.isfinite(rho) else rho,
                   loco_mae=float(mean_absolute_error(loco.observed_loss, loco.predicted_loss)),
                   alert_threshold=cutoff, threshold_status='calibrated' if cutoff is not None else 'insufficient_classes',
                   loss_definition='1 - Dice ; delta_ID est calculé séparément sur les tests',
                   calibration='centres de validation uniquement ; LOCO exclut le centre évalué')
    summary['held_out_metrics'] = {}
    for role in ('id_test', 'ood_test'):
        group = out[out.role == role]
        if group.empty: continue
        observed = 1-group.dice
        metric = dict(centers=len(group), mae=float(mean_absolute_error(observed, group.predicted_loss)))
        failures = group.dice < dice_failure
        if cutoff is not None and failures.nunique() == 2:
            metric['balanced_accuracy'] = float(balanced_accuracy_score(failures, group.alert.astype(bool)))
        else:
            metric['balanced_accuracy'] = None
        summary['held_out_metrics'][role] = metric
    return out, loco, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--model', required=True); p.add_argument('--technique', required=True)
    p.add_argument('--fold', type=int, required=True)
    p.add_argument('--splits', type=Path, default=Path('splits.json'))
    p.add_argument('--histograms', type=Path, default=Path('data/processed/post_stats.parquet'))
    p.add_argument('--output', type=Path, default=Path('analysis/reports/s7'))
    p.add_argument('--dice-failure', type=float, default=0.5)
    args = p.parse_args()
    if not 0 < args.dice_failure < 1: raise ValueError('Seuil Dice entre 0 et 1')
    results, splits, fold = load_results(args.results, args.splits, args.model, args.technique, args.fold)
    hist = pd.read_parquet(args.histograms)
    if hist.case_id.duplicated().any(): raise ValueError('Histogrammes dupliqués')
    sites = {c['case_id']: c['site'] for c in splits['cases']}
    known = hist[hist.case_id.isin(sites)]
    if not (known.site == known.case_id.map(sites)).all(): raise ValueError('Centres des histogrammes incompatibles avec le split')
    needed = set(fold['train']) | set(results.case_id)
    if not needed <= set(hist.case_id): raise ValueError('Histogrammes manquants pour le pli ou les prédictions')
    reference = hist[hist.case_id.isin(fold['train'])]
    rows = []
    for (role, site), group in results.groupby(['role', 'site']):
        ref = reference[reference.site != site]
        scans = hist[hist.case_id.isin(group.case_id)]
        rows.append(dict(role=role, site=site, n=len(group), dice=float(group.dice.mean()),
                         distance=center_distance(np.stack(scans['hist']), np.stack(ref['hist']))))
    table, loco, summary = analyze(pd.DataFrame(rows), args.dice_failure)
    id_dice = results.loc[results.role == 'id_test', 'dice'].mean()
    table['delta_id'] = id_dice - table.dice
    summary.update(model=args.model, technique=args.technique, fold=args.fold,
                   id_dice=None if not np.isfinite(id_dice) else float(id_dice),
                   reference_cases=sorted(reference.case_id), distance='histogram_wasserstein',
                   warning='Calibration exploratoire ; centres peu nombreux, aucune garantie clinique')
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output/'centers.csv', index=False)
    loco.to_csv(args.output/'loco.csv', index=False)
    (args.output/'calibration.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for role, group in table.groupby('role'):
        ax.scatter(group.distance, 1-group.dice, label=role)
        for row in group.itertuples(): ax.annotate(row.site, (row.distance, 1-row.dice), fontsize=7)
    ax.set(xlabel='Distance au train du pli (sans le centre)', ylabel='Erreur : 1 − Dice')
    ax.legend(); fig.tight_layout(); fig.savefig(args.output/'distance_dice.png', dpi=160); plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k != 'reference_cases'}, indent=2))


if __name__ == '__main__': main()
