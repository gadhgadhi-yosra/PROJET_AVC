"""Empreinte du centre S6 (§5.5) : peut-on deviner le centre d'un scan avant et après prétraitement ?

Même protocole que exploration/03_site_fingerprint_test.py : centres d'au moins MIN_N sujets,
exactitude équilibrée, 5 plis stratifiés. Aucune mesure n'utilise le masque de lésion.
"""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('MPLCONFIGDIR', '/tmp/avc-mpl')
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MIN_N = 20
BEFORE_COLOR, AFTER_COLOR = '#c2410c', '#0f766e'


def rebin(histograms, factor):
    """Regroupe les classes d'histogramme pour comparer à résolution proche."""
    usable = histograms.shape[1] - histograms.shape[1] % factor
    return histograms[:, :usable].reshape(len(histograms), -1, factor).sum(axis=2)


def load_features(processed, qc):
    """Assemble les mesures avant/après sur les mêmes sujets et les mêmes centres."""
    after = pd.read_parquet(processed / 'post_stats.parquet')
    subjects = pd.read_parquet(processed / 'subjects.parquet')[['case_id', 'raw_mean', 'raw_std', 'shape']]
    before = pd.read_parquet(qc)[['case_id', 'hist']].rename(columns={'hist': 'hist_before'})
    data = after.merge(subjects, on='case_id').merge(before, on='case_id')
    counts = data.site.value_counts()
    data = data[data.site.map(counts) >= MIN_N].reset_index(drop=True)
    z_hist = np.stack(data['hist'])
    features = {
        ('avant', 'Échelle brute (log moyenne, log écart-type)'): np.log10(data[['raw_mean', 'raw_std']].to_numpy(float)),
        ('avant', 'Forme de l’histogramme, 32 classes'): np.stack(data['hist_before']),
        ('après', 'Forme de l’histogramme, 30 classes'): rebin(z_hist, 5),
        ('après', 'Forme de l’histogramme, 150 classes'): z_hist,
        ('après', 'Percentiles et asymétrie du z-score'): data[['z_p1', 'z_p25', 'z_median', 'z_p75', 'z_p99', 'z_skew']].to_numpy(float),
        ('après', 'Taille du volume recadré'): np.stack(data['shape']).astype(float),
    }
    return data, features


def models():
    return {'logreg': make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0)),
            'random forest': RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1)}


def evaluate(features, y, cv):
    records = []
    for (stage, name), X in features.items():
        for model_name, model in models().items():
            scores = cross_val_score(model, X, y, cv=cv, scoring='balanced_accuracy')
            records.append(dict(stage=stage, features=name, model=model_name,
                                balanced_accuracy=float(scores.mean()), fold_std=float(scores.std())))
            print(records[-1], flush=True)
    return pd.DataFrame(records)


def plot(results, chance, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    best = results.loc[results.groupby('features', sort=False).balanced_accuracy.idxmax()].iloc[::-1]
    figure, axis = plt.subplots(figsize=(9, 3.8))
    colors = [BEFORE_COLOR if s == 'avant' else AFTER_COLOR for s in best.stage]
    axis.errorbar(best.balanced_accuracy, best.features, xerr=best.fold_std, fmt='none', ecolor='#78716c', capsize=3, zorder=2)
    axis.scatter(best.balanced_accuracy, best.features, c=colors, s=48, zorder=3)
    for value, label in zip(best.balanced_accuracy, best.features):
        axis.annotate(f'{value:.0%}', (value, label), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=9)
    axis.axvline(chance, color='#57534e', linestyle='--', linewidth=1)
    axis.annotate(f'Hasard {chance:.1%}', (chance, len(best) - 1), xytext=(5, 4), textcoords='offset points', fontsize=8, color='#57534e')
    axis.scatter([], [], c=BEFORE_COLOR, label='Avant prétraitement')
    axis.scatter([], [], c=AFTER_COLOR, label='Après prétraitement')
    axis.legend(frameon=False, loc='lower right')
    axis.set_xlim(0, 1)
    axis.set_xlabel('Exactitude équilibrée (meilleur des deux modèles, ± écart-type des plis)')
    axis.grid(axis='x', color='#e7e5e4', linewidth=.8)
    axis.set_axisbelow(True)
    axis.spines[['top', 'right']].set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed', type=Path, default=ROOT / 'data/processed')
    parser.add_argument('--qc', type=Path, default=ROOT / 'data/qc_s4/subjects.parquet')
    parser.add_argument('--reports', type=Path, default=ROOT / 'analysis/reports')
    parser.add_argument('--figures', type=Path, default=ROOT / 'figures')
    args = parser.parse_args()
    args.reports.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)

    data, features = load_features(args.processed, args.qc)
    y = data.site.to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    chance = 1 / len(set(y))
    results = evaluate(features, y, cv)
    results.to_csv(args.reports / 'site_fingerprint_s6.csv', index=False)
    plot(results, chance, args.figures / 's6_empreinte_centre.png')

    X = features[('après', 'Forme de l’histogramme, 150 classes')]
    forest = models()['random forest']
    shuffled = np.random.default_rng(2026).permutation(y)
    control = float(cross_val_score(forest, X, shuffled, cv=cv, scoring='balanced_accuracy').mean())
    predicted = cross_val_predict(forest, X, y, cv=cv)
    labels = sorted(set(y))
    recalls = pd.DataFrame(dict(site=labels, n=[int((y == s).sum()) for s in labels],
                                recall=recall_score(y, predicted, labels=labels, average=None)))
    recalls.sort_values('recall').to_csv(args.reports / 'site_fingerprint_recall_s6.csv', index=False)
    summary = dict(date=time.strftime('%Y-%m-%d'), scans=len(y), sites=len(labels), chance=chance,
                   shuffled_labels_control=control, min_subjects=MIN_N)
    (args.reports / 'site_fingerprint_s6_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(recalls.sort_values('recall').to_string(index=False))


if __name__ == '__main__':
    main()
