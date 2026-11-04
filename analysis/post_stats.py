"""Statistiques S5 après prétraitement : mesures par sujet (Spark), comparaison avant/après par centre, distances C6 v1."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('MPLCONFIGDIR', '/tmp/avc-mpl')
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from center_distance import BIN_EDGES, brain_histogram, center_distance

ROOT = Path(__file__).resolve().parents[1]
QC_BIN_EDGES = np.linspace(0, 1.5, 33)
MIN_SCANS = 5
OOD_COLOR, ID_COLOR = '#c2410c', '#0f766e'


def measure(row):
    """Mesures calculées sur l'image et le masque cérébral, jamais sur la lésion."""
    result = dict(case_id=row['case_id'], subject=row['subject'], site=row['site'], error='')
    try:
        folder = Path(row['output_dir'])
        x = nib.load(folder / 't1.nii.gz').get_fdata(dtype=np.float32)
        brain = nib.load(folder / 'brain.nii.gz').get_fdata(dtype=np.float32) > 0
        values = x[brain]
        p1, p25, p50, p75, p99 = np.percentile(values, [1, 25, 50, 75, 99])
        result.update(z_p1=float(p1), z_p25=float(p25), z_median=float(p50), z_p75=float(p75), z_p99=float(p99),
                      z_skew=float(stats.skew(values)), crop_voxels=int(np.prod(x.shape)),
                      hist=brain_histogram(x, brain).tolist())
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    return result


def spark_measures(rows, workers):
    from pyspark.sql import SparkSession
    os.environ['PYSPARK_PYTHON'] = sys.executable
    spark = (SparkSession.builder.master(f'local[{workers}]').appName('ATLAS-S5')
             .config('spark.ui.enabled', 'false').config('spark.driver.host', '127.0.0.1').getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    for module in ('center_distance.py', 'post_stats.py'):
        spark.sparkContext.addPyFile(str(Path(__file__).with_name(module)))
    try:
        def worker(row):
            from post_stats import measure
            return measure(row)
        started = time.perf_counter()
        return spark.sparkContext.parallelize(rows, len(rows)).map(worker).collect(), time.perf_counter() - started
    finally:
        spark.stop()


def eta_squared(frame, column):
    """Part de la variance d'une mesure expliquée par le centre."""
    values = frame[['site', column]].dropna()
    groups = values.groupby('site')[column]
    between = (groups.size() * (groups.mean() - values[column].mean()) ** 2).sum()
    return float(between / ((values[column] - values[column].mean()) ** 2).sum())


def wasserstein(a, b, edges):
    return float(np.abs(np.cumsum(a) - np.cumsum(b)).sum() * (edges[1] - edges[0]))


def distribution_std(histogram, edges):
    centers = (edges[:-1] + edges[1:]) / 2
    mean = (histogram * centers).sum()
    return float(np.sqrt((histogram * (centers - mean) ** 2).sum()))


def distance_table(after, before, splits):
    """Distance de chaque centre au pool d'entraînement privé de ce centre."""
    train = set(splits['train'])
    ood_centers = set(splits['ood_centers'])
    records = []
    for site, group in after.groupby('site'):
        reference = after[after.case_id.isin(train) & (after.site != site)]
        scans, ref = np.stack(group['hist']), np.stack(reference['hist'])
        old = before[before.site == site]
        old_ref = before[before.case_id.isin(train) & (before.site != site)]
        old_mean, old_ref_mean = np.stack(old['hist']).mean(axis=0), np.stack(old_ref['hist']).mean(axis=0)
        record = dict(site=site, n=len(group), group='OOD' if site in ood_centers else 'ID',
                      wasserstein=center_distance(scans, ref), mmd=center_distance(scans, ref, 'mmd'),
                      reference_scans=len(ref))
        # Divided by the reference spread: the two intensity scales are otherwise not comparable.
        record['relative_after'] = record['wasserstein'] / distribution_std(ref.mean(axis=0), BIN_EDGES)
        record['relative_before'] = (wasserstein(old_mean, old_ref_mean, QC_BIN_EDGES)
                                     / distribution_std(old_ref_mean, QC_BIN_EDGES))
        records.append(record)
    return pd.DataFrame(records).sort_values('wasserstein', ascending=False)


def site_table(joined):
    return joined.groupby('site').agg(
        n=('case_id', 'size'), raw_mean=('raw_mean', 'mean'), raw_std=('raw_std', 'mean'),
        z_p1=('z_p1', 'mean'), z_median=('z_median', 'mean'), z_p99=('z_p99', 'mean'), z_skew=('z_skew', 'mean'),
        brain_ml=('brain_ml', 'median'), lesion_ml=('lesion_ml', 'median'), crop_voxels=('crop_voxels', 'median'),
    ).reset_index()


def plot_raw_vs_processed(joined, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    counts = joined.site.value_counts()
    sites = joined[joined.site.map(counts) >= MIN_SCANS].groupby('site').raw_mean.median().sort_values().index
    figure, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
    panels = [('raw_mean', 'Intensité moyenne dans le cerveau, échelle log (avant)'), ('z_median', 'Médiane du z-score (après)')]
    for axis, (column, title) in zip(axes, panels):
        axis.boxplot([joined.loc[joined.site == s, column] for s in sites], showfliers=False, widths=.55,
                     medianprops=dict(color=ID_COLOR, linewidth=2), boxprops=dict(color='#57534e'),
                     whiskerprops=dict(color='#57534e'), capprops=dict(color='#57534e'))
        axis.set_title(title, loc='left', fontsize=11)
        axis.grid(axis='y', color='#e7e5e4', linewidth=.8)
        axis.set_axisbelow(True)
        axis.spines[['top', 'right']].set_visible(False)
    axes[0].set_yscale('log')
    axes[1].set_xticks(range(1, len(sites) + 1), sites, rotation=90, fontsize=8)
    axes[1].set_xlabel(f'Centres avec au moins {MIN_SCANS} scans, triés par intensité brute')
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_distances(distances, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = distances[distances.n >= MIN_SCANS].sort_values('wasserstein')
    figure, axis = plt.subplots(figsize=(8, max(5, .19 * len(data))))
    colors = [OOD_COLOR if g == 'OOD' else ID_COLOR for g in data.group]
    axis.hlines(data.site, 0, data.wasserstein, colors=colors, linewidth=2)
    axis.scatter(data.wasserstein, data.site, c=colors, s=26, zorder=3)
    axis.tick_params(axis='y', labelsize=7)
    axis.set_xlim(left=0)
    axis.set_xlabel('Distance de Wasserstein au pool d’entraînement sans le centre (écarts-types)')
    axis.scatter([], [], c=OOD_COLOR, label='Centre OOD')
    axis.scatter([], [], c=ID_COLOR, label='Centre ID')
    axis.legend(frameon=False, loc='lower right')
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
    parser.add_argument('--splits', type=Path, default=ROOT / 'splits.json')
    parser.add_argument('--reports', type=Path, default=ROOT / 'analysis/reports')
    parser.add_argument('--figures', type=Path, default=ROOT / 'figures')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.reports.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)

    subjects = pd.read_parquet(args.processed / 'subjects.parquet')
    done = subjects[subjects.status == 'ok']
    rows = done[['case_id', 'subject', 'site', 'output_dir']].to_dict('records')
    results, seconds = spark_measures(rows, args.workers)
    measures = pd.DataFrame(results)
    failed = measures[measures.error != '']
    measures = measures[measures.error == ''].drop(columns='error')
    # Subject-level histograms stay next to the private processed data.
    measures.to_parquet(args.processed / 'post_stats.parquet', index=False)

    before = pd.read_parquet(args.qc)
    before = before[before.status == 'ok']
    joined = measures.drop(columns='hist').merge(
        done[['case_id', 'raw_mean', 'raw_std', 'brain_ml', 'lesion_ml']], on='case_id')
    splits = json.loads(args.splits.read_text())
    # Raw scales span five orders of magnitude: the log keeps R042 from driving the variance.
    joined['log_raw_mean'], joined['log_raw_std'] = np.log10(joined.raw_mean), np.log10(joined.raw_std)
    sites = site_table(joined)
    distances = distance_table(measures, before, splits)
    sites.to_csv(args.reports / 'site_stats_before_after.csv', index=False)
    distances.to_csv(args.reports / 'center_distances_v1.csv', index=False)
    plot_raw_vs_processed(joined, args.figures / 's5_avant_apres_par_centre.png')
    plot_distances(distances, args.figures / 's5_distances_centres.png')

    stable = distances[distances.n >= MIN_SCANS]
    summary = dict(
        date=time.strftime('%Y-%m-%d'), subjects=len(measures), sites=int(measures.site.nunique()),
        read_errors=len(failed), seconds=seconds,
        subjects_not_processed=done.shape[0] - len(measures) + int((subjects.status == 'error').sum()),
        eta_squared={c: eta_squared(joined, c) for c in ('raw_mean', 'log_raw_mean', 'log_raw_std', 'z_p1', 'z_median', 'z_p99', 'z_skew')},
        raw_mean_site_range=[float(sites.raw_mean.min()), float(sites.raw_mean.max())],
        relative_distance_median_before=float(stable.relative_before.median()),
        relative_distance_median_after=float(stable.relative_after.median()),
        spearman_wasserstein_mmd=float(stats.spearmanr(stable.wasserstein, stable.mmd).statistic),
        spearman_before_after=float(stats.spearmanr(stable.relative_before, stable.relative_after).statistic),
        spearman_wasserstein_n=float(stats.spearmanr(stable.wasserstein, stable.n).statistic),
        spearman_mmd_n=float(stats.spearmanr(stable.mmd, stable.n).statistic),
        wasserstein_median_by_group=stable.groupby('group').wasserstein.median().to_dict(),
        scope=f'Corrélations limitées aux centres de {MIN_SCANS} scans ou plus',
    )
    (args.reports / 'post_stats_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if len(failed):
        print(failed[['case_id', 'error']].to_string())
        raise SystemExit(1)


if __name__ == '__main__':
    main()
