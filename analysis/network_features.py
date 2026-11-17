"""S8–S10 : extraction C4, distance C6 v2 et empreinte avant/après DANN."""
import argparse
import importlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/avc-mpl')
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def matrix(value):
    x = np.asarray(value, dtype=float)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError('Caractéristiques attendues : matrice finie n_scans × dimensions')
    return x


def center_distance(scans_centre, ref, metric='mmd'):
    """C6 v2. Standardisation sur la référence seule ; pas de masque lésionnel.

    MMD biaisée à noyau RBF, largeur médiane sur référence seulement.
    wasserstein : moyenne des W1 marginales, pas une W1 multivariée.
    """
    x, reference = matrix(scans_centre), matrix(ref)
    if x.shape[1] != reference.shape[1] or len(reference) < 2:
        raise ValueError('Dimensions incompatibles ou référence insuffisante')
    scale = reference.std(axis=0); scale[scale < 1e-8] = 1
    mean = reference.mean(axis=0)
    x, reference = (x-mean)/scale, (reference-mean)/scale
    if metric == 'wasserstein':
        return float(np.mean([wasserstein_distance(x[:,i], reference[:,i]) for i in range(x.shape[1])]))
    if metric != 'mmd': raise ValueError('Métrique inconnue')
    distances = cdist(reference, reference, 'sqeuclidean')
    upper = distances[np.triu_indices(len(reference), k=1)]
    positive = upper[upper > 0]
    if not len(positive): raise ValueError('Référence sans diversité')
    bandwidth = float(np.median(positive))
    value = (np.exp(-cdist(x,x,'sqeuclidean')/bandwidth).mean()
             + np.exp(-distances/bandwidth).mean()
             - 2*np.exp(-cdist(x,reference,'sqeuclidean')/bandwidth).mean())
    return float(np.sqrt(max(0, value)))


def load_bundle(path):
    with np.load(path, allow_pickle=False) as data:
        x, ids = matrix(data['features']), data['case_ids'].astype(str)
    if ids.ndim != 1 or len(x) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError('Identifiants absents, dupliqués ou désalignés')
    return x, ids


def extract(args):
    # Module explicitement choisi par le développeur, jamais un upload utilisateur.
    module, function = args.adapter.split(':')
    extractor = getattr(importlib.import_module(module), function)
    subjects = pd.read_parquet(args.subjects)
    subjects = subjects[subjects.status == 'ok'].sort_values('case_id')
    features = []
    for row in subjects.itertuples():
        value = np.asarray(extractor(str(Path(row.output_dir)/'t1.nii.gz')), dtype=float)
        if value.ndim != 1: raise ValueError('extract_features(path) doit renvoyer un vecteur, après pooling spatial')
        features.append(value)
    x = matrix(np.stack(features))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=x, case_ids=subjects.case_id.to_numpy(str))
    args.output.with_suffix('.json').write_text(json.dumps(dict(adapter=args.adapter,
        checkpoint_sha256=args.checkpoint_sha256, scans=len(x), dimensions=x.shape[1],
        contract='encodeur gelé, mode eval, sans labels ; même preprocessing que le checkpoint'), indent=2))


def distances(args):
    x, ids = load_bundle(args.features)
    splits = json.loads(args.splits.read_text())
    if not 0 <= args.fold < len(splits['folds']): raise ValueError('Pli inconnu')
    lookup = {c['case_id']:c['site'] for c in splits['cases']}
    if not set(ids) <= set(lookup): raise ValueError('Caractéristiques hors splits')
    train = set(splits['folds'][args.fold]['train'])
    if not train <= set(ids): raise ValueError('Caractéristiques train manquantes')
    sites = np.array([lookup[i] for i in ids]); rows = []
    for site in sorted(set(sites)):
        ref = np.array([i in train and s != site for i,s in zip(ids,sites)])
        rows.append(dict(site=site, n=int((sites==site).sum()), fold=args.fold,
                         mmd=center_distance(x[sites==site], x[ref]),
                         marginal_wasserstein=center_distance(x[sites==site], x[ref], 'wasserstein')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


def fingerprint(args):
    splits = json.loads(args.splits.read_text())
    if not 0 <= args.fold < len(splits['folds']): raise ValueError('Pli inconnu')
    lookup = {c['case_id']:c['site'] for c in splits['cases']}
    bundles = [load_bundle(p) for p in args.features]
    common = sorted(set.intersection(*(set(ids) for _,ids in bundles)))
    if not common or not set(common) <= set(lookup): raise ValueError('Pas de cohorte commune valide')
    fold = splits['folds'][args.fold]
    held_out = set(fold['val']) | set(fold['id_test']) | set(splits['ood_test'])
    common = [case for case in common if case in held_out]
    # Même patient, mêmes plis pour chaque représentation.
    labels = pd.Series([lookup[i] for i in common]); counts = labels.value_counts()
    common = [i for i in common if counts[lookup[i]] >= args.min_subjects]
    y = np.array([lookup[i] for i in common])
    if len(set(y)) < 2: raise ValueError('Au moins deux centres suffisamment grands requis')
    cv = list(StratifiedKFold(5, shuffle=True, random_state=2026).split(np.zeros(len(y)), y))
    shuffled = np.random.default_rng(2026).permutation(y)
    records = []
    for path, (x, ids) in zip(args.features, bundles):
        order = {case:i for i,case in enumerate(ids)}
        aligned = x[[order[i] for i in common]]
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        score = cross_val_score(model, aligned, y, cv=cv, scoring='balanced_accuracy')
        control_cv = StratifiedKFold(5, shuffle=True, random_state=2026)
        control = cross_val_score(model, aligned, shuffled, cv=control_cv, scoring='balanced_accuracy')
        records.append(dict(representation=path.stem, n=len(y), centers=len(set(y)),
                            balanced_accuracy=float(score.mean()), fold_std=float(score.std()),
                            shuffled_control=float(control.mean()), chance=1/len(set(y))))
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output/'network_fingerprint.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar([r['representation'] for r in records], [r['balanced_accuracy'] for r in records],
           yerr=[r['fold_std'] for r in records], color='#0f766e')
    ax.axhline(1/len(set(y)), linestyle='--', color='gray'); ax.set(ylim=(0,1), ylabel='Exactitude équilibrée')
    fig.tight_layout(); fig.savefig(args.output/'network_fingerprint.png', dpi=160); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    e = sub.add_parser('extract'); e.add_argument('--adapter', required=True)
    e.add_argument('--checkpoint-sha256', required=True)
    e.add_argument('--subjects', type=Path, default=Path('data/processed/subjects.parquet'))
    e.add_argument('--output', type=Path, required=True); e.set_defaults(run=extract)
    d = sub.add_parser('distance'); d.add_argument('--features', type=Path, required=True)
    d.add_argument('--fold', type=int, required=True)
    d.add_argument('--output', type=Path, required=True); d.set_defaults(run=distances)
    f = sub.add_parser('fingerprint'); f.add_argument('--features', type=Path, nargs='+', required=True)
    f.add_argument('--fold', type=int, required=True)
    f.add_argument('--min-subjects', type=int, default=20)
    f.add_argument('--output', type=Path, default=Path('analysis/reports/s10')); f.set_defaults(run=fingerprint)
    for parser in (d,f): parser.add_argument('--splits', type=Path, default=Path('splits.json'))
    args = p.parse_args()
    if hasattr(args, 'min_subjects') and args.min_subjects < 5: raise ValueError('Au moins 5 sujets par centre')
    args.run(args)


if __name__ == '__main__': main()
