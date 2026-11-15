"""Frontend web du projet AVC : tableau de bord des résultats réels et démo IRM.

Le serveur ne lit que les rapports versionnés du dépôt (aucun chiffre simulé)
et réutilise demo/core.py pour l'analyse d'une IRM chargée par l'utilisateur.

    GRADIO_ANALYTICS_ENABLED=False .venv-demo/bin/python frontend/server.py
    -> http://127.0.0.1:8000
"""
import base64
import hashlib
import csv
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / 'static'
MAX_UPLOAD = 600 * 1024 * 1024

# Dossiers dont les figures PNG peuvent être servies (aucune donnée patient brute).
FIGURE_DIRS = ('figures', 'exploration/figures', 'baselines/outputs', 'analysis/reports',
               'preprocessing/reports')

FIGURES = [
    ('exploration/figures/01_site_fingerprint.png', 'Exploration', 'Empreinte des centres avant prétraitement'),
    ('exploration/figures/02_intensity_histograms.png', 'Exploration', 'Histogrammes d’intensité par centre'),
    ('exploration/figures/03_lesions.png', 'Exploration', 'Distribution des volumes lésionnels'),
    ('exploration/figures/04_time_since_stroke.png', 'Exploration', 'Délai depuis l’AVC'),
    ('exploration/figures/06_qc_anomalies.png', 'Exploration', 'Anomalies détectées au QC'),
    ('analysis/reports/qc_by_center/qc_by_center.png', 'QC', 'Synthèse QC par centre'),
    ('preprocessing/reports/benchmark_speedup.png', 'Prétraitement', 'Accélération Spark selon les workers'),
    ('figures/s5_avant_apres_par_centre.png', 'Domain shift', 'Statistiques par centre avant / après'),
    ('figures/s5_distances_centres.png', 'Domain shift', 'Distances entre centres'),
    ('figures/s6_empreinte_centre.png', 'Domain shift', 'Reconnaissance du centre (empreinte)'),
    ('analysis/reports/s6_uncertainty/confidence_intervals.png', 'Domain shift', 'Intervalles de confiance bootstrap'),
    ('baselines/outputs/fig1_dice_id_vs_ood.png', 'Modèles', 'Dice ID vs OOD (baselines)'),
    ('baselines/outputs/fig2_dice_by_lesion_size.png', 'Modèles', 'Dice selon la taille de lésion'),
    ('figures/s8_delai_metadonnees.png', 'Temps', 'Délai et métadonnées'),
    ('analysis/reports/s8_historical/time_dice.png', 'Temps', 'Délai depuis l’AVC et Dice'),
]


def clean(value):
    """Rend une valeur JSON-compatible (NaN/inf -> None)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def read_json(relative):
    path = ROOT / relative
    return json.loads(path.read_text()) if path.exists() else None


def read_csv(relative):
    path = ROOT / relative
    if not path.exists():
        return []
    rows = []
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            parsed = {}
            for key, value in row.items():
                if value is None or value.strip() == '':
                    parsed[key] = None
                    continue
                try:
                    number = float(value)
                    parsed[key] = int(number) if number.is_integer() and '.' not in value else number
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    return rows


def respond(payload):
    return JSONResponse(clean(payload))


app = FastAPI(title='AVC · Domain shift', docs_url='/api/docs', redoc_url=None)


@app.get('/api/overview')
def overview():
    qc = read_json('analysis/reports/qc_by_center/summary.json') or {}
    release = read_json('preprocessing/reports/release_v2.json') or {}
    verification = read_json('integration/reports/verification.json') or {}
    post = read_json('analysis/reports/post_stats_summary.json') or {}
    fingerprint = read_json('analysis/reports/s6_uncertainty/summary.json') or {}
    baselines = read_csv('baselines/outputs/results.csv')
    best = max(baselines, key=lambda r: r.get('dice_OOD') or 0) if baselines else None
    scores = {s['name']: s for s in fingerprint.get('scores', [])}
    return respond(dict(
        counts=qc.get('counts', {}),
        centers=dict(inventoried=qc.get('centers_inventoried'), retained=qc.get('centers_retained')),
        split=dict(train_validation=release.get('train_validation'), id_test=release.get('id_test'),
                   ood_test=release.get('ood_test'), cases=release.get('cases')),
        tests=dict(total=verification.get('total_passed'), status=verification.get('status'),
                   checkpoints=verification.get('trained_checkpoints_present')),
        shift=dict(before=post.get('relative_distance_median_before'),
                   after=post.get('relative_distance_median_after'),
                   fingerprint_before=scores.get('random forest_avant_32', {}).get('balanced_accuracy'),
                   fingerprint_after=scores.get('random forest_apres_30', {}).get('balanced_accuracy'),
                   chance=fingerprint.get('chance')),
        best_model=best,
    ))


@app.get('/api/qc')
def qc():
    return respond(dict(
        summary=read_json('analysis/reports/qc_by_center/summary.json'),
        centers=read_csv('analysis/reports/qc_by_center/qc_by_center.csv'),
        exclusions=read_csv('analysis/reports/qc_by_center/exclusion_reasons.csv'),
        delays=read_csv('analysis/reports/s8_metadata/delay_by_center.csv'),
    ))


@app.get('/api/preprocessing')
def preprocessing():
    return respond(dict(
        qc=read_json('preprocessing/reports/qc_summary.json'),
        validation=read_json('preprocessing/reports/validation_s4.json'),
        release=read_json('preprocessing/reports/release_v2.json'),
        audit=read_json('preprocessing/reports/delivery_audit.json'),
        tests=read_json('preprocessing/reports/test_summary.json'),
        benchmark=read_csv('preprocessing/reports/benchmark_summary.csv'),
        benchmark_scope=read_json('preprocessing/reports/benchmark_scope.json'),
    ))


@app.get('/api/shift')
def shift():
    return respond(dict(
        stats=read_json('analysis/reports/post_stats_summary.json'),
        distances=read_csv('analysis/reports/center_distances_v1.csv'),
        sites=read_csv('analysis/reports/site_stats_before_after.csv'),
        fingerprint=read_json('analysis/reports/s6_uncertainty/summary.json'),
    ))


@app.get('/api/models')
def models():
    return respond(dict(
        baselines=read_csv('baselines/outputs/results.csv'),
        historical=read_csv('analysis/reports/s8_historical/model_summary.csv'),
        temporal=read_json('analysis/reports/s8_historical/summary.json'),
    ))


RESULTS_C1 = 'data/predictions/c1_v2/results.csv'
RESULT_COLUMNS = {'model', 'technique', 'fold', 'case_id', 'site', 'dice'}


def technique_results(relative=RESULTS_C1, splits_relative='splits.json'):
    """Résume les Dice réels par technique (baseline, DANN, TENT…) et par rôle ID / OOD.

    Rôle tiré du split actif : id_test -> ID, ood_test -> OOD ; les cas de validation
    sont ignorés. Renvoie None tant que le fichier n'est pas livré.
    """
    path, splits_path = ROOT / relative, ROOT / splits_relative
    if not path.is_file() or not splits_path.is_file():
        return None
    splits = json.loads(splits_path.read_text())
    role_of = {c: 'ID' for c in splits.get('id_test', [])}
    role_of.update({c: 'OOD' for c in splits.get('ood_test', [])})
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        missing = RESULT_COLUMNS - set(reader.fieldnames or [])
        if missing:
            return dict(status='invalid', path=relative, reason='Colonnes manquantes : ' + ', '.join(sorted(missing)))
        groups, folds, ignored = {}, {}, 0
        for row in reader:
            role = role_of.get(row['case_id'])
            try:
                dice = float(row['dice'])
            except ValueError:
                dice = math.nan
            if role is None or not math.isfinite(dice):
                ignored += 1
                continue
            key = (row['technique'], role)
            groups.setdefault(key, []).append(dice)
            folds.setdefault((row['technique'], row['fold'], role), []).append(dice)
    summary = [dict(technique=t, role=r, n=len(v), mean_dice=sum(v) / len(v)) for (t, r), v in sorted(groups.items())]
    by_fold = [dict(technique=t, fold=f, role=r, n=len(v), mean_dice=sum(v) / len(v)) for (t, f, r), v in sorted(folds.items())]
    return dict(status='ready', path=relative, summary=summary, by_fold=by_fold, ignored=ignored,
                techniques=sorted({t for t, _ in groups}),
                sha256_16=hashlib.sha256(path.read_bytes()).hexdigest()[:16])


@app.get('/api/adaptation')
def adaptation():
    calibration = read_json('generalization/calibration.json') or {}
    return respond(dict(
        verification=read_json('integration/reports/verification.json'),
        readiness=read_json('integration/reports/remaining_readiness.json'),
        held_out_sites=calibration.get('held_out_sites', []),
        training_sites=len(calibration.get('training_sites', [])),
        splits=calibration.get('splits'),
        results=technique_results(),
        expected=dict(path=RESULTS_C1, columns=['model', 'technique', 'fold', 'case_id', 'site', 'dice']),
    ))


@app.get('/api/centers')
def centers():
    """Fiche par centre : QC, délais, distance au train, statistiques et rôle dans le split v2."""
    splits = read_json('splits.json') or {}
    ood, excluded = set(splits.get('ood_centers', [])), set(splits.get('excluded_centers', []))
    delays = {r['site']: r for r in read_csv('analysis/reports/s8_metadata/delay_by_center.csv')}
    distances = {r['site']: r for r in read_csv('analysis/reports/center_distances_v1.csv')}
    stats = {r['site']: r for r in read_csv('analysis/reports/site_stats_before_after.csv')}
    ranked = sorted((r['wasserstein'] for r in distances.values() if isinstance(r.get('wasserstein'), (int, float))))
    out = []
    for row in read_csv('analysis/reports/qc_by_center/qc_by_center.csv'):
        site = row['site']
        d, s, t = distances.get(site, {}), stats.get(site, {}), delays.get(site, {})
        w = d.get('wasserstein')
        out.append(dict(
            site=site,
            role='Exclu' if site in excluded else ('OOD' if site in ood else 'ID'),
            qc=row,
            delay=dict(median=t.get('median_days'), min=t.get('min_days'), max=t.get('max_days'),
                       available=t.get('delay_available'), missing=t.get('missing_or_invalid')),
            distance=dict(wasserstein=w, mmd=d.get('mmd'), relative_before=d.get('relative_before'),
                          relative_after=d.get('relative_after'),
                          rank=(ranked.index(w) + 1) if w in ranked else None, of=len(ranked)) if d else None,
            stats=dict(brain_ml=s.get('brain_ml'), lesion_ml=s.get('lesion_ml'), raw_mean=s.get('raw_mean')) if s else None,
        ))
    median_w = ranked[len(ranked) // 2] if ranked else None
    return respond(dict(centers=out, median_wasserstein=median_w))


@app.get('/api/conclusions')
def conclusions():
    qc = read_json('analysis/reports/qc_by_center/summary.json') or {}
    release = read_json('preprocessing/reports/release_v2.json') or {}
    post = read_json('analysis/reports/post_stats_summary.json') or {}
    fp = read_json('analysis/reports/s6_uncertainty/summary.json') or {}
    temporal = (read_json('analysis/reports/s8_historical/summary.json') or {}).get('temporal_by_role', {})
    verification = read_json('integration/reports/verification.json') or {}
    readiness = read_json('integration/reports/remaining_readiness.json') or {}
    baselines = read_csv('baselines/outputs/results.csv')
    scores = {s['name']: s for s in fp.get('scores', [])}
    best = max(baselines, key=lambda r: r.get('dice_OOD') or 0) if baselines else {}
    return respond(dict(
        counts=qc.get('counts', {}), centers=qc.get('centers_retained'), exclusions=read_csv('analysis/reports/qc_by_center/exclusion_reasons.csv'),
        shift=dict(before=post.get('relative_distance_median_before'), after=post.get('relative_distance_median_after'),
                   fp_before=scores.get('random forest_avant_32', {}).get('balanced_accuracy'),
                   fp_after=scores.get('random forest_apres_30', {}).get('balanced_accuracy'), chance=fp.get('chance'),
                   eta_max_after=max((v for k, v in (post.get('eta_squared') or {}).items() if k.startswith('z_')), default=None)),
        best=best, temporal=temporal,
        tests=verification.get('total_passed'), checkpoints=verification.get('trained_checkpoints_present'),
        experiments=readiness.get('experiments', {}), team_review_pending=release.get('team_review_pending'),
        adaptation_results=technique_results(),
    ))


def weekly_plan(relative='docs/yosra/SUIVI_SEMAINES.md'):
    """Lignes « | Semaines | Livrables | » du suivi de Yosra, texte inchangé."""
    path = ROOT / relative
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) == 2 and cells[0].startswith('S') and cells[0][1:2].isdigit():
            rows.append(dict(weeks=cells[0], text=cells[1]))
    return rows


@app.get('/api/contribution')
def contribution():
    verification = read_json('integration/reports/verification.json') or {}
    qc = read_json('analysis/reports/qc_by_center/summary.json') or {}
    fp = read_json('analysis/reports/s6_uncertainty/summary.json') or {}
    return respond(dict(
        weeks=weekly_plan(),
        tests=(verification.get('yosra_tests') or {}).get('passed'),
        counts=qc.get('counts', {}), centers=qc.get('centers_retained'),
        fingerprint_scans=fp.get('scans'),
        team=[
            dict(who='Anas', what='Baselines CNN 2.5D et ML (protocole historique)', page='modeles'),
            dict(who='Fayfa', what='Généralisation, augmentations, modules DANN et TENT', page='adaptation'),
            dict(who='Équipe', what='Checkpoints entraînés et résultats C1 v2 (fichier C3)', page='adaptation'),
        ],
    ))


@app.get('/api/figures')
def figures():
    return [dict(path=p, group=g, caption=c) for p, g, c in FIGURES if (ROOT / p).exists()]


@app.get('/files/{relative:path}', include_in_schema=False)
def files(relative: str):
    path = (ROOT / relative).resolve()
    allowed = any(path.is_relative_to((ROOT / d).resolve()) for d in FIGURE_DIRS)
    if not allowed or path.suffix.lower() != '.png' or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


@app.get('/vendor/niivue.es.js', include_in_schema=False)
def niivue():
    path = ROOT / 'demo/vendor/niivue.es.js'
    if not path.exists():
        raise HTTPException(404, 'NiiVue absent : lancer demo/setup_viewer.py')
    return FileResponse(path, media_type='text/javascript')


def save_upload(upload, folder, name):
    if upload is None or not upload.filename:
        return None
    suffix = '.nii.gz' if upload.filename.endswith('.nii.gz') else Path(upload.filename).suffix
    if suffix not in ('.nii', '.nii.gz'):
        raise ValueError(f'{upload.filename} : fichier NIfTI (.nii / .nii.gz) attendu')
    target = folder / f'{name}{suffix}'
    with target.open('wb') as handle:
        shutil.copyfileobj(upload.file, handle, length=1 << 20)
    if target.stat().st_size > MAX_UPLOAD:
        raise ValueError('Fichier trop volumineux')
    return target


@app.post('/api/analyze')
def analyze(t1: UploadFile = File(...), mask: UploadFile | None = File(None),
            uncertainty: UploadFile | None = File(None), position: float = 0.5):
    sys.path.insert(0, str(ROOT / 'demo'))
    from core import aligned, load_volume, plot_slices, predict, summarize
    folder = Path(tempfile.mkdtemp(prefix='avc-front-'))
    try:
        t1_path = save_upload(t1, folder, 't1')
        mask_path = save_upload(mask, folder, 'mask')
        unc_path = save_upload(uncertainty, folder, 'uncertainty')
        image, x = load_volume(t1_path)
        provenance = 'Masque fourni — aucune inférence exécutée'
        if mask_path is None:
            adapter = os.environ.get('AVC_PREDICT_ADAPTER')
            if not adapter:
                raise ValueError('Aucun modèle configuré : chargez un masque existant '
                                 'ou définissez AVC_PREDICT_ADAPTER.')
            result = predict(t1_path, adapter)
            mask_path, unc_path = result['mask_path'], result.get('uncertainty_path')
            provenance = f'Prédiction par {adapter}'
        lesion = aligned(mask_path, image, binary=True)
        unc = aligned(unc_path, image) if unc_path else None
        report = summarize(image, lesion, unc)
        report['provenance'] = provenance
        report['shape'] = list(image.shape)
        report['voxel_mm'] = [round(float(v), 3) for v in image.header.get_zooms()[:3]]
        figure = plot_slices(image, x, lesion, min(max(position, 0.0), 1.0))
        buffer = io.BytesIO()
        figure.savefig(buffer, format='png', dpi=110, facecolor='white')
        import matplotlib.pyplot as plt
        plt.close(figure)
        report['slices_png'] = base64.b64encode(buffer.getvalue()).decode()
        return respond(report)
    except (ValueError, OSError, KeyError) as error:
        raise HTTPException(400, str(error)) from error
    finally:
        shutil.rmtree(folder, ignore_errors=True)


@app.get('/', include_in_schema=False)
def index():
    return FileResponse(STATIC / 'index.html')


app.mount('/static', StaticFiles(directory=STATIC), name='static')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=os.environ.get('AVC_HOST', '127.0.0.1'), port=int(os.environ.get('AVC_PORT', '8000')))
