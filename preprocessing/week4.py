"""Prétraitement S4, QC, partitions par centre et export nnU-Net."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import nibabel as nib
from nibabel.processing import resample_to_output, resample_from_to
import numpy as np
import pandas as pd
from pipeline import inventory, ROOT

VERSION = 's4-v1.2'
# Z-scores are stored as int16 with scl_slope = Z_STEP (about half the disk space of float32).
Z_STEP = 1e-3
# Measured output/input size ratio is about 1.07 with int16 storage; keep a margin.
OUTPUT_SIZE_FACTOR = 1.3
# Largest share of lesion voxels that may fall outside the image-based crop.
MAX_CROP_DROP_FRACTION = 0.01


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def case_id(row):
    return '_'.join(row[k] for k in ('site', 'subject', 'session'))


def load_pair(row):
    a, b = nib.load(row['t1_path']), nib.load(row['mask_path'])
    if len(a.shape) != 3 or a.shape != b.shape or not np.allclose(a.affine, b.affine, atol=1e-3):
        raise ValueError('invalid_image_mask_grid')
    if not np.isfinite(a.affine).all() or abs(np.linalg.det(a.affine[:3,:3])) < 1e-9:
        raise ValueError('invalid_affine')
    a, b = nib.as_closest_canonical(a), nib.as_closest_canonical(b)
    x, y = a.get_fdata(dtype=np.float32), b.get_fdata(dtype=np.float32)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('nonfinite_voxels')
    # Validate slabs to avoid multiple full float arrays on 512x512 volumes.
    for start in range(0, y.shape[2], 16):
        slab = y[:,:,start:start+16]
        if not np.all(np.isclose(slab, 0) | np.isclose(slab, 1)):
            raise ValueError('nonbinary_source_mask')
    values, counts = np.unique(x[::4,::4,::4], return_counts=True)
    background = float(values[counts.argmax()])
    brain = x > background
    if np.count_nonzero(brain) < 10:
        raise ValueError('empty_brain')
    labels = y > .5
    return a, x, labels, brain, background


def fingerprint(row, margin):
    items = [VERSION, margin]
    for key in ('t1_path', 'mask_path'):
        p = Path(row[key]); stat = p.stat()
        items.append([str(p.resolve()), stat.st_size, stat.st_mtime_ns])
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()


def transform(row, destination, margin=5, resume=True, reserve_bytes=2*1024**3):
    out = Path(destination) / row['subject']
    marker = out / 'qc.json'
    try:
        signature = fingerprint(row, margin)
    except Exception:
        marker.unlink(missing_ok=True)
        raise
    if resume and marker.exists():
        old = json.loads(marker.read_text())
        if old.get('signature') == signature and all((out / f).is_file() for f in ('t1.nii.gz','mask.nii.gz','brain.nii.gz')):
            return old
    # Invalidate the old completion marker before any attempted recomputation.
    marker.unlink(missing_ok=True)
    if shutil.disk_usage(Path(destination)).free < reserve_bytes:
        raise RuntimeError('disk_space_under_reserve')
    a, x, labels, brain, background = load_pair(row)
    source_lesion_ml = float(labels.sum() * abs(np.linalg.det(a.affine[:3,:3])) / 1000)
    image = resample_to_output(nib.Nifti1Image(x, a.affine), (1,1,1), order=1, cval=background)
    target = (image.shape, image.affine)
    label_img = resample_from_to(nib.Nifti1Image(labels.astype(np.uint8), a.affine), target, order=0)
    brain_img = resample_from_to(nib.Nifti1Image(brain.astype(np.uint8), a.affine), target, order=0)
    x = image.get_fdata(dtype=np.float32)
    y = np.asarray(label_img.dataobj, dtype=np.uint8)
    support = np.asarray(brain_img.dataobj, dtype=bool)
    if source_lesion_ml > 0 and not y.any():
        raise ValueError('lesion_lost_in_resampling')
    # Crop from the image alone: no ground-truth-dependent input geometry.
    bounds = []
    for axis in range(3):
        indices = np.flatnonzero(support.any(axis=tuple(i for i in range(3) if i != axis)))
        if not len(indices):
            raise ValueError('brain_lost_in_resampling')
        bounds.append((max(0,int(indices[0])-margin), min(x.shape[axis],int(indices[-1])+margin+1)))
    slices = tuple(slice(lo,hi) for lo,hi in bounds)
    # Stray label voxels outside the image-based box have no image support: drop them
    # when they are a negligible part of the lesion, otherwise keep the explicit failure.
    total_lesion = int(y.sum())
    dropped = total_lesion - int(y[slices].sum())
    if dropped > MAX_CROP_DROP_FRACTION * total_lesion:
        raise ValueError('lesion_outside_image_based_crop')
    mean, std = float(x[support].mean(dtype=np.float64)), float(x[support].std(dtype=np.float64))
    if not np.isfinite(std) or std < 1e-8:
        raise ValueError('constant_brain_intensity')
    normalized = np.zeros(x.shape, np.float32)
    normalized[support] = (x[support] - mean) / std
    translation = np.eye(4)
    translation[:3,3] = [lo for lo,hi in bounds]
    affine = image.affine @ translation
    normalized, y, support = normalized[slices], y[slices], support[slices]
    if np.abs(normalized).max() > np.iinfo(np.int16).max * Z_STEP:
        raise ValueError('zscore_outside_int16_range')
    quantized = np.rint(normalized / Z_STEP).astype(np.int16)
    # Statistics describe the stored (quantized) values.
    normalized = quantized.astype(np.float32) * np.float32(Z_STEP)
    outside = int(np.count_nonzero((y > 0) & ~support))
    info = dict(signature=signature, version=VERSION, background=background,
                raw_mean=mean, raw_std=std, normalized_mean=float(normalized[support].mean()),
                normalized_std=float(normalized[support].std()), shape=list(normalized.shape),
                crop_start=[lo for lo,hi in bounds], brain_ml=float(support.sum()/1000),
                source_lesion_ml=source_lesion_ml, lesion_ml=float(y.sum()/1000),
                lesion_volume_change_fraction=(float(y.sum()/1000)/source_lesion_ml-1) if source_lesion_ml else None,
                lesion_outside_brain_voxels=outside, lesion_voxels_dropped_by_crop=dropped,
                empty_mask=not bool(y.any()), output_dir=str(out.resolve()))
    out.mkdir(parents=True, exist_ok=True)
    # Marker is committed last, so interrupted pairs cannot pass resume checks.
    marker.unlink(missing_ok=True)
    for name, data in [('t1',quantized), ('mask',y), ('brain',support.astype(np.uint8))]:
        temp = out / f'.{name}.{os.getpid()}.nii.gz'
        result = nib.Nifti1Image(data, affine)
        result.set_qform(affine, code=1); result.set_sform(affine, code=1)
        if name == 't1':
            result.set_data_dtype(np.int16); result.header.set_slope_inter(Z_STEP, 0)
        nib.save(result,temp)
        temp.replace(out / f'{name}.nii.gz')
    write_json(marker, info)
    return info


def work(row, output, qc_only=False, margin=5, resume=True, reserve_bytes=2*1024**3):
    result = dict(row, case_id=case_id(row), status='ok', error='')
    started = time.perf_counter()
    cache = Path(output) / '_qc_cache' / f'{case_id(row)}.json'
    cache_signature = None
    try:
        if qc_only and row.get('t1_path') and row.get('mask_path'):
            cache_signature = hashlib.sha256((fingerprint(row, margin) + json.dumps(row,sort_keys=True)).encode()).hexdigest()
            if resume and cache.exists():
                previous = json.loads(cache.read_text())
                if previous.get('cache_signature') == cache_signature and previous['status'] in ('ok','excluded'):
                    previous['seconds'] = 0.0
                    return previous
        if row['site'] == 'R032':
            result.update(status='excluded', error='R032_spacing_unreliable')
        elif not row['t1_path'] or not row['mask_path']:
            raise ValueError('missing_or_ambiguous_pair')
        elif qc_only:
            a,x,y,brain,bg = load_pair(row)
            vals = x[brain]
            scale = float(np.percentile(vals,99))
            if scale <= 0:
                raise ValueError('invalid_intensity_scale')
            h,_ = np.histogram(np.clip(vals/scale,0,1.5), bins=np.linspace(0,1.5,33))
            vml = abs(float(np.linalg.det(a.affine[:3,:3]))) / 1000
            result.update(background=bg, raw_mean=float(vals.mean(dtype=np.float64)), raw_std=float(vals.std(dtype=np.float64)),
                          brain_ml=float(brain.sum()*vml), lesion_ml=float(y.sum()*vml),
                          empty_mask=not bool(y.any()), hist=(h/h.sum()).tolist())
        else:
            result.update(transform(row,output,margin,resume,reserve_bytes))
        days = row.get('days_post_stroke')
        result['invalid_days'] = days is not None and (not np.isfinite(days) or days < 0)
        if days is not None and not np.isfinite(days): result['days_post_stroke'] = None
    except Exception as exc:
        result.update(status='error', error=f'{type(exc).__name__}: {exc}')
    result['seconds'] = time.perf_counter() - started
    if qc_only:
        result['cache_signature'] = cache_signature
        write_json(cache, result)
    return result


def spark_run(rows, output, workers, qc_only=False, margin=5, resume=True, reserve_bytes=2*1024**3):
    from pyspark.sql import SparkSession
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ.setdefault('SPARK_LOCAL_IP','127.0.0.1')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'):
        os.environ[key] = '1'
    spark = (SparkSession.builder.master(f'local[{workers}]').appName('ATLAS-S4')
             .config('spark.ui.enabled','false').config('spark.driver.host','127.0.0.1').getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    for module in ('pipeline.py','week4.py'):
        spark.sparkContext.addPyFile(str(Path(__file__).with_name(module)))
    output = str(Path(output).resolve())
    Path(output).mkdir(parents=True,exist_ok=True)
    try:
        def worker(row):
            from week4 import work
            return work(row,output,qc_only,margin,resume,reserve_bytes)
        started = time.perf_counter()
        small, large = [], []
        for row in rows:
            try:
                big = row['site'] != 'R032' and row['t1_path'] and np.prod(nib.load(row['t1_path']).shape) > 40_000_000
            except Exception:
                big = False  # Let the worker record the unreadable file as a QC error.
            (large if big else small).append(row)
        result = []
        if small:
            result.extend(spark.sparkContext.parallelize(small,len(small)).map(worker).collect())
        if large:
            # One partition serializes high-memory cases on this local workstation.
            result.extend(spark.sparkContext.parallelize(large,1).map(worker).collect())
        seconds = time.perf_counter()-started
    finally:
        spark.stop()
    return result,seconds


def run(args):
    rows = inventory(args.root)
    if args.sites: rows = [r for r in rows if r['site'] in args.sites]
    if args.cases:
        selected_ids = set(json.loads(args.cases.read_text()))
        rows = [r for r in rows if case_id(r) in selected_ids]
        if {case_id(r) for r in rows} != selected_ids: raise ValueError('Sélection contenant des cas inconnus')
    if args.limit: rows = rows[:args.limit]
    if not rows: raise ValueError('Aucun sujet sélectionné')
    if len({r['subject'] for r in rows}) != len(rows):
        raise ValueError('Plusieurs sessions par sujet : définir un contrat de sortie avant traitement')
    args.output.mkdir(parents=True,exist_ok=True)
    if not args.qc_only:
        pending = [r for r in rows if r['site']!='R032' and not (args.output/r['subject']/'qc.json').exists()]
        estimated = sum(Path(r['t1_path']).stat().st_size * OUTPUT_SIZE_FACTOR for r in pending if r['t1_path'])
        if estimated + 2*1024**3 > shutil.disk_usage(args.output).free:
            raise RuntimeError(f'Espace insuffisant : estimation prudente {estimated/1024**3+2:.1f} Gio libres nécessaires. Utiliser --output sur un disque adapté ou --limit pour valider.')
    results,seconds = spark_run(rows,args.output,args.workers,args.qc_only,args.margin)
    frame = pd.DataFrame(results)
    frame.to_parquet(args.output/'subjects.parquet',index=False)
    summary = dict(version=VERSION, subjects=len(rows), sites=len(frame.site.unique()),
                   statuses=frame.status.value_counts().to_dict(), seconds=seconds,
                   qc_only=args.qc_only, workers=args.workers,
                   metadata_warnings=int(frame.inventory_warnings.ne('').sum()))
    write_json(args.output/'summary.json',summary)
    numeric = [c for c in ('brain_ml','lesion_ml','raw_mean','raw_std','normalized_mean','normalized_std') if c in frame]
    if numeric: frame[frame.status=='ok'].groupby('site')[numeric].agg(['count','mean','median']).to_csv(args.output/'site_summary.csv')
    print(json.dumps(summary,indent=2))
    if (frame.status=='error').any(): raise SystemExit(1)


def make_splits(frame, seed=2026, folds=5):
    good = frame[frame.status=='ok'].copy().sort_values('case_id')
    if good.subject.duplicated().any(): raise ValueError('Subject leakage: repeated subject')
    groups = {s:g.case_id.tolist() for s,g in good.groupby('site')}
    candidates = [s for s in groups if len(groups[s])>=5]
    if len(candidates)<max(9,folds+3): raise ValueError('Pas assez de centres pour un découpage fiable')
    # Feature-only selection, before any model fitting or Dice evaluation.
    hist = {s:np.mean(np.stack(g['hist'].to_list()),axis=0) for s,g in good.groupby('site')}
    reference = np.mean(np.stack(list(hist.values())),axis=0)
    distances = {s:float(np.abs(np.cumsum(hist[s])-np.cumsum(reference)).sum()*(1.5/32)) for s in groups}
    ranked = sorted(candidates,key=lambda s:(len(groups[s]),s))
    ood = []
    for stratum in np.array_split(ranked,3):
        by_distance = sorted(stratum.tolist(),key=lambda s:(distances[s],s))
        ood.extend([by_distance[0],by_distance[-1]])
    ood = sorted(set(ood))
    rng = np.random.default_rng(seed)
    training,id_test=[],[]
    for site, ids in sorted(groups.items()):
        if site in ood: continue
        shuffled = list(rng.permutation(ids))
        count = max(1,int(round(.2*len(ids)))) if len(ids)>=2 else 0
        id_test.extend(shuffled[:count]); training.extend(shuffled[count:])
    lookup = dict(zip(good.case_id,good.site))
    remaining = sorted(set(lookup[i] for i in training))
    bins=[[] for _ in range(folds)]; loads=[0]*folds
    for site in sorted(remaining,key=lambda s:(-sum(lookup[i]==s for i in training),s)):
        index=min(range(folds),key=lambda i:(loads[i],i))
        bins[index].append(site); loads[index]+=sum(lookup[i]==site for i in training)
    splits=[]
    for sites in bins:
        val=sorted(i for i in training if lookup[i] in sites)
        train=sorted(i for i in training if lookup[i] not in sites)
        splits.append(dict(train=train,val=val,validation_centers=sorted(sites),
                           id_test=sorted(i for i in id_test if lookup[i] not in sites)))
    result=dict(version=1,seed=seed, data_validation='raw_QC_complete; full_processed_QC_required_before_training', selection='6 centres: 3 strates de taille, distance histogramme proche/lointaine; aucun Dice utilisé',
                excluded_centers=['R032'], empty_masks='retained_as_negative_cases',
                ood_centers=ood,train=sorted(training),id_test=sorted(id_test),
                ood_test=sorted(i for s in ood for i in groups[s]),folds=splits,
                cases=[dict(case_id=r.case_id,subject=r.subject,site=r.site,session=r.session) for r in good.itertuples()],
                center_selection=[dict(site=s,n=len(groups[s]),histogram_distance=distances[s],ood=s in ood) for s in sorted(groups)],
                excluded_cases=frame.loc[frame.status!='ok',['case_id','status','error']].sort_values('case_id').to_dict('records'))
    validate_splits(result)
    return result


def validate_splits(s):
    lookup={r['case_id']:r['site'] for r in s['cases']}
    a,b,c=map(set,(s['train'],s['id_test'],s['ood_test']))
    assert not (a&b or a&c or b&c), 'overlapping subjects'
    assert a|b|c==set(lookup), 'missing subjects'
    assert not ({lookup[i] for i in a|b}&{lookup[i] for i in c}), 'OOD site leakage'
    assert {lookup[i] for i in b}<={lookup[i] for i in a}, 'ID without training centre'
    validation=[]
    for f in s['folds']:
        tr,va=set(f['train']),set(f['val'])
        assert tr and va and tr|va==a and not tr&va
        assert not ({lookup[i] for i in tr}&{lookup[i] for i in va}), 'fold site leakage'
        assert set(f['id_test'])<=b
        assert {lookup[i] for i in f['id_test']}<={lookup[i] for i in tr}
        validation.extend(va)
    assert Counter(validation)==Counter(a), 'validation coverage'


def export_dataset(splits, processed, destination):
    validate_splits(splits)
    if destination.exists(): raise ValueError('Destination existe déjà : choisir un nouveau dossier')
    lookup={r['case_id']:r for r in splits['cases']}
    all_ids=splits['train']+splits['id_test']+splits['ood_test']
    # Fail before creating anything if a full contractual export is unavailable.
    missing=[i for i in all_ids if not all((processed/lookup[i]['subject']/f).is_file() for f in ('qc.json','t1.nii.gz','mask.nii.gz'))]
    if missing: raise ValueError(f'Export incomplet refusé : {len(missing)} sujets sans sorties validées')
    staging=destination.with_name(destination.name+'.staging')
    if staging.exists(): raise ValueError('Dossier staging préexistant, à inspecter')
    staging.mkdir(parents=True)
    try:
        for folder in ('imagesTr','labelsTr','imagesTs','evaluation_labels/id','evaluation_labels/ood'):
            (staging/folder).mkdir(parents=True)
        for subset,ids in [('train',splits['train']),('id',splits['id_test']),('ood',splits['ood_test'])]:
            for case in ids:
                source=processed/lookup[case]['subject']
                info=json.loads((source/'qc.json').read_text())
                if info.get('version')!=VERSION: raise ValueError('Version de traitement incompatible')
                image, mask = nib.load(source/'t1.nii.gz'), nib.load(source/'mask.nii.gz')
                if (image.shape != mask.shape or not np.allclose(image.affine,mask.affine)
                    or nib.aff2axcodes(image.affine) != ('R','A','S')
                    or not np.allclose(image.header.get_zooms(),(1,1,1))):
                    raise ValueError(f'Géométrie export invalide: {case}')
                if not np.isfinite(image.get_fdata(dtype=np.float32)).all():
                    raise ValueError(f'IRM non finie: {case}')
                if not np.isin(mask.get_fdata(dtype=np.float32),[0,1]).all():
                    raise ValueError(f'Masque export non binaire: {case}')
                image_dir='imagesTr' if subset=='train' else 'imagesTs'
                mask_dir='labelsTr' if subset=='train' else f'evaluation_labels/{subset}'
                for src,dst in [(source/'t1.nii.gz',staging/image_dir/f'{case}_0000.nii.gz'),(source/'mask.nii.gz',staging/mask_dir/f'{case}.nii.gz')]:
                    # Hard links save space; source and export must be treated as immutable.
                    try: os.link(src,dst)
                    except OSError: shutil.copy2(src,dst)
        write_json(staging/'dataset.json',dict(channel_names={'0':'T1'},labels={'background':0,'lesion':1},numTraining=len(splits['train']),file_ending='.nii.gz'))
        write_json(staging/'splits_final.json',[dict(train=f['train'],val=f['val']) for f in splits['folds']])
        write_json(staging/'evaluation_split.json',dict(id_test=splits['id_test'],ood_test=splits['ood_test']))
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('run')
    p.add_argument('--root',type=Path,default=ROOT/'data/ATLAS3_Training_Raw')
    p.add_argument('--output',type=Path,default=ROOT/'data/processed')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--limit',type=int)
    p.add_argument('--sites',nargs='+')
    p.add_argument('--cases',type=Path,help='Liste JSON de case_id à traiter')
    p.add_argument('--margin',type=int,default=5)
    p.add_argument('--qc-only',action='store_true')
    p=sub.add_parser('splits')
    p.add_argument('--qc',type=Path,default=ROOT/'data/qc_s4/subjects.parquet')
    p.add_argument('--output',type=Path,default=ROOT/'splits.json')
    p.add_argument('--seed',type=int,default=2026)
    p=sub.add_parser('export')
    p.add_argument('--splits',type=Path,default=ROOT/'splits.json')
    p.add_argument('--processed',type=Path,default=ROOT/'data/processed')
    p.add_argument('--output',type=Path,default=ROOT/'data/nnUNet_raw/Dataset001_ATLAS')
    args=parser.parse_args()
    if args.command=='run':
        if args.workers<1 or args.margin<0 or (args.limit is not None and args.limit<1): parser.error('Arguments numériques invalides')
        run(args)
    elif args.command=='splits':
        if args.output.exists(): raise ValueError('Splits déjà figés : utiliser un autre --output pour une nouvelle proposition')
        write_json(args.output,make_splits(pd.read_parquet(args.qc),args.seed))
        print(args.output)
    else: export_dataset(json.loads(args.splits.read_text()),args.processed,args.output)


if __name__=='__main__': main()
