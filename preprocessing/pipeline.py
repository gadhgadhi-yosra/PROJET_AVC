"""Semaine 3 : inventaire Spark et géométrie RAS à 1 mm (sans normalisation)."""
import argparse
import csv
import json
import itertools
import os
from pathlib import Path
import shutil
import sys

import nibabel as nib
from nibabel.processing import resample_to_output, resample_from_to
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def inventory(root):
    rows = []
    for anat in sorted(root.glob('*/sub-*/ses-*/anat')):
        row = dict(site=anat.parts[-4], subject=anat.parts[-3], session=anat.parts[-2])
        problems = []
        for key, pattern in [('t1_path', '*_desc-brain_T1w.nii.gz'),
                             ('mask_path', '*_label-lesion_*mask.nii.gz'),
                             ('metadata_path', '*_metadata.csv')]:
            files = sorted(anat.glob(pattern))
            row[key] = str(files[0].resolve()) if len(files) == 1 else None
            if len(files) != 1:
                problems.append(f'{key}: {len(files)} fichiers')
        row['days_post_stroke'] = None
        row['chronicity'] = None
        if row['metadata_path']:
            try:
                with open(row['metadata_path'], newline='', encoding='utf-8-sig') as handle:
                    records = list(csv.DictReader(handle))
                if not records:
                    problems.append('metadata_empty')
                else:
                    meta = records[0]
                    if meta.get('SITE') and meta['SITE'] != row['site']:
                        problems.append('metadata_site_mismatch')
                    value = meta.get('DAYS_POST_STROKE', '')
                    row['days_post_stroke'] = float(value) if value and value.lower() not in ('nan', 'n/a', 'na') else None
                    # The CSVs encode the flag as '1' or '1.0'; empty means not flagged.
                    flag = (meta.get('CHRONICITY') or '').strip()
                    row['chronicity'] = True if flag and float(flag) == 1 else None
            except (ValueError, OSError, csv.Error) as exc:
                problems.append(f'metadata_error: {exc}')
        row['inventory_warnings'] = '; '.join(problems)
        rows.append(row)
    return rows


def geometry(t1_path, mask_path, output):
    image, mask = nib.load(t1_path), nib.load(mask_path)
    if len(image.shape) != 3 or len(mask.shape) != 3:
        raise ValueError('Volumes 3D requis')
    if image.shape != mask.shape or not np.allclose(image.affine, mask.affine, atol=1e-3):
        raise ValueError('Image et masque sur des grilles différentes')
    image = nib.as_closest_canonical(image)
    mask = nib.as_closest_canonical(mask)
    values = image.get_fdata(dtype=np.float32)
    labels = mask.get_fdata(dtype=np.float32)
    if not np.isfinite(values).all() or not np.isfinite(labels).all():
        raise ValueError('Valeurs non finies')
    # Linear interpolation for intensities, nearest neighbour for binary labels.
    source = nib.Nifti1Image(values, image.affine)
    binary = nib.Nifti1Image((labels > .5).astype(np.uint8), mask.affine)
    result = resample_to_output(source, voxel_sizes=(1, 1, 1), order=1)
    segmentation = resample_from_to(binary, (result.shape, result.affine), order=0)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Atomic replacement: Spark retries never expose half-written NIfTI files.
    for name, volume in [('t1', result), ('mask', segmentation)]:
        temp = output / f'.{name}.{os.getpid()}.nii.gz'
        nib.save(volume, temp)
        temp.replace(output / f'{name}.nii.gz')
    return dict(shape=list(result.shape), orientation=''.join(nib.aff2axcodes(result.affine)),
                spacing=[float(x) for x in result.header.get_zooms()],
                lesion_voxels=int(np.count_nonzero(segmentation.get_fdata())))


def done(output):
    return (Path(output) / 't1.nii.gz').is_file() and (Path(output) / 'mask.nii.gz').is_file()


def process(row, destination):
    row = dict(row)
    row.update(status='ok', error='', output_dir='')
    if row['site'] == 'R032':
        row.update(status='excluded', error='R032 : espacement suspect, décision QC avant traitement')
        return row
    try:
        if not row['t1_path'] or not row['mask_path']:
            raise ValueError('Paire image/masque manquante ou ambiguë')
        output = Path(destination) / row['site'] / row['subject'] / row['session']
        # Resume: files are only ever written atomically, so presence means complete.
        if not done(output):
            geometry(row['t1_path'], row['mask_path'], output)
        row['output_dir'] = str(output)
    except Exception as exc:
        row.update(status='error', error=f'{type(exc).__name__}: {exc}')
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'data/ATLAS3_Training_Raw')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/preprocessing_s3')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--sites', nargs='+')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--inventory-only', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error('workers et limit doivent être positifs')
    rows = inventory(args.root)
    if not rows:
        parser.error('Aucune session trouvée')
    selected = [r for r in rows if not args.sites or r['site'] in args.sites]
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        parser.error('Sélection vide')
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.inventory_only:
        estimated = 0
        for row in selected:
            if row['site'] == 'R032' or not row['t1_path'] or not row['mask_path']:
                continue
            if done(args.output / 'volumes' / row['site'] / row['subject'] / row['session']):
                continue
            img = nib.load(row['t1_path'])
            corners = np.array(list(itertools.product(*[(0, n-1) for n in img.shape[:3]])))
            world = nib.affines.apply_affine(img.affine, corners)
            voxel_ratio = np.prod(np.ceil(np.ptp(world, axis=0)) + 1) / np.prod(img.shape[:3])
            # Outputs are compressed too: measured ~1.45x the input .nii.gz size at equal
            # voxel count (int16 -> float32), so 2x the scaled input size is a safe margin.
            input_size = os.path.getsize(row['t1_path']) + os.path.getsize(row['mask_path'])
            estimated += int(2 * input_size * voxel_ratio)
        # Retain 2 GiB for the OS and Spark.
        if estimated + 2 * 1024**3 > shutil.disk_usage(args.output).free:
            parser.error(f'Espace insuffisant : prévoir {estimated / 1024**3 + 2:.1f} Gio libres ou réduire --limit')
    if not args.inventory_only and shutil.disk_usage(args.output).free < 2 * 1024**3:
        parser.error('Moins de 2 Gio libres : libérer de l’espace avant le traitement')
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ.setdefault('SPARK_LOCAL_IP', '127.0.0.1')
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, BooleanType
    types = dict(days_post_stroke=DoubleType(), chronicity=BooleanType())
    schema = StructType([StructField(k, types.get(k, StringType()), True) for k in rows[0]])
    spark = SparkSession.builder.master(f'local[{args.workers}]').appName('ATLAS-S3').config('spark.ui.enabled', 'false').getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')
    spark.sparkContext.addPyFile(str(Path(__file__).resolve()))
    try:
        spark.createDataFrame(rows, schema).write.mode('overwrite').parquet(str(args.output / 'subjects.parquet'))
        summary = dict(subjects=len(rows), sites=len({r['site'] for r in rows}),
                       metadata_warnings=sum(bool(r['inventory_warnings']) for r in rows))
        if not args.inventory_only:
            destination = str(args.output.resolve() / 'volumes')
            # Import the shipped module on executors, also works with spawned workers.
            def worker(row):
                from pipeline import process
                return process(row, destination)
            # One partition per subject: Spark hands subjects to idle cores as they finish,
            # so a few heavy volumes (e.g. 512x512 scans) do not stall a whole core's batch.
            results = spark.sparkContext.parallelize(selected, len(selected)).map(worker).collect()
            result_schema = StructType(schema.fields + [StructField(k, StringType(), True) for k in ('status', 'error', 'output_dir')])
            spark.createDataFrame(results, result_schema).write.mode('overwrite').parquet(str(args.output / 'geometry_results.parquet'))
            summary.update(selected=len(results), ok=sum(r['status']=='ok' for r in results),
                           excluded=sum(r['status']=='excluded' for r in results), errors=sum(r['status']=='error' for r in results))
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2))
        if summary.get('errors'):
            raise SystemExit(1)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
