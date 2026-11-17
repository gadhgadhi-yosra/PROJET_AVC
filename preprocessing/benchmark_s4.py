"""Benchmark réel S4 sur un sous-ensemble fixe, sorties fraîches à chaque essai."""
import argparse
import json
import os
import shutil
from pathlib import Path
import tempfile
import time

os.environ.setdefault('MPLCONFIGDIR','/tmp/avc-mpl')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'):
    os.environ[key]='1'
import nibabel as nib
import numpy as np
import pandas as pd
from pipeline import inventory, ROOT
from week4 import spark_run, write_json, case_id


def select(rows, count):
    selected=[]
    # Heterogeneous centres including R029 and SOOP, bounded raw size for 8 workers.
    sites=['R001','R005','R009','R014','R029','R041','R050','SOOP']
    for site in sites:
        for row in rows:
            if row['site']!=site or not row['t1_path'] or not row['mask_path']:continue
            if np.prod(nib.load(row['t1_path']).shape) <= 20_000_000:
                selected.append(row);break
    if count>len(selected):raise ValueError('Not enough representative cases')
    return selected[:count]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'data/benchmark_s4')
    p.add_argument('--report',type=Path,default=ROOT/'preprocessing/reports')
    p.add_argument('--count',type=int,default=8)
    p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--workers',nargs='+',type=int,default=[1,2,4,8])
    a=p.parse_args()
    if a.count<1 or a.repeats<1 or any(w<1 for w in a.workers):p.error('Positive values required')
    a.output.mkdir(parents=True,exist_ok=True);a.report.mkdir(parents=True,exist_ok=True)
    sample=select(inventory(ROOT/'data/ATLAS3_Training_Raw'),a.count)
    write_json(a.output/'sample.json',[dict(case_id=case_id(r),site=r['site']) for r in sample])
    reserve=512*1024**2
    estimated=sum(Path(r['t1_path']).stat().st_size*2 for r in sample)
    if shutil.disk_usage(a.output).free < estimated+reserve:
        raise RuntimeError('Espace insuffisant même pour le benchmark temporaire')
    results=[]
    for repeat in range(a.repeats):
        # Fixed seeded order avoids always measuring 1 core on a cold file cache.
        order=list(np.random.default_rng(2026+repeat).permutation(a.workers))
        for workers in order:
            started=time.perf_counter()
            with tempfile.TemporaryDirectory(prefix='trial-',dir=a.output) as tmp:
                records,seconds=spark_run(sample,tmp,int(workers),resume=False,reserve_bytes=reserve)
                errors=[r for r in records if r['status']!='ok']
                if errors:raise RuntimeError(str([(r['case_id'],r['error']) for r in errors]))
            results.append(dict(workers=int(workers),repeat=repeat,subjects=len(sample),compute_seconds=seconds,total_seconds=time.perf_counter()-started))
            pd.DataFrame(results).to_csv(a.report/'benchmark_trials.csv',index=False)
            print(results[-1],flush=True)
    frame=pd.DataFrame(results).groupby('workers',as_index=False).compute_seconds.median()
    if 1 not in frame.workers.values:raise ValueError('1-worker reference required')
    reference=float(frame.loc[frame.workers==1,'compute_seconds'].iloc[0])
    frame['speedup']=reference/frame.compute_seconds
    frame['efficiency']=frame.speedup/frame.workers
    frame.to_csv(a.report/'benchmark_summary.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,4))
    ax.plot(frame.workers,frame.speedup,'o-',label=f'Mesuré, {len(sample)} sujets')
    ax.plot(frame.workers,frame.workers,'--',color='gray',label='Idéal')
    ax.set(xlabel='Workers Spark locaux',ylabel='Accélération vs 1 worker',title='S4 : rééchantillonnage + normalisation + recadrage')
    ax.set_xticks(frame.workers);ax.legend();fig.tight_layout()
    fig.savefig(a.report/'benchmark_speedup.png',dpi=150);plt.close(fig)
    write_json(a.report/'benchmark_scope.json',dict(subjects=len(sample),repeats=a.repeats,seed=2026,
              scope='Sous-ensemble de validation, pas le dataset complet', reserve_MiB=512,
              timing='compute_seconds: map/collect incluant lecture, calcul et écriture; JVM/Spark startup exclu. total_seconds inclut startup et nettoyage.',
              cache='Cache système non purgé; ordre mélangé déterministe; aucune reprise des volumes',
              concurrency='Workers Spark, pas des cœurs physiques dédiés'))

if __name__=='__main__':main()
