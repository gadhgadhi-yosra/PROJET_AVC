"""Référence et seuil descriptif OOD individuel, calibrés sur validation seule."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from center_distance import center_distance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--histograms',type=Path,default=Path('data/processed/post_stats.parquet'))
    p.add_argument('--splits',type=Path,default=Path('splits.json'))
    p.add_argument('--fold',type=int,default=0)
    p.add_argument('--quantile',type=float,default=.95)
    p.add_argument('--output',type=Path,default=Path('data/demo/reference_ood.npz'))
    a=p.parse_args()
    if not .5<a.quantile<1: raise ValueError('Quantile attendu dans ]0.5,1[')
    s=json.loads(a.splits.read_text())
    if not 0 <= a.fold < len(s['folds']): raise ValueError('Pli inconnu')
    f=s['folds'][a.fold];h=pd.read_parquet(a.histograms)
    if h.case_id.duplicated().any(): raise ValueError('Histogrammes dupliqués')
    if not (set(f['train'])|set(f['val']))<=set(h.case_id): raise ValueError('Histogrammes train/validation incomplets')
    ref=np.stack(h.loc[h.case_id.isin(f['train']),'hist'])
    val=np.stack(h.loc[h.case_id.isin(f['val']),'hist'])
    distances=np.array([center_distance(scan[None,:],ref) for scan in val])
    threshold=float(np.quantile(distances,a.quantile,method='higher'))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,histograms=ref,threshold=threshold,fold=a.fold,
                        quantile=a.quantile,validation_n=len(val),kind='individual_histogram_wasserstein')
    a.output.with_suffix('.json').write_text(json.dumps(dict(fold=a.fold,quantile=a.quantile,
        threshold=threshold,train_scans=len(ref),validation_scans=len(val),
        scope='Atypie d’intensité individuelle ; ne prédit pas un échec clinique ou une baisse de Dice',
        calibration='Quantile des distances sur validation, test ID/OOD jamais consulté'),indent=2,ensure_ascii=False))


if __name__=='__main__':main()
