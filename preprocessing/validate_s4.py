"""Relit les sorties S4 et contrôle les invariants, indépendamment du calcul."""
import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from week4 import write_json


def validate(root):
    records=[]
    markers=sorted(root.glob('*/qc.json'))
    if not markers:raise ValueError('Aucune sortie validable')
    for marker in markers:
        folder=marker.parent
        info=json.loads(marker.read_text())
        a,b,c=[nib.load(folder/f'{name}.nii.gz') for name in ('t1','mask','brain')]
        if not (a.shape==b.shape==c.shape):raise ValueError('Shapes differ')
        if not (np.allclose(a.affine,b.affine) and np.allclose(a.affine,c.affine)):raise ValueError('Affines differ')
        if nib.aff2axcodes(a.affine)!=('R','A','S') or not np.allclose(a.header.get_zooms(),1):raise ValueError('Not RAS 1 mm')
        x,y,brain=a.get_fdata(dtype=np.float32),b.get_fdata(dtype=np.float32),c.get_fdata(dtype=np.float32)
        if not np.isfinite(x).all() or not np.isin(y,[0,1]).all() or not np.isin(brain,[0,1]).all():raise ValueError('Invalid voxel data')
        support=brain>0
        mean,std=float(x[support].mean()),float(x[support].std())
        if abs(mean)>1e-4 or abs(std-1)>1e-4 or np.any(x[~support]!=0):raise ValueError('Z-score/fond incorrect')
        if not np.isclose(y.sum()/1000,info['lesion_ml']):raise ValueError('QC volume differs')
        records.append(dict(mean=mean,std=std))
    return dict(subjects=len(records),status='passed',checks=['finite','binary_masks','RAS','1mm','shared_grid','zero_background','brain_zscore','lesion_volume'],
                max_abs_brain_mean=max(abs(r['mean']) for r in records),max_abs_std_error=max(abs(r['std']-1) for r in records))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--report',type=Path,default=Path('preprocessing/reports/validation_s4.json'))
    a=p.parse_args();result=validate(a.root);write_json(a.report,result);print(json.dumps(result,indent=2))
