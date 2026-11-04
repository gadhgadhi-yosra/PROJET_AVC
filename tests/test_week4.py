import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'preprocessing'))
from week4 import VERSION, transform, work, make_splits, validate_splits, export_dataset


class Week4Tests(unittest.TestCase):
    def setUp(self):
        self.disk_patch=patch('week4.shutil.disk_usage',return_value=SimpleNamespace(free=100*1024**3))
        self.disk_patch.start();self.addCleanup(self.disk_patch.stop)
        self.tmp=tempfile.TemporaryDirectory(); self.p=Path(self.tmp.name)
        self.row=dict(site='R029',subject='sub-1',session='ses-1',t1_path=str(self.p/'raw.nii.gz'),mask_path=str(self.p/'label.nii.gz'),days_post_stroke=None,inventory_warnings='')
        x=np.full((20,22,24),.0007,dtype=np.float32)
        x[5:15,6:16,7:17]=np.random.default_rng(0).uniform(2,20,(10,10,10))
        y=np.zeros_like(x); y[8:10,9:11,10:12]=1
        self.aff=np.diag([-2.,1.,1.,1.]);self.aff[0,3]=40
        self.save(x,y)
    def tearDown(self): self.tmp.cleanup()
    def save(self,x,y):
        nib.save(nib.Nifti1Image(x,self.aff),self.row['t1_path'])
        nib.save(nib.Nifti1Image(y,self.aff),self.row['mask_path'])
    def test_normalization_crop_alignment_resume(self):
        info=transform(self.row,self.p,margin=2)
        out=self.p/'sub-1'
        a,b,c=[nib.load(out/f'{n}.nii.gz') for n in ('t1','mask','brain')]
        x=a.get_fdata(); support=c.get_fdata()>0
        self.assertEqual(a.get_data_dtype(),np.int16)
        self.assertAlmostEqual(float(a.dataobj.slope),1e-3)
        self.assertAlmostEqual(x[support].mean(),0,places=3)
        self.assertAlmostEqual(x[support].std(),1,places=3)
        self.assertTrue((x[~support]==0).all())
        self.assertEqual(nib.aff2axcodes(a.affine),('R','A','S'))
        np.testing.assert_allclose(a.header.get_zooms(),1)
        np.testing.assert_allclose(a.affine,b.affine)
        self.assertEqual(a.shape,b.shape)
        self.assertLess(a.shape[1],22)
        world=self.aff@np.array([8,9,10,1])
        ijk=np.rint(np.linalg.inv(b.affine)@world).astype(int)[:3]
        self.assertEqual(b.get_fdata()[tuple(ijk)],1)
        with patch('week4.load_pair',side_effect=AssertionError('recomputed')):
            self.assertEqual(transform(self.row,self.p,margin=2),info)
        (out/'qc.json').unlink()
        with patch('week4.load_pair',side_effect=ValueError('recompute_required')):
            with self.assertRaisesRegex(ValueError,'recompute_required'): transform(self.row,self.p,margin=2)
    def test_empty_masks_retained_and_corruption_reported(self):
        x=nib.load(self.row['t1_path']).get_fdata(dtype=np.float32)
        self.save(x,np.zeros_like(x))
        self.assertTrue(work(self.row,self.p,qc_only=True)['empty_mask'])
        Path(self.row['t1_path']).write_bytes(b'corrupt')
        self.assertEqual(work(self.row,self.p,qc_only=True)['status'],'error')
    def test_no_label_driven_crop(self):
        x=nib.load(self.row['t1_path']).get_fdata(dtype=np.float32)
        y=np.zeros_like(x);y[0,0,0]=1;self.save(x,y)
        with self.assertRaisesRegex(ValueError,'lesion_outside_image_based_crop'): transform(self.row,self.p,margin=0)
    def test_failed_recompute_invalidates_old_marker(self):
        transform(self.row,self.p,margin=2)
        Path(self.row['t1_path']).write_bytes(b'corrupt')
        self.assertEqual(work(self.row,self.p)['status'],'error')
        self.assertFalse((self.p/'sub-1/qc.json').exists())

    def test_small_lesion_cannot_disappear_silently(self):
        x=nib.load(self.row['t1_path']).get_fdata(dtype=np.float32)
        y=np.zeros_like(x);y[8,8,8]=1
        self.aff=np.diag([.1,.1,.1,1.])
        self.save(x,y)
        with self.assertRaisesRegex(ValueError,'lesion_lost_in_resampling'):
            transform(self.row,self.p,margin=2)

    def test_low_disk_refuses_new_output(self):
        with patch('week4.shutil.disk_usage',return_value=SimpleNamespace(free=1024)):
            with self.assertRaisesRegex(RuntimeError,'disk_space'):
                transform(self.row,self.p)
        self.assertFalse((self.p/'sub-1/qc.json').exists())

    def test_qc_cache_reuses_only_unchanged_inputs(self):
        self.assertEqual(work(self.row,self.p,qc_only=True)['status'],'ok')
        with patch('week4.load_pair',side_effect=ValueError('recomputed')):
            self.assertEqual(work(self.row,self.p,qc_only=True)['status'],'ok')
            changed=dict(self.row,days_post_stroke=42)
            self.assertEqual(work(changed,self.p,qc_only=True)['status'],'error')

    def test_reject_nonfinite(self):
        x=nib.load(self.row['t1_path']).get_fdata(dtype=np.float32);x[8,8,8]=np.nan
        self.save(x,np.zeros_like(x))
        self.assertIn('nonfinite',work(self.row,self.p,qc_only=True)['error'])


class SplitTests(unittest.TestCase):
    def frame(self):
        rows=[]
        for site in range(15):
            h=np.zeros(32);h[site]=1
            for patient in range(5+site):
                case=f'R{site:03}_s{patient}'
                rows.append(dict(case_id=case,subject=case,session='ses-1',site=f'R{site:03}',status='ok',error='',hist=h.tolist()))
        return pd.DataFrame(rows)
    def test_reproducible_grouped_splits(self):
        frame=self.frame();a=make_splits(frame);b=make_splits(frame.sample(frac=1,random_state=4))
        self.assertEqual(a,b);validate_splits(a)
        self.assertEqual(len(a['ood_centers']),6)
        a['folds'][0]['train'].append(a['ood_test'][0])
        with self.assertRaises(AssertionError):validate_splits(a)
    def test_export_rejects_missing_and_valid_layout(self):
        s=make_splits(self.frame())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);processed=root/'processed';processed.mkdir()
            with self.assertRaisesRegex(ValueError,'incomplet'):export_dataset(s,processed,root/'missing')
            self.assertFalse((root/'missing').exists())
            for case in s['cases']:
                d=processed/case['subject'];d.mkdir()
                (d/'qc.json').write_text(json.dumps(dict(version=VERSION)))
                for name in ('t1','mask'):nib.save(nib.Nifti1Image(np.zeros((2,2,2),np.uint8),np.eye(4)),d/f'{name}.nii.gz')
            out=root/'Dataset001_ATLAS';export_dataset(s,processed,out)
            self.assertEqual(len(list((out/'imagesTr').glob('*'))),len(s['train']))
            self.assertEqual(len(list((out/'imagesTs').glob('*'))),len(s['id_test'])+len(s['ood_test']))
            self.assertFalse((out/'labelsTr'/f"{s['ood_test'][0]}.nii.gz").exists())
            self.assertEqual(json.loads((out/'dataset.json').read_text())['numTraining'],len(s['train']))

if __name__=='__main__':unittest.main()
