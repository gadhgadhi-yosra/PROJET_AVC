import json
from pathlib import Path
import sys
import tempfile
import unittest
import nibabel as nib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
sys.path.insert(0,str(ROOT/'demo'))
from failure_prediction import analyze, fit_alert
from temporal_shift import fit_effects
from network_features import center_distance as feature_distance, load_bundle
from result_contract import load_results
from core import summarize, aligned, load_volume


class AnalysisTests(unittest.TestCase):
    def test_ood_scores_do_not_change_calibration(self):
        table=pd.DataFrame(dict(site=list('ABCDEF'),role=['validation']*5+['ood_test'],
                                distance=[.1,.2,.3,.4,.5,1.],dice=[.9,.8,.7,.4,.3,.1]))
        _,loco,a=analyze(table)
        table.loc[5,'dice']=.99
        _,loco2,b=analyze(table)
        self.assertEqual({k:v for k,v in a.items() if k!='held_out_metrics'},
                         {k:v for k,v in b.items() if k!='held_out_metrics'})
        pd.testing.assert_frame_equal(loco,loco2)
        changed=table.copy();changed.loc[0,'dice']=0
        _,loco3,_=analyze(changed)
        self.assertEqual(loco.iloc[0].predicted_loss,loco3.iloc[0].predicted_loss)
        self.assertEqual(loco.iloc[0].threshold,loco3.iloc[0].threshold)
    def test_alert_requires_both_classes(self):
        self.assertIsNone(fit_alert([1,2,3],[False]*3))
        self.assertEqual(fit_alert([1,2,3,4],[False,False,True,True]),2.5)
    def test_feature_distance_detects_shift(self):
        ref=np.random.default_rng(1).normal(size=(30,8))
        self.assertAlmostEqual(feature_distance(ref,ref),0,places=7)
        self.assertGreater(feature_distance(ref+5,ref),.5)
        with self.assertRaises(ValueError):feature_distance(ref,np.ones((4,8)))
    def test_time_effect_conditioned_on_center(self):
        rng=np.random.default_rng(4);n=100
        days=np.tile(np.linspace(1,100,n),2)
        data=pd.DataFrame(dict(site=['A']*n+['B']*n,days_post_stroke=days,
            dice=np.repeat([.3,.7],n)+.02*np.log1p(days)+rng.normal(0,.002,2*n)))
        result=fit_effects(data)
        self.assertAlmostEqual(result['coefficient_log1p_days'],.02,places=3)
        self.assertGreater(result['incremental_r2_center'],result['incremental_r2_time'])
    def test_c3_rejects_training_cases_and_wrong_site(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);s=dict(folds=[dict(train=['a'],val=['b'],id_test=['c'])],ood_test=['d'],
                cases=[dict(case_id=k,site=k.upper()) for k in 'abcd'])
            (p/'splits.json').write_text(json.dumps(s))
            def check(case,site):
                pd.DataFrame([dict(model='m',technique='t',fold=0,case_id=case,dice=.5,site=site)]).to_csv(p/'results.csv',index=False)
                return load_results(p/'results.csv',p/'splits.json','m','t',0)
            with self.assertRaisesRegex(ValueError,'entraînement'):check('a','A')
            with self.assertRaisesRegex(ValueError,'Centre'):check('b','A')
            self.assertEqual(check('b','B')[0].iloc[0].role,'validation')


class DemoTests(unittest.TestCase):
    def test_volume_and_empty_mask_uncertainty(self):
        img=nib.Nifti1Image(np.zeros((3,4,5),np.float32),np.diag([2.,3.,4.,1.]))
        mask=np.zeros(img.shape,bool);mask[0]=True
        result=summarize(img,mask)
        self.assertAlmostEqual(result['lesion_volume_ml'],20*.024)
        empty=summarize(img,np.zeros(img.shape,bool),np.ones(img.shape))
        self.assertIsNone(empty['uncertainty']['mean_lesion'])
    def test_wrong_grid_and_probability_masks_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'mask.nii.gz'
            image=nib.Nifti1Image(np.zeros((3,3,3),np.float32),np.eye(4))
            nib.save(nib.Nifti1Image(np.zeros(image.shape),np.diag([2.,1.,1.,1.])),p)
            with self.assertRaisesRegex(ValueError,'grille'):aligned(p,image,True)
            nib.save(nib.Nifti1Image(np.full(image.shape,.3),np.eye(4)),p)
            with self.assertRaisesRegex(ValueError,'binaire'):aligned(p,image,True)


if __name__=='__main__':unittest.main()
