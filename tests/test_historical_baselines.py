import sys
from pathlib import Path
import unittest
import pandas as pd
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from historical_baselines import load_historical, temporal_association


class HistoricalTests(unittest.TestCase):
    def test_reject_training_and_role_reassignment(self):
        metrics=pd.DataFrame(dict(case=['a'],split=['OOD'],dice=[.5],model=['cnn'],gt_ml=[1]))
        for role in ('train','id_val'):
            partition=pd.DataFrame(dict(case=['a'],site=['A'],set=[role],days_post_stroke=[10]))
            with self.assertRaises(ValueError):load_historical(metrics,partition)
        partition['set']='ood_test'
        self.assertEqual(load_historical(metrics,partition)['set'].iloc[0],'ood_test')

    def test_missing_delays_remain_accounted_for(self):
        metrics=pd.DataFrame(dict(case=['a','b'],split=['OOD']*2,dice=[.5,.7],model=['cnn']*2,gt_ml=[1,0]))
        partition=pd.DataFrame(dict(case=['a','b'],site=['A']*2,set=['ood_test']*2,days_post_stroke=[-1,np.nan]))
        result=load_historical(metrics,partition)
        self.assertEqual(len(result),2)
        self.assertFalse(result.valid_delay.any())

    def test_conditional_time_signal_and_volume_sensitivity(self):
        rng=np.random.default_rng(4);n=60
        days=np.tile(np.arange(1,21),3)
        data=pd.DataFrame(dict(site=np.repeat(['A','B','C'],20),days_post_stroke=days,
                              dice=.2+.15*np.log1p(days)+rng.normal(0,.01,n),
                              gt_ml=rng.uniform(1,10,n),valid_delay=True))
        result=temporal_association(data,39)
        self.assertGreater(result['incremental_r2_time'],.9)
        self.assertEqual(result['permutation_p_time_given_center'],.025)
        self.assertIsNotNone(result['volume_sensitivity'])

    def test_missing_dice_not_imputed_but_infinity_rejected(self):
        metrics=pd.DataFrame(dict(case=['a'],split=['ID'],dice=[np.nan],model=['cnn'],gt_ml=[0]))
        partition=pd.DataFrame(dict(case=['a'],site=['A'],set=['id_val'],days_post_stroke=[10]))
        self.assertTrue(load_historical(metrics,partition).dice.isna().all())
        metrics['dice']=np.inf
        with self.assertRaises(ValueError):load_historical(metrics,partition)


if __name__=='__main__':unittest.main()
