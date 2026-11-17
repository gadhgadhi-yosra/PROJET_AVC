from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from qc_by_center import summarize


class CenterQCTests(unittest.TestCase):
    def data(self):
        return pd.DataFrame([
            dict(case_id='a',site='A',status='ok',empty_mask=True,days_post_stroke=12,chronicity=None,inventory_warnings='',error=''),
            dict(case_id='b',site='A',status='ok',empty_mask=None,days_post_stroke=-2,chronicity=True,inventory_warnings='metadata_empty',error=''),
            dict(case_id='c',site='A',status='excluded',empty_mask=None,days_post_stroke=None,chronicity=None,inventory_warnings='',error='QC'),
            dict(case_id='d',site='B',status='excluded',empty_mask=None,days_post_stroke=None,chronicity=None,inventory_warnings='',error='spacing'),
        ])

    def test_denominators_exclude_quarantined_subjects(self):
        table=summarize(self.data())
        self.assertEqual(table.loc['A','inventoried'],3)
        self.assertEqual(table.loc['A','delay_missing_or_invalid_percent'],50)
        self.assertEqual(table.loc['A','empty_masks'],1)
        self.assertEqual(table.loc['A','mask_status_unknown'],1)

    def test_excluded_center_rate_not_zero(self):
        table=summarize(self.data())
        self.assertEqual(table.loc['B','retained'],0)
        self.assertTrue(np.isnan(table.loc['B','delay_missing_or_invalid_percent']))

    def test_duplicate_and_unknown_status_rejected(self):
        frame=self.data();frame.loc[1,'case_id']='a'
        with self.assertRaises(ValueError):summarize(frame)
        frame=self.data();frame.loc[0,'status']='unknown'
        with self.assertRaises(ValueError):summarize(frame)

    def test_nonfinite_missing_and_text_delays(self):
        for value in (np.inf,np.nan,'bad'):
            frame=self.data();frame['days_post_stroke']=frame.days_post_stroke.astype(object);frame.loc[0,'days_post_stroke']=value
            self.assertEqual(summarize(frame).loc['A','delay_missing_or_invalid'],2)


if __name__=='__main__':unittest.main()
