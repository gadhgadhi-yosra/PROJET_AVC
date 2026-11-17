import copy
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'preprocessing'))
from week4 import make_splits,validate_splits
from finalize_release import revise_split


class ReleaseTests(unittest.TestCase):
    def test_quarantine_preserves_all_other_assignments(self):
        rows=[]
        for site in range(15):
            hist=np.zeros(32);hist[site]=1
            for index in range(8):
                case=f'R{site}_{index}'
                rows.append(dict(case_id=case,subject=case,session='ses-1',site=f'R{site}',
                                 status='ok',error='',hist=hist.tolist()))
        before=make_splits(pd.DataFrame(rows));saved=copy.deepcopy(before)
        excluded=before['train'][0];after=revise_split(before,{excluded:'qc'})
        validate_splits(after)
        self.assertEqual(before,saved)
        self.assertEqual(before['ood_centers'],after['ood_centers'])
        for old,new in zip(before['folds'],after['folds']):
            for key in ('train','val','id_test'):
                self.assertEqual(set(old[key])-{excluded},set(new[key]))
        self.assertEqual(len(after['cases']),len(before['cases'])-1)
        with self.assertRaises(ValueError):revise_split(before,{'unknown':'qc'})


if __name__=='__main__':unittest.main()
