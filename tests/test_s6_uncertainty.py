from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from site_fingerprint_uncertainty import stratified_paired_bootstrap


class UncertaintyTests(unittest.TestCase):
    def test_equal_center_weights_despite_unequal_sizes(self):
        scores=np.array([[1],[1],[1],[0]])
        point,draws=stratified_paired_bootstrap(scores,['a','a','a','b'],200)
        self.assertAlmostEqual(point[0],.5)
        np.testing.assert_allclose(draws,.5)

    def test_pairing_preserves_identical_predictions(self):
        scores=np.array([[1,1],[0,0],[1,1],[0,0]])
        _,draws=stratified_paired_bootstrap(scores,['a','a','b','b'],500)
        np.testing.assert_array_equal(draws[:,0]-draws[:,1],0)

    def test_reproducible_and_bounded(self):
        scores=np.array([[1,0],[0,1],[1,0],[1,1]])
        one=stratified_paired_bootstrap(scores,['a','a','b','b'],500,7)
        two=stratified_paired_bootstrap(scores,['a','a','b','b'],500,7)
        np.testing.assert_array_equal(one[1],two[1])
        self.assertTrue(((one[1]>=0)&(one[1]<=1)).all())
        self.assertGreater(one[1].std(),0)

    def test_invalid_inputs_rejected(self):
        for scores,sites in (([[1]],[]),([[np.nan]],['a']),([[2]],['a'])):
            with self.assertRaises(ValueError):stratified_paired_bootstrap(scores,sites)


if __name__=='__main__':unittest.main()
