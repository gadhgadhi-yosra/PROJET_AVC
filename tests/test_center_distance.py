from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from center_distance import BIN_EDGES, BIN_WIDTH, brain_histogram, center_distance


def gaussians(means,seed=0,size=20000):
    rng=np.random.default_rng(seed)
    return np.stack([brain_histogram(rng.normal(m,1,size),np.ones(size)) for m in means])


class CenterDistanceTests(unittest.TestCase):
    def setUp(self):
        self.ref=gaussians(np.linspace(-.1,.1,12))
    def test_histogram_ignores_outside_brain_and_clips(self):
        x=np.array([0.,99.,-99.,1000.]);brain=np.array([1,1,1,0])
        h=brain_histogram(x,brain)
        self.assertAlmostEqual(h.sum(),1)
        self.assertAlmostEqual(h[0],1/3);self.assertAlmostEqual(h[-1],1/3)
        with self.assertRaisesRegex(ValueError,'empty_brain'):brain_histogram(x,np.zeros(4))
    def test_wasserstein_matches_a_known_shift(self):
        self.assertAlmostEqual(center_distance(gaussians([1.5],seed=1),self.ref),1.5,delta=2*BIN_WIDTH)
    def test_identical_sets_are_at_zero(self):
        for metric in ('wasserstein','mmd'):
            self.assertAlmostEqual(center_distance(self.ref,self.ref,metric),0,places=7)
    def test_distance_grows_with_shift_and_returns_float(self):
        for metric in ('wasserstein','mmd'):
            near=center_distance(gaussians([.3]*6,seed=1),self.ref,metric)
            far=center_distance(gaussians([1.5]*6,seed=2),self.ref,metric)
            self.assertIsInstance(far,float);self.assertGreater(far,near)
    def test_single_scan_accepted(self):
        self.assertGreater(center_distance(self.ref[0]*0+gaussians([1])[0],self.ref),0)
    def test_invalid_inputs_rejected(self):
        with self.assertRaisesRegex(ValueError,'invalid_histogram'):center_distance(self.ref*2,self.ref)
        with self.assertRaisesRegex(ValueError,'expected_histograms'):center_distance(np.ones((2,7))/7,self.ref)
        with self.assertRaisesRegex(ValueError,'unknown_metric'):center_distance(self.ref,self.ref,'cosine')
        with self.assertRaisesRegex(ValueError,'mmd_requires_distinct'):center_distance(self.ref,self.ref[:1],'mmd')


if __name__=='__main__':
    unittest.main()
