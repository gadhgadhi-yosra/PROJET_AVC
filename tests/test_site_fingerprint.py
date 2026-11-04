from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from site_fingerprint import rebin


class RebinTests(unittest.TestCase):
    def test_sums_groups_and_preserves_mass(self):
        h=np.arange(1,151,dtype=float)[None,:];h/=h.sum()
        out=rebin(h,5)
        self.assertEqual(out.shape,(1,30))
        self.assertAlmostEqual(out.sum(),1)
        self.assertAlmostEqual(out[0,0],h[0,:5].sum())


if __name__=='__main__':
    unittest.main()
