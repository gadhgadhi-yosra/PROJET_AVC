"""Tests de la géométrie RAS / 1 mm sur des volumes synthétiques (aucune donnée patient)."""
from pathlib import Path
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'preprocessing'))
from pipeline import geometry, inventory, process  # noqa: E402


def save(path, data, affine):
    nib.save(nib.Nifti1Image(data, affine), path)
    return str(path)


class GeometryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        # Orientation LPS et voxels anisotropes 2 x 1 x 1 mm.
        self.affine = np.diag([-2.0, -1.0, 1.0, 1.0])
        self.image = np.random.default_rng(0).random((10, 20, 20), dtype=np.float32) * 100
        self.mask = np.zeros((10, 20, 20), dtype=np.float32)
        self.mask[3:6, 5:10, 5:10] = 0.99999998  # valeurs flottantes vues dans ATLAS

    def tearDown(self):
        self.tmp.cleanup()

    def test_ras_isotrope_et_masque_binaire(self):
        t1 = save(self.dir / 't1.nii.gz', self.image, self.affine)
        mask = save(self.dir / 'mask.nii.gz', self.mask, self.affine)
        info = geometry(t1, mask, self.dir / 'out')
        self.assertEqual(info['orientation'], 'RAS')
        self.assertEqual(info['spacing'], [1.0, 1.0, 1.0])
        out_t1 = nib.load(self.dir / 'out/t1.nii.gz')
        out_mask = nib.load(self.dir / 'out/mask.nii.gz')
        self.assertEqual(out_t1.shape, out_mask.shape)
        self.assertTrue(np.allclose(out_t1.affine, out_mask.affine))
        values = np.unique(np.asarray(out_mask.dataobj))
        self.assertTrue(set(values.tolist()) <= {0, 1})
        # Volume de lésion conservé en mm³ (3 x 5 x 5 voxels de 2 mm³ = 150 mm³).
        self.assertAlmostEqual(info['lesion_voxels'], 150, delta=150 * 0.2)

    def test_grilles_differentes_refusees(self):
        t1 = save(self.dir / 't1.nii.gz', self.image, self.affine)
        mask = save(self.dir / 'mask.nii.gz', self.mask, np.diag([-1.0, -1.0, 1.0, 1.0]))
        with self.assertRaises(ValueError):
            geometry(t1, mask, self.dir / 'out')

    def test_r032_exclu(self):
        row = dict(site='R032', subject='sub-x', session='ses-1', t1_path='a', mask_path='b')
        self.assertEqual(process(row, self.dir)['status'], 'excluded')

    def test_reprise_sans_recalcul(self):
        output = self.dir / 'R001/sub-x/ses-1'
        output.mkdir(parents=True)
        for name in ('t1', 'mask'):
            (output / f'{name}.nii.gz').write_bytes(b'deja fait')
        # Chemins inexistants : un recalcul échouerait.
        row = dict(site='R001', subject='sub-x', session='ses-1', t1_path='absent', mask_path='absent')
        self.assertEqual(process(row, self.dir)['status'], 'ok')
        self.assertEqual((output / 't1.nii.gz').read_bytes(), b'deja fait')

    def test_paire_manquante_signalee(self):
        row = dict(site='R001', subject='sub-x', session='ses-1', t1_path=None, mask_path='b')
        self.assertEqual(process(row, self.dir)['status'], 'error')


class InventoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def metadata(self, subject, content):
        anat = self.dir / 'R001' / subject / 'ses-1/anat'
        anat.mkdir(parents=True)
        (anat / f'{subject}_metadata.csv').write_text(content)

    def test_metadonnees_vides_signalees(self):
        self.metadata('sub-x', 'SITE,DAYS_POST_STROKE\n')
        (row,) = inventory(self.dir)
        self.assertIn('metadata_empty', row['inventory_warnings'])
        self.assertIsNone(row['days_post_stroke'])

    def test_chronicity_booleen(self):
        # ATLAS code le drapeau '1' ou '1.0' ; vide = non marqué.
        for subject, flag in [('sub-a', '1'), ('sub-b', '1.0'), ('sub-c', '')]:
            self.metadata(subject, f'SITE,CHRONICITY\nR001,{flag}\n')
        flags = {r['subject']: r['chronicity'] for r in inventory(self.dir)}
        self.assertEqual(flags, {'sub-a': True, 'sub-b': True, 'sub-c': None})


if __name__ == '__main__':
    unittest.main()
