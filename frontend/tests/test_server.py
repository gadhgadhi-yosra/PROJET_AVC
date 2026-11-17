"""Vérifie que chaque section du frontend lit les vrais rapports du dépôt."""
import gzip
import io
from pathlib import Path
import sys
import unittest

import nibabel as nib
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import app  # noqa: E402

client = TestClient(app)


def nifti(data):
    buffer = io.BytesIO()
    image = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
    buffer.write(gzip.compress(image.to_bytes()))
    return buffer.getvalue()


class FrontendTest(unittest.TestCase):
    def test_sections(self):
        for path in ('overview', 'qc', 'preprocessing', 'shift', 'models', 'adaptation', 'figures'):
            response = client.get(f'/api/{path}')
            self.assertEqual(response.status_code, 200, path)
        overview = client.get('/api/overview').json()
        self.assertEqual(overview['counts']['retained'], 1415)
        self.assertEqual(overview['split']['ood_test'], 268)

    def test_centers_and_conclusions(self):
        centers = client.get('/api/centers').json()['centers']
        r008 = next(c for c in centers if c['site'] == 'R008')
        self.assertEqual(r008['role'], 'OOD')
        self.assertEqual(next(c for c in centers if c['site'] == 'R032')['role'], 'Exclu')
        conclusions = client.get('/api/conclusions').json()
        self.assertEqual(conclusions['counts']['retained'], 1415)

    def test_technique_results_from_contract_file(self):
        import json, tempfile
        from server import technique_results, ROOT
        splits = json.loads((ROOT / 'splits.json').read_text())
        id_case, ood_case = splits['id_test'][0], splits['ood_test'][0]
        with tempfile.TemporaryDirectory(dir=ROOT / 'frontend' / 'tests') as folder:
            path = Path(folder) / 'results.csv'
            path.write_text('model,technique,fold,case_id,site,dice\n'
                            f'm,baseline,0,{id_case},R001,0.6\nm,baseline,0,{ood_case},R008,0.4\n'
                            f'm,dann,0,{ood_case},R008,0.5\nm,dann,0,inconnu,R001,0.9\n')
            result = technique_results(str(path.relative_to(ROOT)))
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['techniques'], ['baseline', 'dann'])
        self.assertEqual(result['ignored'], 1)
        dann_ood = next(x for x in result['summary'] if x['technique'] == 'dann' and x['role'] == 'OOD')
        self.assertAlmostEqual(dann_ood['mean_dice'], 0.5)

    def test_contribution_reads_weekly_plan(self):
        data = client.get('/api/contribution').json()
        self.assertEqual(data['weeks'][0]['weeks'], 'S1–S2')
        self.assertEqual(data['weeks'][-1]['weeks'], 'S13–S14')
        from server import ROOT
        import json
        expected = json.loads((ROOT / 'integration/reports/verification.json').read_text())['pipeline_tests']['passed']
        self.assertEqual(data['tests'], expected)

    def test_pages_and_files(self):
        self.assertIn('Projet AVC', client.get('/').text)
        figure = client.get('/api/figures').json()[0]['path']
        self.assertEqual(client.get(f'/files/{figure}').status_code, 200)
        self.assertEqual(client.get('/files/splits.json').status_code, 404)
        self.assertEqual(client.get('/files/../README.md').status_code, 404)

    def test_analyze_with_supplied_mask(self):
        t1 = np.random.default_rng(0).random((20, 20, 20))
        mask = np.zeros((20, 20, 20)); mask[8:12, 8:12, 8:12] = 1
        response = client.post('/api/analyze', files={
            't1': ('t1.nii.gz', nifti(t1)), 'mask': ('mask.nii.gz', nifti(mask))})
        self.assertEqual(response.status_code, 200, response.text)
        report = response.json()
        self.assertAlmostEqual(report['lesion_volume_ml'], 64 / 1000)
        self.assertTrue(report['slices_png'])

    def test_analyze_rejects_misaligned_mask(self):
        response = client.post('/api/analyze', files={
            't1': ('t1.nii.gz', nifti(np.ones((10, 10, 10)))), 'mask': ('m.nii.gz', nifti(np.zeros((8, 8, 8))))})
        self.assertEqual(response.status_code, 400)


if __name__ == '__main__':
    unittest.main()
