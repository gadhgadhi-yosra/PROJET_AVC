"""La voie locale de Fayfa respecte les exclusions du pipeline C1."""
import json
from pathlib import Path
import tempfile
import unittest
from generalization.preprocess_c1_local import partition_quarantine


class LocalC1PolicyTests(unittest.TestCase):
    def test_quarantined_subject_never_reaches_workers(self):
        rows = [dict(site='R001', subject='sub-a', session='ses-1'),
                dict(site='R065', subject='sub-b', session='ses-1')]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'qc.json'
            path.write_text(json.dumps({'cases': [{'case_id': 'R065_sub-b_ses-1', 'reason': 'review'}]}))
            active, excluded = partition_quarantine(rows, path)
        self.assertEqual(active, rows[:1])
        self.assertEqual(excluded[0]['status'], 'excluded')
        self.assertEqual(excluded[0]['case_id'], 'R065_sub-b_ses-1')
        self.assertNotIn('status', rows[1])


if __name__ == '__main__':
    unittest.main()
