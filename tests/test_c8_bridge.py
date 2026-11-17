"""Le pont démo → modèle C8 : erreurs claires et résultat relu, sans PyTorch ni vrai checkpoint."""
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'demo'))
import c8_bridge  # noqa: E402


def fake_python(folder, body):
    """Interpréteur factice : un script shell qui remplace `python -c CODE IRM`."""
    path = Path(folder) / 'python'
    path.write_text('#!/bin/sh\n' + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


class C8BridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.t1 = self.dir / 't1.nii.gz'
        self.t1.write_bytes(b'x')
        self.ckpt = self.dir / 'modele.pt'
        self.ckpt.write_bytes(b'x')

    def tearDown(self):
        self.tmp.cleanup()

    def run_with(self, python, **extra):
        env = {'AVC_C8_PYTHON': str(python), 'PROJET_AVC_MODELE': str(self.ckpt), **extra}
        with mock.patch.dict(os.environ, env):
            return c8_bridge.predict(self.t1)

    def test_missing_checkpoint_is_explicit(self):
        python = fake_python(self.dir, 'exit 0\n')
        with self.assertRaisesRegex(ValueError, 'Checkpoint C8 absent'):
            self.run_with(python, PROJET_AVC_MODELE=str(self.dir / 'absent.pt'))

    def test_missing_environment_is_explicit(self):
        with self.assertRaisesRegex(ValueError, 'Environnement du modèle introuvable'):
            self.run_with(self.dir / 'pas_de_python')

    def test_mask_written_next_to_input_is_returned(self):
        # Le processus enfant reçoit AVC_MASK_DIR = dossier de l'IRM et y écrit le masque.
        python = fake_python(self.dir, 'touch "$AVC_MASK_DIR/m.nii.gz"\n'
                                       'echo "AVC_C8_RESULT={\\"mask_path\\": \\"$AVC_MASK_DIR/m.nii.gz\\"}"\n')
        result = self.run_with(python)
        self.assertEqual(Path(result['mask_path']), self.dir.resolve() / 'm.nii.gz')

    def test_child_failure_reports_last_error_line(self):
        python = fake_python(self.dir, 'echo "RuntimeError: CUDA absente" >&2\nexit 1\n')
        with self.assertRaisesRegex(ValueError, 'CUDA absente'):
            self.run_with(python)

    def test_result_without_mask_file_is_rejected(self):
        python = fake_python(self.dir, 'echo "AVC_C8_RESULT={\\"mask_path\\": \\"/nulle/part.nii.gz\\"}"\n')
        with self.assertRaisesRegex(ValueError, 'pas produit de masque'):
            self.run_with(python)


if __name__ == '__main__':
    unittest.main()
