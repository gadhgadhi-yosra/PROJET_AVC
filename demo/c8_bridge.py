"""Pont entre la démo et le modèle final C8 de l'équipe (`adaptation.demo_adapter`).

La démo et le frontend tournent dans `.venv-demo` (Gradio, FastAPI, sans PyTorch) ; le modèle
a besoin de PyTorch et MONAI (`.venv-integration`). Plutôt que de mélanger les deux environnements,
`predict` lance l'adaptateur de l'équipe dans un processus séparé et renvoie son résultat.

    AVC_PREDICT_ADAPTER=c8_bridge:predict .venv-demo/bin/python frontend/server.py

Variables facultatives :
- `AVC_C8_PYTHON` : interpréteur qui a PyTorch et MONAI (défaut : `.venv-integration/bin/python`) ;
- `PROJET_AVC_MODELE` : checkpoint, transmis tel quel (défaut de l'équipe :
  `data/modeles/unet3d_intensity_pli0_best.pt`, jamais versionné) ;
- `AVC_C8_TIMEOUT` : durée maximale en secondes (défaut 600).

Le masque est écrit à côté de l'IRM reçue : le dossier temporaire de la démo le supprime avec elle.
"""
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = ROOT / '.venv-integration' / 'bin' / 'python'
DEFAULT_CHECKPOINT = ROOT / 'data' / 'modeles' / 'unet3d_intensity_pli0_best.pt'
CHILD = ('import json, sys\n'
         'from adaptation.demo_adapter import predict\n'
         'print("AVC_C8_RESULT=" + json.dumps(predict(sys.argv[1])))\n')


def predict(t1_path):
    t1 = Path(t1_path).resolve()
    python = Path(os.environ.get('AVC_C8_PYTHON', DEFAULT_PYTHON))
    if not python.exists():
        raise ValueError(f'Environnement du modèle introuvable : {python} (définir AVC_C8_PYTHON)')
    checkpoint = Path(os.environ.get('PROJET_AVC_MODELE', DEFAULT_CHECKPOINT))
    if not checkpoint.is_file():
        raise ValueError(f'Checkpoint C8 absent : {checkpoint} (à récupérer auprès de l’équipe, jamais dans Git)')
    env = dict(os.environ, AVC_MASK_DIR=str(t1.parent), PROJET_AVC_MODELE=str(checkpoint))
    try:
        done = subprocess.run([str(python), '-c', CHILD, str(t1)], cwd=ROOT, env=env, capture_output=True,
                              text=True, timeout=float(os.environ.get('AVC_C8_TIMEOUT', 600)))
    except subprocess.TimeoutExpired as error:
        raise ValueError('Le modèle C8 a dépassé le temps maximal') from error
    lines = [l for l in done.stdout.splitlines() if l.startswith('AVC_C8_RESULT=')]
    if done.returncode != 0 or not lines:
        detail = (done.stderr.strip().splitlines() or ['sans message'])[-1]
        raise ValueError(f'Échec du modèle C8 : {detail}')
    result = json.loads(lines[-1].split('=', 1)[1])
    if not isinstance(result, dict) or not Path(result.get('mask_path', '')).is_file():
        raise ValueError('Le modèle C8 n’a pas produit de masque')
    return result
