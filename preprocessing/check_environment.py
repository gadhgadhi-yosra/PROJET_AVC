"""Vérifie les imports et une opération ANTsPy, sans lire de données patient."""
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'avc-mpl'))
result = {'python': sys.version.split()[0]}
assert sys.version_info[:2] == (3, 12), 'Python 3.12 requis'
for distribution, module in [('pyspark','pyspark'), ('nibabel','nibabel'),
                             ('SimpleITK','SimpleITK'), ('numpy','numpy'),
                             ('pandas','pandas'), ('pyarrow','pyarrow'),
                             ('scipy','scipy'), ('antspyx','ants')]:
    importlib.import_module(module)
    result[distribution] = importlib.metadata.version(distribution)
import ants
import numpy as np
assert ants.from_numpy(np.zeros((3,3,3), dtype=np.float32)).shape == (3,3,3)
java_home = os.environ.get('JAVA_HOME') or str(Path(__file__).resolve().parents[1] / '.tools/jdk17/Contents/Home')
java = subprocess.run([str(Path(java_home)/'bin/java'), '-version'], capture_output=True, text=True, check=True)
result['java'] = java.stderr.splitlines()[0]
assert '"17.' in result['java'], 'Java 17 requis'
print(json.dumps(result, indent=2))
