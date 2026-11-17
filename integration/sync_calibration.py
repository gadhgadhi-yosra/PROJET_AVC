"""Raccorde les configurations C5 à une calibration calculée sur le split actif."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / 'generalization/calibration.json'
    data = json.loads(path.read_text())
    fingerprint = hashlib.sha256((ROOT / 'splits.json').read_bytes()).hexdigest()[:16]
    if data['splits']['sha256_16'] != fingerprint:
        raise ValueError('Relancer la calibration avant de synchroniser les configurations')
    config_dir = ROOT / 'generalization/configs'
    replacements = {
        'intensity.yaml': {'gamma': data['gamma']['range'], 'factor': data['contrast']['range']},
        'bigaug.yaml': {'gamma': data['gamma']['range']},
        'fda.yaml': {'reference_sites': [data['reference']['site']],
                     'fold_reference_sites': [v['site'] for v in data['fold_references']]},
        'histogram_matching.yaml': {'reference_sites': [data['reference']['site']],
                     'fold_reference_sites': [v['site'] for v in data['fold_references']]},
    }
    for name, values in replacements.items():
        config = config_dir / name
        text = config.read_text()
        for key, value in values.items():
            text, count = re.subn(rf'(?m)^(\s+{key}:)\s*\[[^\]]*\].*$',
                                  lambda m: m[1] + ' ' + json.dumps(value), text)
            if count != 1:
                raise ValueError(f'{name}: clé {key} absente ou ambiguë')
        config.write_text(text)
    # Conserver les observations historiques des balayages, sans prétendre les avoir rejouées.
    old = json.loads((ROOT / 'integration/archive/calibration_v1.json').read_text())
    for key in ('fda_beta', 'bigaug_gamma'):
        data[key]['sweep_provenance'] = {'split_version': old['splits']['version'],
                                        'sha256_16': old['splits']['sha256_16'],
                                        'status': 'historical_sweep_not_rerun'}
    data['bigaug_gamma']['range'] = data['gamma']['range']
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print('Configurations C5 synchronisées avec la calibration du split actif.')


if __name__ == '__main__':
    main()
