"""Audit de raccordement C1 Pipeline / chargeur Fayfa / résultats historiques baselines."""
import hashlib
import json
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def baseline_compatibility(splits):
    old = pd.read_csv(ROOT / 'baselines/outputs/split.csv')
    cases = pd.DataFrame(splits['cases'])
    mapped = old.merge(cases[['subject', 'case_id']], on='subject', how='left', validate='one_to_one')
    roles = {c: role for role in ('train', 'id_test', 'ood_test') for c in splits[role]}
    mapped['current_role'] = mapped.case_id.map(roles).fillna('excluded')
    summary = pd.crosstab(mapped['set'], mapped.current_role)
    result = {'historical_protocol': '903 train / 226 id_val / 287 ood_test; no C1 fold equivalence',
              'current_protocol': 'C1 split v2',
              'assignment_counts': json.loads(summary.to_json(orient='index')),
              'compatible_with_current_s7': False,
              'reason': 'Different patient/center split; ID used as validation, not an independent C1 test.'}
    output = ROOT / 'data/predictions/historical_baselines'
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for file in sorted((ROOT / 'baselines/outputs/fast_results').glob('*_cases.csv')):
        metrics = pd.read_csv(file)
        data = metrics.merge(mapped[['case', 'case_id', 'site', 'set', 'current_role']], on='case',
                             validate='one_to_one')
        if len(data) != len(metrics):
            raise ValueError(f'Missing identifiers: {file}')
        data['protocol'] = 'historical_baselines_00e2692'
        data.to_csv(output / file.name, index=False)
        counts[file.stem] = len(data)
    result['mapped_result_rows'] = counts
    return result


def main():
    from generalization.data_processed import list_processed
    path = ROOT / 'splits.json'
    splits = json.loads(path.read_text())
    counts = {}
    for role in ('train', 'id_test', 'ood_test'):
        table = list_processed(ROOT / 'data/processed', role, splits=path, warn_empty=False)
        assert set(table.case_id) == set(splits[role])
        counts[role] = len(table)
    for fold, partition in enumerate(splits['folds']):
        for role in ('train', 'val', 'id_test'):
            table = list_processed(ROOT / 'data/processed', role, fold, path, warn_empty=False)
            assert set(table.case_id) == set(partition[role])
    result = {'status': 'passed', 'split_version': splits['version'],
              'split_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'c1_loader_counts': counts, 'folds_checked': len(splits['folds']),
              'baseline_results': baseline_compatibility(splits)}
    output = ROOT / 'integration/reports'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'team_compatibility.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
