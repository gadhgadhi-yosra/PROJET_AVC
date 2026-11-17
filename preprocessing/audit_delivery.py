"""Audit C1/C2 : couverture du split, anomalies et présence de l'export."""
import json
from pathlib import Path
import pandas as pd


def main():
    root=Path(__file__).resolve().parents[1]
    subjects=pd.read_parquet(root/'data/processed/subjects.parquet')
    splits=json.loads((root/'splits.json').read_text())
    expected=set(splits['train'])|set(splits['id_test'])|set(splits['ood_test'])
    available=set(subjects.loc[subjects.status=='ok','case_id'])
    errors=subjects.loc[subjects.status=='error',['case_id','error']].to_dict('records')
    report=dict(expected_in_splits=len(expected),available=len(available),
        missing_from_splits=sorted(expected-available),unexpected_available=sorted(available-expected),
        errors=errors,excluded=int((subjects.status=='excluded').sum()),
        ready_for_full_export=expected==available and not errors,
        export_exists=(root/'data/nnUNet_raw/Dataset001_ATLAS/dataset.json').is_file())
    output=root/'preprocessing/reports/delivery_audit.json'
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
