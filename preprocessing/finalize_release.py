"""Versionne un split après quarantaine QC, sans déplacer les autres cas ou centres."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import pandas as pd
from week4 import ROOT, validate_splits, write_json


def revise_split(original, exclusions):
    result=copy.deepcopy(original)
    removed=set(exclusions)
    known={c['case_id'] for c in original['cases']}
    if not removed <= known: raise ValueError('Exclusion absente du split source')
    for key in ('train','id_test','ood_test'):
        result[key]=[c for c in original[key] if c not in removed]
    for fold in result['folds']:
        for key in ('train','val','id_test'):fold[key]=[c for c in fold[key] if c not in removed]
    result['cases']=[c for c in original['cases'] if c['case_id'] not in removed]
    result['version']=int(original['version'])+1
    result['revision']={'date':'2026-11-11','reason':'technical_QC_quarantine_before_training',
        'removed_cases':sorted(removed),'preserved':'centres OOD et affectation de chaque cas restant aux mêmes plis',
        'review_status':'technical_review_completed; team_review_pending'}
    result['excluded_cases'].extend(dict(case_id=c,status='excluded',error=exclusions[c]) for c in sorted(removed))
    result['data_validation']='processed_QC_passed_for_all_release_cases; team_review_pending'
    validate_splits(result)
    return result


def main():
    source=ROOT/'splits.json';archive=ROOT/'docs/pipeline/releases/splits_v1.json'
    original=json.loads(source.read_text())
    policy=json.loads((ROOT/'preprocessing/qc_exclusions.json').read_text())
    exclusions={c['case_id']:c['reason'] for c in policy['cases']}
    if original.get('version')!=1:
        if original.get('version')==2 and original.get('revision',{}).get('removed_cases')==sorted(exclusions):
            print('Version 2 déjà créée ; aucune modification.');return
        raise ValueError('Version source inattendue')
    processed=ROOT/'data/processed';frame=pd.read_parquet(processed/'subjects.parquet')
    if not set(exclusions)<=set(frame.loc[frame.status=='error','case_id']):
        raise ValueError('Les exclusions doivent correspondre aux échecs QC constatés')
    revised=revise_split(original,exclusions)
    expected={c['case_id'] for c in revised['cases']}
    available=set(frame.loc[frame.status=='ok','case_id'])
    if expected!=available:raise ValueError('La nouvelle cohorte ne correspond pas aux volumes validés')
    validation=json.loads((ROOT/'preprocessing/reports/validation_s4_complete_available.json').read_text())
    if validation['status']!='passed' or validation['subjects']!=len(expected):
        raise ValueError('Validation complète de la cohorte requise')
    archive.parent.mkdir(parents=True,exist_ok=True)
    if archive.exists() and archive.read_bytes()!=source.read_bytes():raise ValueError('Archive existante différente')
    if not archive.exists():shutil.copy2(source,archive)
    revised['revision']['source_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
    history=processed/'history_before_release_v2';history.mkdir(exist_ok=True)
    for name in ('subjects.parquet','summary.json'):
        if not (history/name).exists():shutil.copy2(processed/name,history/name)
    frame.loc[frame.case_id.isin(exclusions),'status']='excluded'
    for case,reason in exclusions.items():frame.loc[frame.case_id==case,'error']='QC_quarantine: '+reason
    temporary=processed/'subjects.release.tmp.parquet';frame.to_parquet(temporary,index=False)
    temporary.replace(processed/'subjects.parquet')
    summary=json.loads((processed/'summary.json').read_text())
    summary.update(statuses=frame.status.value_counts().to_dict(),release_split_version=2,
                   qc_exclusions='preprocessing/qc_exclusions.json')
    write_json(processed/'summary.json',summary);write_json(source,revised)
    write_json(ROOT/'preprocessing/reports/release_v2.json',dict(
        date='2026-11-11',split_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        archived_split=str(archive.relative_to(ROOT)),cases=len(expected),
        train_validation=len(revised['train']),id_test=len(revised['id_test']),ood_test=len(revised['ood_test']),
        quarantined=sorted(exclusions),raw_data_modified=False,team_review_pending=True))
    print('Release v2 créée :',len(expected),'cas, sans modification des IRM ni des annotations.')


if __name__=='__main__':main()
