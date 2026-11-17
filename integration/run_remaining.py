"""Contrôle une livraison C3 puis exécute S7/S8, sans inventer les entrées manquantes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'analysis'))
from result_contract import load_results


def validate_experiment(item, splits_path, root=ROOT):
    required = {'model','technique','fold','results','split_sha256'}
    if not required <= set(item):
        raise ValueError('Champs manquants : ' + ', '.join(sorted(required-set(item))))
    actual = hashlib.sha256(splits_path.read_bytes()).hexdigest()
    if item['split_sha256'] != actual:
        raise ValueError('Empreinte du split différente : résultats non utilisables comme C1 courant')
    result_path = root/item['results']
    if not result_path.is_file():
        return {'status':'missing', 'path':item['results']}
    data, splits, fold = load_results(result_path, splits_path, item['model'], item['technique'], item['fold'])
    expected = set(fold['val']) | set(fold['id_test']) | set(splits['ood_test'])
    missing = expected-set(data.case_id)
    if missing:
        raise ValueError(f'Livraison C3 incomplète : {len(missing)} scores val/ID/OOD manquants')
    return {'status':'ready', 'scores':len(data), 'centers':int(data.site.nunique()),
            'results_sha256':hashlib.sha256(result_path.read_bytes()).hexdigest()}


def execute_step(command, destination):
    destination.mkdir(parents=True,exist_ok=True)
    with (destination/'execution.log').open('w') as handle:
        completed = subprocess.run(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
    return {'status':'completed' if completed.returncode==0 else 'failed',
            'returncode':completed.returncode,'log':str((destination/'execution.log').relative_to(ROOT))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'integration/experiments.json')
    parser.add_argument('--run',action='store_true',help='Exécuter les expériences dont les données sont prêtes')
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    splits_path=ROOT/manifest.get('splits','splits.json')
    entries=manifest.get('experiments',[])
    if not entries:
        raise ValueError('Aucune expérience déclarée')
    identifiers=[entry['id'] for entry in entries]
    if len(set(identifiers)) != len(identifiers) or any(not identifier or not all(c.isalnum() or c in '_-' for c in identifier) for identifier in identifiers):
        raise ValueError('Identifiants uniques attendus : lettres, chiffres, tirets et underscores')
    report={'date':datetime.now(timezone.utc).isoformat(),'mode':'run' if args.run else 'audit',
            'split_sha256':hashlib.sha256(splits_path.read_bytes()).hexdigest(),'experiments':{}}
    run_root=ROOT/'data/analysis/final_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    failed=False
    for item in entries:
        try:
            status=validate_experiment(item,splits_path)
        except (ValueError,KeyError,TypeError,OSError) as error:
            status={'status':'invalid','reason':str(error)}
        report['experiments'][item['id']]=status
        if status['status']!='ready':
            failed=True
            continue
        if not args.run:
            continue
        base=[ '--results',str(ROOT/item['results']),'--model',item['model'],
               '--technique',item['technique'],'--fold',str(item['fold']),'--splits',str(splits_path)]
        status['steps']={}
        for name,script,extra in (
            ('s7','failure_prediction.py',['--histograms',str(ROOT/'data/processed/post_stats.parquet')]),
            ('s8','temporal_shift.py',['--subjects',str(ROOT/'data/processed/subjects.parquet')])):
            output=run_root/item['id']/name
            status['steps'][name]=execute_step([sys.executable,str(ROOT/'analysis'/script),*base,*extra,'--output',str(output)],output)
        status['status']='completed' if all(s['status']=='completed' for s in status['steps'].values()) else 'failed'
        failed |= status['status']=='failed'
    report['status']='inputs_required_or_failed' if failed else ('completed' if args.run else 'ready')
    report['scope']='S7/S8 C1 uniquement ; aucune inférence, extraction réseau ou comparaison DANN simulée'
    output=ROOT/'integration/reports/remaining_readiness.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 2 if failed else 0


if __name__=='__main__':
    raise SystemExit(main())
