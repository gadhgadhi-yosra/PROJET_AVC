"""Vérifie les fichiers et plis de l’export courant, puis écrit sa provenance."""
from pathlib import Path
import json,hashlib
r=Path(__file__).resolve().parents[1]; p=r/'data/nnUNet_raw/Dataset001_ATLAS';s=json.loads((r/'splits.json').read_text())
checks={}
for folder,ids,suffix in [('imagesTr',s['train'],'_0000.nii.gz'),('labelsTr',s['train'],'.nii.gz'),('imagesTs',s['id_test']+s['ood_test'],'_0000.nii.gz'),('evaluation_labels/id',s['id_test'],'.nii.gz'),('evaluation_labels/ood',s['ood_test'],'.nii.gz')]:
    actual={f.name for f in (p/folder).glob('*.nii.gz')}; expected={c+suffix for c in ids}
    assert actual==expected,(folder,len(actual),len(expected));checks[folder]=len(actual)
assert json.loads((p/'splits_final.json').read_text())==[dict(train=f['train'],val=f['val']) for f in s['folds']]
assert json.loads((p/'evaluation_split.json').read_text())==dict(id_test=s['id_test'],ood_test=s['ood_test'])
report=dict(status='passed',split_version=s['version'],split_sha256=hashlib.sha256((r/'splits.json').read_bytes()).hexdigest(),file_counts=checks,fold_membership='exact_match',official_nnunet_validation='pending_training_environment')
(p/'split_provenance.json').write_text(json.dumps(report,indent=2)+'\n');(r/'preprocessing/reports/export_v2.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
