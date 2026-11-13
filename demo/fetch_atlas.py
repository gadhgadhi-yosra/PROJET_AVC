"""Prépare Harvard–Oxford et son template MNI152NLin6Asym compatible, en local."""
import hashlib
import json
from pathlib import Path
import tarfile
from urllib.request import urlopen
from xml.etree import ElementTree
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCES={
    'template.nii.gz':'https://templateflow.s3.amazonaws.com/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_desc-brain_T1w.nii.gz',
    'HarvardOxford.tgz':'https://www.nitrc.org/frs/download.php/9902/HarvardOxford.tgz',
    'template_description.json':'https://templateflow.s3.amazonaws.com/tpl-MNI152NLin6Asym/template_description.json',
}


def main():
    out=ROOT/'data/demo/atlas';out.mkdir(parents=True,exist_ok=True)
    records=[]
    for name,url in SOURCES.items():
        path=out/name
        if not path.exists():
            print('Téléchargement',name,flush=True)
            with urlopen(url,timeout=120) as response:data=response.read()
            temporary=path.with_suffix('.tmp');temporary.write_bytes(data);temporary.replace(path)
        records.append(dict(file=name,url=url,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    needed=['HarvardOxford-cort-maxprob-thr25-2mm.nii.gz','HarvardOxford-sub-maxprob-thr25-2mm.nii.gz',
            'HarvardOxford-Cortical.xml','HarvardOxford-Subcortical.xml']
    with tarfile.open(out/'HarvardOxford.tgz') as archive:
        for name in needed:
            matches=[m for m in archive.getmembers() if Path(m.name).name==name and m.isfile()]
            if len(matches)!=1:raise ValueError('Fichier atlas absent ou ambigu : '+name)
            with archive.extractfile(matches[0]) as stream:(out/name).write_bytes(stream.read())
    template=nib.load(out/'template.nii.gz');arrays=[];labels={'0':'Non attribué par Harvard–Oxford'}
    for prefix,offset,xml in [('cort',0,'Cortical'),('sub',100,'Subcortical')]:
        image=nib.load(out/f'HarvardOxford-{prefix}-maxprob-thr25-2mm.nii.gz')
        image=resample_from_to(image,(template.shape,template.affine),order=0)
        arrays.append(np.rint(image.get_fdata()).astype(np.uint16))
        for label in ElementTree.parse(out/f'HarvardOxford-{xml}.xml').findall('.//label'):
            labels[str(int(label.get('index'))+1+offset)]=label.text.strip()
    cortical,subcortical=arrays
    combined=np.where(cortical>0,cortical,np.where(subcortical>0,subcortical+100,0)).astype(np.uint16)
    nib.save(nib.Nifti1Image(combined,template.affine),out/'atlas_labels.nii.gz')
    (out/'labels.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2))
    provenance=dict(template='MNI152NLin6Asym',resolution_mm=2,atlas='Harvard–Oxford maxprob thr25',
        combination='Labels corticaux prioritaires ; labels sous-corticaux +100 dans les autres voxels',
        sources=records,reference='https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_atlas_harvard_oxford.html',
        registration_review_required=True)
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print('Atlas prêt :',out)


if __name__=='__main__':main()
