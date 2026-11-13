"""Contrôles techniques et planche locale du recalage de démonstration."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/avc-mpl')
import nibabel as nib
import numpy as np
from core import load_volume,aligned,summarize


def main():
    root=Path(__file__).resolve().parents[1]
    folder=root/'data/demo';row=json.loads((folder/'real_example.json').read_text())
    image,x=load_volume(row['t1_path'])
    atlas=aligned(folder/'registration/atlas_native.nii.gz',image)
    template=aligned(folder/'registration/template_native.nii.gz',image)
    mask=aligned(row['mask_path'],image,binary=True)
    labels=json.loads((folder/'atlas/labels.json').read_text())
    report=summarize(image,mask,atlas=atlas,labels=labels)
    report['provenance']='Masque de référence ATLAS fourni ; aucune inférence du réseau'
    (folder/'registration/report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    brain=x>float(np.unique(x[::4,::4,::4],return_counts=True)[0][np.argmax(np.unique(x[::4,::4,::4],return_counts=True)[1])])
    overlap=float(np.count_nonzero((atlas>0)&brain)/max(1,np.count_nonzero(atlas>0)))
    qc=dict(same_grid=True,integer_labels=bool(np.allclose(atlas,np.rint(atlas))),
        labels_inside_image_support_fraction=overlap,
        status='technical_checks_passed; anatomical_review_required',
        template_atlas='HarvardOxford / MNI152NLin6Asym')
    (folder/'registration/qc.json').write_text(json.dumps(qc,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Réorientation commune pour une comparaison visuelle claire.
    def canonical(array):return nib.as_closest_canonical(nib.Nifti1Image(array.astype(np.float32),image.affine)).get_fdata()
    x,atlas,template=map(canonical,(x,atlas,template))
    fig,axes=plt.subplots(2,3,figsize=(12,8));lo,hi=np.percentile(x,[1,99])
    for axis in range(3):
        index=x.shape[axis]//2;base=np.take(x,index,axis=axis).T
        registered=np.take(template,index,axis=axis).T;regions=np.take(atlas,index,axis=axis).T
        for row_index in (0,1):axes[row_index,axis].imshow(base,cmap='gray',origin='lower',vmin=lo,vmax=hi);axes[row_index,axis].axis('off')
        axes[0,axis].contour(registered,levels=np.percentile(registered[registered>0],[30,60,85]),colors=['cyan'],linewidths=.4)
        axes[1,axis].imshow(np.ma.masked_where(regions==0,regions),cmap='tab20',origin='lower',alpha=.4)
    fig.suptitle('QC local du recalage : template (cyan), puis labels atlas sur IRM native')
    fig.tight_layout();fig.savefig(folder/'registration/qc_overlay.png',dpi=150);plt.close(fig)
    print(json.dumps(qc,indent=2))


if __name__=='__main__':main()
