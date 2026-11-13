"""Animation de secours synthétique : visualisation, jamais résultat d'un modèle."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/avc-mpl')
from pathlib import Path
import io
import numpy as np
import nibabel as nib
from PIL import Image
from core import plot_slices, summarize


def main():
    import matplotlib.pyplot as plt
    root=Path('data/demo/synthetic');root.mkdir(parents=True,exist_ok=True)
    grid=np.indices((64,64,64),dtype=float)
    radius=((grid-32)**2).sum(axis=0)
    brain=radius<27**2
    volume=np.where(brain,80+20*np.cos(radius/110)+grid[0]/3,0).astype(np.float32)
    mask=(((grid[0]-40)/7)**2+((grid[1]-32)/9)**2+((grid[2]-32)/8)**2)<1
    image=nib.Nifti1Image(volume,np.eye(4))
    nib.save(image,root/'t1.nii.gz');nib.save(nib.Nifti1Image(mask.astype(np.uint8),np.eye(4)),root/'mask.nii.gz')
    frames=[]
    for position in np.linspace(.25,.75,16):
        fig=plot_slices(image,volume,mask,float(position))
        fig.suptitle('EXEMPLE SYNTHÉTIQUE · masque construit, aucune inférence',fontsize=11)
        buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=80);plt.close(fig)
        buffer.seek(0);frames.append(Image.open(buffer).convert('RGB'))
    out=Path('docs/yosra/demo_secours_synthetique.gif')
    frames[0].save(out,save_all=True,append_images=frames[1:],duration=200,loop=0)
    import json
    (root/'report.json').write_text(json.dumps(summarize(image,mask),indent=2))
    print(out)


if __name__=='__main__':main()
