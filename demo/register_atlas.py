"""Recaler un template MNI et son atlas de labels vers l'IRM native via ANTsPy."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/avc-mpl')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--t1', type=Path, required=True)
    p.add_argument('--template', type=Path, required=True)
    p.add_argument('--atlas', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    import ants
    fixed, moving, labels = [ants.image_read(str(v)) for v in (args.t1,args.template,args.atlas)]
    if not ants.image_physical_space_consistency(moving, labels):
        raise ValueError('Le template et les labels MNI ne partagent pas la même grille')
    args.output.mkdir(parents=True,exist_ok=True)
    registration=ants.registration(fixed=fixed,moving=moving,type_of_transform='SyN',
                                   outprefix=str(args.output/'mni_to_subject_'),random_seed=2026)
    warped=ants.apply_transforms(fixed=fixed,moving=labels,
        transformlist=registration['fwdtransforms'],interpolator='genericLabel')
    ants.image_write(warped,str(args.output/'atlas_native.nii.gz'))
    ants.image_write(registration['warpedmovout'],str(args.output/'template_native.nii.gz'))
    (args.output/'registration.json').write_text(json.dumps(dict(
        fixed=str(args.t1.resolve()),template=str(args.template.resolve()),atlas=str(args.atlas.resolve()),
        transforms=registration['fwdtransforms'],seed=2026,
        status='Recalage calculé ; superposition template/IRM à inspecter avant interprétation régionale'),indent=2))


if __name__=='__main__': main()
