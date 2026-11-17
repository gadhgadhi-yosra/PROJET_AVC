"""Calculs de démo indépendants de Gradio ; aucune prédiction simulée."""
import importlib
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLCONFIGDIR', '/tmp/avc-mpl')
import nibabel as nib
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'analysis'))
from center_distance import brain_histogram, center_distance


def load_volume(path):
    image = nib.load(str(path))
    if len(image.shape) != 3 or np.prod(image.shape) > 150_000_000:
        raise ValueError('Une IRM 3D de moins de 150 millions de voxels est requise')
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3,:3])) < 1e-9:
        raise ValueError('Géométrie NIfTI invalide')
    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all(): raise ValueError('Valeurs non finies dans le volume')
    return image, data


def aligned(path, reference, binary=False):
    image, data = load_volume(path)
    if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-4):
        raise ValueError('Les volumes doivent partager la même grille physique')
    if binary:
        if not (np.isclose(data,0) | np.isclose(data,1)).all(): raise ValueError('Masque non binaire')
        data = data > .5
    return data


def predict(path, adapter):
    module, name = adapter.split(':')
    function = getattr(importlib.import_module(module), name)
    result = function(str(path))
    if not isinstance(result, dict) or 'mask_path' not in result:
        raise ValueError('predict(path) doit renvoyer mask_path et éventuellement uncertainty_path')
    return result


def summarize(image, mask, uncertainty=None, atlas=None, labels=None):
    voxel_ml = abs(float(np.linalg.det(image.affine[:3,:3])))/1000
    report = dict(usage='Démonstration de recherche — non validée pour une décision clinique',
                  lesion_volume_ml=float(mask.sum()*voxel_ml), lesion_voxels=int(mask.sum()),
                  uncertainty=None, regions=None)
    if uncertainty is not None:
        if not np.isfinite(uncertainty).all() or (uncertainty < 0).any():
            raise ValueError('Incertitude attendue finie et positive')
        report['uncertainty'] = dict(mean_volume=float(uncertainty.mean()),
            mean_lesion=float(uncertainty[mask].mean()) if mask.any() else None,
            interpretation='Échelle propre au modèle ; pas une probabilité d’erreur calibrée')
    if atlas is not None:
        if not np.allclose(atlas, np.rint(atlas)) or (atlas < 0).any(): raise ValueError('Atlas de labels entiers positifs requis')
        ids, counts = np.unique(atlas[mask].astype(int), return_counts=True)
        report['regions'] = [dict(label=int(i), name=(labels or {}).get(str(i), 'Non attribué' if i==0 else f'Région {i}'),
                                  volume_ml=float(n*voxel_ml)) for i,n in zip(ids,counts)]
    return report


def ood_distance(image, x, brain_path, reference_path):
    """Distance et alerte individuelles ; seuil distinct de la calibration S7 par centre."""
    brain = aligned(brain_path, image, binary=True)
    if not brain.any(): raise ValueError('Masque cérébral vide')
    values = x[brain]
    if abs(float(values.mean())) > .01 or abs(float(values.std())-1) > .01:
        raise ValueError('OOD exige une image prétraitée z-score et son masque cérébral')
    with np.load(reference_path, allow_pickle=False) as ref:
        hist = ref['histograms']
        threshold = float(ref['threshold']) if 'threshold' in ref else None
        kind = str(ref['kind']) if 'kind' in ref else ''
    distance = center_distance(brain_histogram(x, brain)[None,:], hist)
    calibrated = threshold is not None and np.isfinite(threshold) and kind == 'individual_histogram_wasserstein'
    return dict(distance=distance, threshold=threshold if calibrated else None,
                alert=bool(distance > threshold) if calibrated else None,
                status='Atypie d’intensité ; aucune garantie sur le Dice' if calibrated else 'Distance seule, sans seuil individuel calibré')


def plot_slices(image, x, mask, fraction=.5):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Réorientation conjointe pour ne pas inverser le masque par rapport à l'IRM.
    canonical = nib.as_closest_canonical(nib.Nifti1Image(x, image.affine))
    label = nib.as_closest_canonical(nib.Nifti1Image(mask.astype(np.uint8), image.affine))
    x, mask = canonical.get_fdata(), label.get_fdata() > .5
    lo, hi = np.percentile(x, [1,99]); fig, axes = plt.subplots(1,3,figsize=(12,4))
    for axis, (ax, title) in enumerate(zip(axes, ('Sagittale','Coronale','Axiale'))):
        index = min(x.shape[axis]-1, int(fraction*(x.shape[axis]-1)))
        base, overlay = np.take(x,index,axis=axis).T, np.take(mask,index,axis=axis).T
        ax.imshow(base, cmap='gray', origin='lower', vmin=lo,vmax=hi)
        ax.imshow(np.ma.masked_where(~overlay, overlay), cmap='autumn', alpha=.55, origin='lower', vmin=0,vmax=1)
        ax.set_title(f'{title} · coupe {index}'); ax.axis('off')
    fig.tight_layout()
    return fig
