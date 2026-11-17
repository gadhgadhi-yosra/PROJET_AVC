"""Distance de domaine v1 (contrat C6) : écart d'un centre à une référence, sur histogrammes d'intensité."""
import numpy as np

# Z-scores in the brain: values beyond the range are clipped into the edge bins.
BIN_EDGES = np.linspace(-5, 10, 151)
BIN_WIDTH = float(BIN_EDGES[1] - BIN_EDGES[0])


def brain_histogram(volume, brain):
    """Histogramme normalisé des voxels du cerveau d'un volume prétraité."""
    values = np.clip(np.asarray(volume)[np.asarray(brain) > 0], BIN_EDGES[0], BIN_EDGES[-1])
    if not values.size:
        raise ValueError('empty_brain')
    counts, _ = np.histogram(values, bins=BIN_EDGES)
    return counts / counts.sum()


def _as_scans(histograms):
    scans = np.atleast_2d(np.asarray(histograms, dtype=np.float64))
    if scans.ndim != 2 or scans.shape[1] != len(BIN_EDGES) - 1 or not len(scans):
        raise ValueError('expected_histograms_n_scans_x_n_bins')
    if not np.isfinite(scans).all() or (scans < 0).any() or not np.allclose(scans.sum(axis=1), 1):
        raise ValueError('invalid_histogram')
    return scans


def _squared_distances(a, b):
    return ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)


def _bandwidth(ref):
    """Largeur fixée sur la seule référence, pour comparer les centres entre eux."""
    distances = _squared_distances(ref, ref)[np.triu_indices(len(ref), k=1)]
    positive = distances[distances > 0]
    if not len(positive):
        raise ValueError('mmd_requires_distinct_reference_scans')
    return float(np.median(positive))


def center_distance(scans_centre, ref, metric='wasserstein'):
    """Renvoie l'écart entre les scans d'un centre et la référence (n_scans x n_bins).

    wasserstein : distance W1 entre histogrammes moyens, en écarts-types du z-score.
    mmd : MMD (estimateur biaisé, noyau gaussien) entre scans vus comme des échantillons.
    La référence ne doit pas contenir le centre mesuré. Les deux mesures dépendent du nombre
    de scans : comparer des centres d'effectifs proches, ou sous-échantillonner.
    """
    scans, reference = _as_scans(scans_centre), _as_scans(ref)
    if metric == 'wasserstein':
        gap = np.cumsum(scans.mean(axis=0)) - np.cumsum(reference.mean(axis=0))
        return float(np.abs(gap).sum() * BIN_WIDTH)
    if metric == 'mmd':
        scale = _bandwidth(reference)
        k = lambda a, b: np.exp(-_squared_distances(a, b) / scale)
        return float(np.sqrt(max(k(scans, scans).mean() + k(reference, reference).mean()
                                 - 2 * k(scans, reference).mean(), 0)))
    raise ValueError(f'unknown_metric: {metric}')
