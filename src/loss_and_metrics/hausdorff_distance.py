import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import torch 


def cdist(x, y):
    """
    Compute distance between each pair of the two collections of inputs.
    :param x: Nxd Tensor
    :param y: Mxd Tensor
    :res: NxM matrix where dist[i,j] is the norm between x[i,:] and y[j,:],
          i.e. dist[i,j] = ||x[i,:]-y[j,:]||
    """
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences**2, -1).sqrt()
    return distances


# Code taken from https://github.com/javiribera/locating-objects-without-bboxes/blob/master/object-locator/losses.py
def averaged_hausdorff_distance(set1, set2, max_ahd=np.inf):
    """
    Compute the Averaged Hausdorff Distance function
    between two unordered sets of points (the function is symmetric).
    Batches are not supported, so squeeze your inputs first!
    :param set1: Array/list where each row/element is an N-dimensional point.
    :param set2: Array/list where each row/element is an N-dimensional point.
    :param max_ahd: Maximum AHD possible to return if any set is empty. Default: inf.
    :return: The Averaged Hausdorff Distance between set1 and set2.
    """

    if len(set1) == 0 or len(set2) == 0:
        return max_ahd

    set1 = np.array(set1)
    set2 = np.array(set2)

    assert set1.ndim == 2, 'got %s' % set1.ndim
    assert set2.ndim == 2, 'got %s' % set2.ndim

    assert set1.shape[1] == set2.shape[1], \
        'The points in both sets must have the same number of dimensions, got %s and %s.'\
        % (set2.shape[1], set2.shape[1])

    d2_matrix = pairwise_distances(set1, set2, metric='euclidean')

    res = np.average(np.min(d2_matrix, axis=0)) + \
        np.average(np.min(d2_matrix, axis=1))

    return (res, 1)