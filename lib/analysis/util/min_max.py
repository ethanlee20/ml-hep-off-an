
import numpy as np


def min_max(a):
    """
    Find the min, max (tuple) of an array or list of arrays.
    
    Parameters
    ----------
    a : array or list of arrays

    Returns
    -------
    min : float
        The minimum
    max : float
        The maximum
    """

    if type(a) == list:
        a = np.concatenate(a, axis=None)
    min = np.nanmin(a)
    max = np.nanmax(a)
    return min, max   