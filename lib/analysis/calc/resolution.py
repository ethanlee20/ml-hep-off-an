

import numpy as np


def calculate_resolution(d_recon, d_truth, periodic=False):
    """
    Calculate the resolution.

    The resolution of a variable is defined as its
    reconstructed value minus its MC truth value.

    If specified, periodicity is accounted for.

    Parameters
    ----------
    d_truth : pd.Series
        The pandas series containing the truth data.
    d_recon : pd.Series
        The pandas series containing the reconstructed data.
    periodic : bool, optional
        Whether or not to account for periodicity.

    Returns
    -------
    pd.Series
        The resolution per event.
    """

    resolution = d_recon - d_truth

    if not periodic:
        return resolution
    
    def apply_periodicity(resolution):
        resolution = resolution.where(
                resolution < np.pi, resolution - 2 * np.pi
        )
        resolution = resolution.where(
            resolution > -np.pi, resolution + 2 * np.pi
        )
        return resolution

    return apply_periodicity(resolution)


