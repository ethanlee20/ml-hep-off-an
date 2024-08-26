
"""
Calculation functions for distribution asymmetries.

Includes calculations for A_FB and S_5.
"""


from math import sqrt, pi
import numpy as np

from .util import count_events, bin_data, find_bin_middles



def afb_fn(ell):
    """
    Generate a function to calculate A_FB
    for a particular lepton flavor.

    Parameters
    ----------
    ell : str
        Lepton flavor ("mu" or "e")
    
    Returns
    -------
    function
        Function that takes an ntuple dataframe as input
        and outputs A_FB.
    """

    assert ell in {"mu", "e"}

    def fn(df):
        if ell == 'mu':
            d_cos_theta_l = df['costheta_mu']
        elif ell == 'e':
            d_cos_theta_l = df['costheta_e']

        f = count_events(df[(d_cos_theta_l > 0) & (d_cos_theta_l < 1)])
        b = count_events(df[(d_cos_theta_l > -1) & (d_cos_theta_l < 0)])
        
        afb = (f - b) / (f + b)
        return afb
    
    return fn


def afb_err_fn(ell):
    """
    Generate a function to calculate the error of A_FB
    for a particular lepton flavor.

    The error is calculated by assuming the forward and backward
    regions have uncorrelated Poisson errors and propagating
    the errors.

    Parameters
    ----------
    ell : str
        Lepton flavor ("mu" or "e")
    
    Returns
    -------
    function
        Function that takes an ntuple dataframe as input
        and outputs the error of A_FB.
    """

    assert ell in {"mu", "e"}
    
    def fn(df):
        if ell == 'mu':
            d_cos_theta_l = df['costheta_mu']
        elif ell == 'e':
            d_cos_theta_l = df['costheta_e']

        f = count_events(df[(d_cos_theta_l > 0) & (d_cos_theta_l < 1)])
        b = count_events(df[(d_cos_theta_l > -1) & (d_cos_theta_l < 0)])
        
        f_stdev = sqrt(f)
        b_stdev = sqrt(b)

        afb_stdev = 2*f*b / (f+b)**2 * sqrt((f_stdev/f)**2 + (b_stdev/b)**2)
        return afb_stdev
    
    return fn

  
def calc_afb_of_q_squared(df, ell, num_points):
    """
    Calcuate A_FB for each bin of a distribution binned in q^2.

    Parameters
    ----------
    df : pd.DataFrame 
        Ntuple dataframe
    ell : str
        Lepton flavor ("mu" or "e")
    num_points : int
        Number of q^2 bins

    Returns
    -------
    q_squareds : np.ndarray
        Middles of q^2 bins
    afbs : pd.Series
        A_FB per bin
    errs : pd.Series
        A_FB error per bin
    """

    df = df[(df['q_squared']>0) & (df['q_squared']<20)]
    binned, edges = bin_data(df, 'q_squared', num_points, ret_edges=True)
    
    afbs = binned.apply(afb_fn(ell))
    errs = binned.apply(afb_err_fn(ell))
    q_squareds = find_bin_middles(edges)

    return q_squareds, afbs, errs


def calc_s5(df):
    """
    Calculate S_5.

    Parameters
    ----------
    df : pd.DataFrame
        Ntuple dataframe

    Returns
    -------
    s5: float
        S_5
    """
    
    costheta_k = df["costheta_K"]
    chi = df["chi"]
    
    f = count_events(df[
        (((costheta_k > 0) & (costheta_k < 1)) & ((chi > 0) & (chi < pi/2)))
        | (((costheta_k > 0) & (costheta_k < 1)) & ((chi > 3*pi/2) & (chi < 2*pi)))
        | (((costheta_k > -1) & (costheta_k < 0)) & ((chi > pi/2) & (chi < 3*pi/2)))
    ])

    b = count_events(df[
        (((costheta_k > 0) & (costheta_k < 1)) & ((chi > pi/2) & (chi < 3*pi/2)))
        | (((costheta_k > -1) & (costheta_k < 0)) & ((chi > 0) & (chi < pi/2)))
        | (((costheta_k > -1) & (costheta_k < 0)) & ((chi > 3*pi/2) & (chi < 2*pi)))
    ])

    try: 
        s5 = 4/3 * (f - b) / (f + b)
    except ZeroDivisionError:
        print("division by 0, returning nan")
        s5 = np.nan
    
    return s5


def calc_s5_err(df):
    """
    Calculate the error of S_5.

    The error is calculated by assuming the "forward" and "backward"
    regions have uncorrelated Poisson errors and propagating
    the errors.

    Parameters
    ----------
    df  : pd.DataFrame
        Ntuple dataframe

    Returns
    -------
    err : float
        Error of S_5
    """

    costheta_k = df["costheta_K"]
    chi = df["chi"]
    
    f = count_events(df[
        (((costheta_k > 0) & (costheta_k < 1)) & ((chi > 0) & (chi < pi/2)))
        | (((costheta_k > 0) & (costheta_k < 1)) & ((chi > 3*pi/2) & (chi < 2*pi)))
        | (((costheta_k > -1) & (costheta_k < 0)) & ((chi > pi/2) & (chi < 3*pi/2)))
    ])

    b = count_events(df[
        (((costheta_k > 0) & (costheta_k < 1)) & ((chi > pi/2) & (chi < 3*pi/2)))
        | (((costheta_k > -1) & (costheta_k < 0)) & ((chi > 0) & (chi < pi/2)))
        | (((costheta_k > -1) & (costheta_k < 0)) & ((chi > 3*pi/2) & (chi < 2*pi)))
    ])

    f_stdev = sqrt(f)
    b_stdev = sqrt(b)

    try: 
        stdev = 4/3 * 2*f*b / (f+b)**2 * sqrt((f_stdev/f)**2 + (b_stdev/b)**2)
        err = stdev

    except ZeroDivisionError:
        print("division by 0, returning nan")
        err = np.nan
    
    return err


def calc_s5_of_q_squared(df, num_points):
    """
    Calcuate S_5 for each bin of a distribution
    binned in q^2.

    Parameters
    ----------
    df : pd.DataFrame
        Ntuple dataframe
    num_points : int
        Number of q^2 bins

    Returns
    -------
    q_squareds : np.ndarray
        Middles of q^2 bins
    afbs : pd.Series
        S_5 per bin
    errs : pd.Series
        S_5 error per bin
    """

    df = df[(df['q_squared']>0) & (df['q_squared']<20)]
    binned, edges = bin_data(df, 'q_squared', num_points, ret_edges=True)
    
    s5s = binned.apply(calc_s5)
    errs = binned.apply(calc_s5_err)
    q_squareds = find_bin_middles(edges)

    return q_squareds, s5s, errs