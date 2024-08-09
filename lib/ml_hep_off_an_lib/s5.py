
"""
Calculation function for S5 (angular asymmetry).
"""

from math import pi, sqrt  
import numpy as np
from .util import count_events, bin_data, find_bin_middles


def calc_s5(df):
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
    Calculate S5 as a function of q squared.
    S5 is an angular asymmetry.
    """

    df = df[(df['q_squared']>0) & (df['q_squared']<20)]
    binned, edges = bin_data(df, 'q_squared', num_points, ret_edges=True)
    
    s5s = binned.apply(calc_s5)
    errs = binned.apply(calc_s5_err)
    q_squareds = find_bin_middles(edges)

    return q_squareds, s5s, errs