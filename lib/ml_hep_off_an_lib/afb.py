
"""
Calculation functions for Afb (Forward - Backward Asymmetry).
"""

from math import sqrt
from util import bin_data, find_bin_middles


def _afb_fn(ell):
    def fn(df):
        if ell == 'mu':
            d_cos_theta_l = df['costheta_mu']
        elif ell == 'e':
            d_cos_theta_l = df['costheta_e']

        f = d_cos_theta_l[(d_cos_theta_l > 0) & (d_cos_theta_l < 1)].count()
        b = d_cos_theta_l[(d_cos_theta_l > -1) & (d_cos_theta_l < 0)].count()
        
        afb = (f - b) / (f + b)
        return afb
    return fn


def _afb_err_fn(ell):
    def fn(df):
        if ell == 'mu':
            d_cos_theta_l = df['costheta_mu']
        elif ell == 'e':
            d_cos_theta_l = df['costheta_e']

        f = d_cos_theta_l[(d_cos_theta_l > 0) & (d_cos_theta_l < 1)].count()
        b = d_cos_theta_l[(d_cos_theta_l > -1) & (d_cos_theta_l < 0)].count()
        
        f_stdev = sqrt(f)
        b_stdev = sqrt(b)

        afb_stdev = 2*f*b / (f+b)**2 * sqrt((f_stdev/f)**2 + (b_stdev/b)**2)
        return afb_stdev
    return fn

  
def calc_afb_of_q_squared(df, ell, num_points):
    """
    Calcuate Afb as a function of q squared.
    Afb is the forward-backward asymmetry.
    """

    df = df[(df['q_squared']>0) & (df['q_squared']<20)]
    binned, edges = bin_data(df, 'q_squared', num_points, ret_edges=True)
    
    afbs = binned.apply(_afb_fn(ell))
    errs = binned.apply(_afb_err_fn(ell))
    q_squareds = find_bin_middles(edges)

    return q_squareds, afbs, errs
