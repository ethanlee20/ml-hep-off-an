

def calc_eff(d_gen, d_det, n):
    """
    Calculate the efficiency per bin.

    The efficiency of bin i is defined as the number of
    detector entries in i divided by the number of generator
    entries in i.

    The error for bin i is calculated as the squareroot of the
    number of detector entries in i divided by the number of
    generator entries in i.

    d_gen is a series of generator level values.
    d_det is a series of detector level values.
    n is the number of efficiency datapoints to calculate.
    """
    
    min, max = min_max([d_gen, d_det])
    bin_edges, bin_mids = make_bin_edges(min, max, n, ret_middles=True)

    hist_gen, _ = np.histogram(d_gen, bins=bin_edges)
    hist_det, _ = np.histogram(d_det, bins=bin_edges)

    eff = hist_det / hist_gen
    err = np.sqrt(hist_det) / hist_gen

    return bin_mids, eff, err
