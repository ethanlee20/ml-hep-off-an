
def calculate_resolution(data, variable, q_squared_split):
    """
    Calculate the resolution.

    The resolution of a variable is defined as the
    reconstructed value minus the MC truth value.

    If the variable is chi, periodicity is accounted for.
    """

    data_calc = section(data, sig_noise='sig', var=variable, q_squared_split=q_squared_split).loc["det"]
    data_mc = section(data, sig_noise='sig', var=variable+'_mc', q_squared_split=q_squared_split).loc["det"]

    resolution = data_calc - data_mc

    if variable != "chi":
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


