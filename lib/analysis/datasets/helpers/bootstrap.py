

def sample_sets(df, num_events_per_dist:int, num_dists_per_dc9:int, replacement:bool):
    """
    Create a list of distribution dataframes.
    
    Each distribution is homogenous in delta C9.

    Distributions are bootstrapped from the original data.
    
    Parameters
    ----------
    df: pd.DataFrame
        The original ntuple dataframe.
    num_events_per_dist: int
        The number of events per sampled distribution.
    num_dists_per_dc9: int
        The number of sampled distributions per delta C9 value.
    
    Returns
    -------
    list of pd.DataFrame
        The sampled dataframes.
    """
    
    df_by_dc9 = df.groupby("dc9")
    
    result = []
    
    for _, df in df_by_dc9:
        for i in range(num_dists_per_dc9):
            sample = df.sample(n=num_events_per_dist, replace=replacement)
            result.append(sample)

    return result




