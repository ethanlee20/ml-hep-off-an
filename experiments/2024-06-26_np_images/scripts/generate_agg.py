
"""
Generate aggregated datafiles.

Datafiles are aggregated over the simulation trials.
"""


from analysis.datasets import generate_agg_data


features = ["q_squared", "costheta_mu", "costheta_K", "chi", "q_squared_mc", "costheta_mu_mc", "costheta_K_mc", "chi_mc"]

for split in {"train", "test"}:
    generate_agg_data(split, features)