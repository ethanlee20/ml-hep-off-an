
"""
Generate aggregated datafiles.

Datafiles are aggregated over the simulation trials.
"""


from analysis.datasets import generate_agg_data


features = ["q_squared", "costheta_mu", "costheta_K", "chi"]

for split in {"train", "test"}:
    generate_agg_data(split, features)