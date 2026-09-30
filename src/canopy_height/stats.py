"""Per-channel normalisation statistics of the 3DEP training pool (used for new models).

annual_*: 76 values for the annual raster; seasonal_*: 45 values for [DEM, 44 seasonal bands].
Statistics were computed over all training chips with zero (no-data) values excluded.
"""
from importlib import resources

import numpy as np

from . import channels


def load():
    with resources.files(__package__).joinpath("data/channel_stats.npz").open("rb") as f:
        z = np.load(f)
        return {k: z[k] for k in z.files}


def for_input(input_name):
    s = load()
    return channels.statistics(input_name, s["annual_mean"], s["annual_std"], s["seasonal_mean"], s["seasonal_std"])
