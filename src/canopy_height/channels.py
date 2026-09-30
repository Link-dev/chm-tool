"""Input channel layouts.

Two raw rasters are used, both uint16 at 10 m with 0 = no data:

  annual   76 bands: 0-63 annual Earth embedding (stored as round((e + 1) * 10000)),
           64 DEM, 65-66 Sentinel-1 VV / VH, 67-75 Sentinel-2 B2 B3 B4 B5 B6 B7 B8 B11 B12
  seasonal 44 bands: 0-7 Sentinel-1 (VV, VH) x seasons DJF, MAM, JJA, SON;
           8-43 Sentinel-2 (B2 B3 B4 B5 B6 B7 B8 B11 B12) x the same four seasons

This is the order and encoding the tool downloads and stacks (tool.project.X_LAYERS, tool.gee.layers.S1_BANDS /
S2_BANDS, tool.stack); the tool's training stacks equal the paper's bit for bit.

Model inputs are concatenations of band ranges of these rasters (INPUTS). `n_embed` is the number of
leading embedding channels, which the model maps to x / 10000 - 1; all other channels are standardised
with the per-channel statistics stored in the checkpoint.
"""
import numpy as np

ANNUAL_BANDS = ([f"embedding_{i:02d}" for i in range(64)] + ["DEM", "S1_VV", "S1_VH"] +
                [f"S2_{b}" for b in ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12")])
SEASONS = ("DJF", "MAM", "JJA", "SON")
SEASONAL_BANDS = ([f"S1_{p}_{s}" for s in SEASONS for p in ("VV", "VH")] +
                  [f"S2_{b}_{s}" for s in SEASONS
                   for b in ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12")])
assert len(ANNUAL_BANDS) == 76 and len(SEASONAL_BANDS) == 44

# Input representations (the letters: A = annual Sentinel-1/2 + DEM, E = Earth embedding, T = four seasonal
# (temporal) Sentinel-1/2 composites + DEM):
#   AE  embedding + annual Sentinel-1/2 + DEM (76 channels)
#   A   annual Sentinel-1/2 + DEM (12)
#   E   embedding (64)
#   T   DEM + four-season Sentinel-1/2 (45)
#   TE  embedding + DEM + four-season Sentinel-1/2 (109)
# AE and A were called IE and I before; those names are still accepted (ALIASES), e.g. in checkpoints, RF
# metadata and project files written earlier, and are translated to the current ones by canonical().
DESCRIPTIONS = {
    "AE": "Earth embedding + annual Sentinel-1/2 + DEM",
    "A": "annual Sentinel-1/2 + DEM",
    "E": "Earth embedding",
    "T": "four-season Sentinel-1/2 + DEM",
    "TE": "Earth embedding + four-season Sentinel-1/2 + DEM",
}
ALIASES = {"IE": "AE", "I": "A"}


class _Inputs(dict):
    """dict of the input representations that also finds them by their earlier names (ALIASES); iterating lists
    the current names only."""

    def __missing__(self, key):
        if key in ALIASES:
            return self[ALIASES[key]]
        raise KeyError(key)

    def __contains__(self, key):
        return dict.__contains__(self, key) or key in ALIASES

    def get(self, key, default=None):
        return self[key] if key in self else default


# name -> list of (raster, start, stop) and the number of leading embedding channels
# zero_restore: default for new models of this layout (the setting of the paper's models).
INPUTS = _Inputs({
    "AE": dict(parts=[("annual", 0, 76)], n_embed=64, zero_restore=True),                        # obs + embedding
    "A": dict(parts=[("annual", 64, 76)], n_embed=0, zero_restore=False),                        # satellite + DEM
    "E": dict(parts=[("annual", 0, 64)], n_embed=64, zero_restore=True),                         # embedding
    "T": dict(parts=[("annual", 64, 65), ("seasonal", 0, 44)], n_embed=0, zero_restore=False),   # DEM + seasonal
    "TE": dict(parts=[("annual", 0, 65), ("seasonal", 0, 44)], n_embed=64, zero_restore=True),   # emb + DEM + seasonal
})
for _v in INPUTS.values():
    _v["n_channels"] = sum(b - a for _, a, b in _v["parts"])
NAMES = list(INPUTS) + list(ALIASES)      # every accepted name (command-line choices)


def canonical(name):
    """Current name of an input representation (earlier names -> current ones); None stays None."""
    if name is None:
        return None
    name = str(name)
    if name not in INPUTS:
        raise ValueError(f"unknown input representation {name!r}; choose from "
                         + "; ".join(f"{k} = {v}" for k, v in DESCRIPTIONS.items()))
    return ALIASES.get(name, name)


def needs_seasonal(input_name):
    return any(r == "seasonal" for r, _, _ in INPUTS[input_name]["parts"])


def band_names(input_name):
    names = {"annual": ANNUAL_BANDS, "seasonal": SEASONAL_BANDS}
    return [n for r, a, b in INPUTS[input_name]["parts"] for n in names[r][a:b]]


def assemble(input_name, annual, seasonal=None):
    """Model input from raw rasters with bands on axis -3 ([..., bands, H, W])."""
    src = {"annual": annual, "seasonal": seasonal}
    parts = []
    for r, a, b in INPUTS[input_name]["parts"]:
        if src[r] is None:
            raise ValueError(f"input {input_name} needs the {r} raster")
        parts.append(src[r][..., a:b, :, :])
    return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=-3)


def statistics(input_name, annual_mean, annual_std, seasonal_mean=None, seasonal_std=None):
    """Per-channel mean / std vectors for a model input.

    annual_*: 76 values; seasonal_*: 45 values (DEM, then the 44 seasonal bands), as computed for the
    seasonal models. Embedding channels get mean 0 / std 1 (not used by the model).
    """
    spec = INPUTS[input_name]
    if needs_seasonal(input_name):
        e = spec["n_embed"]
        mean = np.concatenate([np.zeros(e), np.asarray(seasonal_mean, dtype=np.float64)])
        std = np.concatenate([np.ones(e), np.asarray(seasonal_std, dtype=np.float64)])
    else:
        idx = np.concatenate([np.arange(a, b) for _, a, b in spec["parts"]])
        mean, std = np.asarray(annual_mean)[idx], np.asarray(annual_std)[idx]
    assert mean.shape == (spec["n_channels"],)
    return mean, std
