"""
weights.py

Returns the zone-adaptive weight vector for a given geomorphic zone label.
Centralises weight lookup so all pipeline scripts use the same source of truth.
"""

import numpy as np
from processing.utils.constants import ZONE_WEIGHTS


def get_weights(zone_label: str) -> np.ndarray:
    """
    Return the [w_DEM, w_NDWI, w_tidal, w_veg, w_sensor] weight vector
    for the given zone label.

    Raises ValueError for unknown zone labels.
    """
    if zone_label not in ZONE_WEIGHTS:
        valid = list(ZONE_WEIGHTS.keys())
        raise ValueError(
            f"Unknown zone label '{zone_label}'. Valid labels: {valid}"
        )
    return np.array(ZONE_WEIGHTS[zone_label], dtype=np.float32)
