"""Helper functions for the Multimodal-IAD application."""

import numpy as np


def to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """Convert a (H, W, 3) float or uint8 array to uint8 RGB [0,255]."""
    if arr.dtype == np.uint8:
        return arr
    # Assume float in [0,1] or larger; robustly scale/clamp
    arr = np.clip(arr, 0, 1)
    return (arr * 255).round().astype(np.uint8)
