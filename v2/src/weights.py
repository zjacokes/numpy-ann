"""
Utilities for saving and loading model weights to/from JSON.

Unchanged from V1. JSON is used for portability and human-inspectability.

V2 change: save/load operate on the model's `params` dict directly rather
than accessing named attributes, which matches the new model structure.
"""

import json

import numpy as np


def save_weights(params: dict, filename: str) -> None:
    """Serialize a parameter dict to JSON."""
    serializable = {k: v.tolist() for k, v in params.items()}
    with open(filename, "w") as f:
        json.dump(serializable, f)
    print(f"Weights saved to {filename}")


def load_weights(filename: str, dtype: np.dtype = np.float64) -> dict:
    """
    Load a parameter dict from JSON.

    Returns a dict of NumPy arrays with the given dtype.
    """
    with open(filename, "r") as f:
        raw = json.load(f)
    return {k: np.array(v, dtype=dtype) for k, v in raw.items()}
