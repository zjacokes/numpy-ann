"""
Utilities for saving and loading trained model weights to JSON.

JSON was chosen over pickle/npz for portability and human-inspectability,
at the cost of larger file size.
"""

import json

import numpy as np


def load_weights(model, filename: str) -> None:
    """
    Load weights from a JSON file into the given model in place.

    The JSON file is expected to contain keys W1..W4 and b1..b4, each
    storing a nested list that np.array can convert to an ndarray.
    """
    with open(filename, "r") as f:
        weights = json.load(f)
    model.W1 = np.array(weights["W1"])
    model.b1 = np.array(weights["b1"])
    model.W2 = np.array(weights["W2"])
    model.b2 = np.array(weights["b2"])
    model.W3 = np.array(weights["W3"])
    model.b3 = np.array(weights["b3"])
    model.W4 = np.array(weights["W4"])
    model.b4 = np.array(weights["b4"])


def save_weights(model, filename: str) -> None:
    """Serialize the model's weight matrices to a JSON file."""
    weights = {
        "W1": model.W1.tolist(), "b1": model.b1.tolist(),
        "W2": model.W2.tolist(), "b2": model.b2.tolist(),
        "W3": model.W3.tolist(), "b3": model.b3.tolist(),
        "W4": model.W4.tolist(), "b4": model.b4.tolist(),
    }
    with open(filename, "w") as f:
        json.dump(weights, f)
