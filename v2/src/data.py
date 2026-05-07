"""
Data loading, preprocessing, and augmentation for Tiny ImageNet.

Changes from V1:
- Proper three-way train / validation / test split.
- Horizontal flip augmentation applied per-batch during training.
- Augmentation is a separate function so it can be toggled and tested
  independently of the data loading logic.
"""

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Label utilities
# ---------------------------------------------------------------------------

def load_label_names(wnids_path: str) -> Tuple[List[str], dict]:
    """
    Load class label names from wnids.txt.

    Returns a list of label strings (in index order) and a dict mapping
    label string -> integer index.
    """
    with open(wnids_path, "r") as f:
        labelnames = [line.strip() for line in f.readlines()]
    label2index = {label: i for i, label in enumerate(labelnames)}
    return labelnames, label2index


def onehot(y: np.ndarray, k: int) -> np.ndarray:
    """One-hot encode integer labels into a (n_samples, k) array."""
    out = np.zeros((len(y), k), dtype=np.float64)
    out[np.arange(len(y)), y] = 1.0
    return out


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(train_dir: str, label2index: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Walk the Tiny ImageNet train directory and load all images into memory.

    Directory structure expected:
        train/<wnid>/images/<wnid>_<i>.JPEG

    Returns:
        X: uint8 image array of shape (n, 64, 64, 3).
        Y: integer labels of shape (n,).
    """
    paths = sorted(Path(train_dir).glob("*/images/*.*"))
    images: List[np.ndarray] = []
    labels: List[int] = []

    for path in paths:
        labelname = path.parent.parent.stem
        with Image.open(path) as img:
            images.append(np.array(img.convert("RGB"), dtype=np.uint8))
        labels.append(label2index[labelname])

    return np.array(images), np.array(labels, dtype=np.int32)


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------

def split_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, ...]:
    """
    Shuffle and split into train, validation, and test partitions.

    V1 used a single 80/20 train/test split, which meant hyperparameters
    were implicitly tuned on the test set. This three-way split keeps the
    test set genuinely held out.

    Returns:
        X_train, Y_train, X_val, Y_val, X_test, Y_test
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n = len(X)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)

    test_idx = idx[:n_test]
    val_idx = idx[n_test: n_test + n_val]
    train_idx = idx[n_test + n_val:]

    return (
        X[train_idx], Y[train_idx],
        X[val_idx], Y[val_idx],
        X[test_idx], Y[test_idx],
    )


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(
    X: np.ndarray,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """
    Flatten and normalize images to [-1, 1].

    Input X may be (n, H, W, C) or already flat (n, H*W*C).
    Zero-centered normalization pairs with He initialization and ReLU.
    """
    if X.ndim == 4:
        n = X.shape[0]
        X = X.reshape(n, -1)
    return (X.astype(dtype) / 128.0) - 1.0


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment_batch(
    X_batch: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply random horizontal flips to a preprocessed flat-image batch.

    Each image is flipped independently with probability 0.5. Flipping a
    flattened 64x64x3 image requires reshaping, flipping along the width
    axis, and reflattening.

    This is the only augmentation applied in V2 — deliberately conservative.
    Random crops and color jitter are left for V3.

    Args:
        X_batch: Preprocessed flat images, shape (batch, 12288).
        rng: NumPy random Generator (from the training loop).

    Returns:
        Augmented batch of the same shape.
    """
    batch = X_batch.reshape(-1, 64, 64, 3)
    flip_mask = rng.random(len(batch)) > 0.5
    batch[flip_mask] = batch[flip_mask, :, ::-1, :]
    return batch.reshape(X_batch.shape)
