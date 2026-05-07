"""
Data loading and preprocessing utilities for Tiny ImageNet.

The dataset has 200 classes, each with 500 training images sized 64x64x3.
This module handles the steps needed to feed those images into a fully
connected network: loading from disk, flattening, normalizing pixel
values, and one-hot encoding labels.
"""

import os
from typing import List, Tuple

import numpy as np
from PIL import Image


def onehot(y: np.ndarray, k: int) -> np.ndarray:
    """
    One-hot encode integer class labels.

    Args:
        y: Integer labels of shape (n_samples,).
        k: Total number of classes.

    Returns:
        Array of shape (n_samples, k) with a 1 in the column matching each label.
    """
    probs = np.zeros((len(y), k))
    for i, p in enumerate(y):
        probs[i][p] = 1
    return probs


def load_label_names(wnids_path: str) -> Tuple[List[str], dict]:
    """
    Load the class label names from the Tiny ImageNet wnids.txt file.

    Returns a list of label strings and a dict mapping label string to
    integer index.
    """
    with open(wnids_path, "r") as f:
        labelnames = [line.strip() for line in f.readlines()]
    label2index = {label: i for i, label in enumerate(labelnames)}
    return labelnames, label2index


def load_test_images(
    image_dir: str,
    label_file: str,
    label2index: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load test images and labels from disk.

    Reads the annotation file, then loads each referenced image as an RGB
    numpy array. Images that cannot be found on disk are skipped with a
    warning.

    Returns:
        X: Array of images, shape (n_samples, height, width, channels).
        Y: Integer labels, shape (n_samples,).
    """
    imagepaths: List[str] = []
    labels: List[str] = []

    with open(label_file, newline="") as f:
        for line in f.readlines():
            tokens = line.strip().split()
            filename = tokens[0]
            label = tokens[1]
            imagepath = os.path.join(image_dir, filename)
            if os.path.exists(imagepath):
                imagepaths.append(imagepath)
                labels.append(label)
            else:
                print(f"Warning: {imagepath} cannot be found. Skipping from the list.")

    X: List[np.ndarray] = []
    Y: List[int] = []
    for imagepath, label in zip(imagepaths, labels):
        with Image.open(imagepath) as image:
            X.append(np.array(image.convert("RGB")))
            Y.append(label2index[label])

    return np.array(X), np.array(Y)


def preprocess(X: np.ndarray, Y: np.ndarray, n_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the standard preprocessing pipeline used in this project.

    1. Flatten each image from (H, W, C) to a single vector of length H*W*C.
    2. Normalize pixel values from [0, 255] to [-1, 1].
    3. One-hot encode the labels.

    Zero-centered normalization (rather than [0, 1]) was chosen to pair
    with He initialization and ReLU activations.
    """
    height, width, channels = X.shape[1:]
    X_flat = np.reshape(X, (-1, width * height * channels))
    X_norm = (X_flat / 128.0) - 1
    Y_onehot = onehot(Y, n_classes)
    return X_norm, Y_onehot
