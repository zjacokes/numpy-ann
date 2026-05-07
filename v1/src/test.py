"""
Test-set evaluation entry point.

Loads pre-trained weights, runs them against a labelled test set, and
prints the top-1 accuracy. This is the script that was submitted as the
deliverable for the original assignment.

Usage:
    python test.py <imagedir> <labelfile>
"""

import argparse

import numpy as np

from data import load_label_names, load_test_images, preprocess
from model import HeThreeLayerNN
from weights import load_weights


# Architecture and training hyperparameters (must match the training
# configuration used to produce model_weights.json).
INPUT_SIZE = 12288  # 64 * 64 * 3
HIDDEN_SIZE_1 = 512
HIDDEN_SIZE_2 = 256
HIDDEN_SIZE_3 = 128
OUTPUT_SIZE = 200
LEARNING_RATE = 0.001
L2_LAMBDA = 0.001

WNIDS_PATH = "tiny-imagenet-200/wnids.txt"
WEIGHTS_PATH = "model_weights.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="My Image Classifier",
        description="Zach Jacokes - UVA SDS DS:6210 Final Project",
    )
    parser.add_argument("imagedir", type=str, help="Test image directory path.")
    parser.add_argument("labelfile", type=str, help="Test annotation file path.")
    args = parser.parse_args()

    # Load class labels and the test set
    _, label2index = load_label_names(WNIDS_PATH)
    X, Y = load_test_images(args.imagedir, args.labelfile, label2index)
    X, Y = preprocess(X, Y, n_classes=OUTPUT_SIZE)

    # Build the model and load trained weights
    model = HeThreeLayerNN(
        input_size=INPUT_SIZE,
        hidden_size_1=HIDDEN_SIZE_1,
        hidden_size_2=HIDDEN_SIZE_2,
        hidden_size_3=HIDDEN_SIZE_3,
        output_size=OUTPUT_SIZE,
        learning_rate=LEARNING_RATE,
        l2_lambda=L2_LAMBDA,
    )
    load_weights(model, WEIGHTS_PATH)

    # Predict and report top-1 accuracy
    y_pred = model.predict(X)
    y_true_idx = np.argmax(Y, axis=1)
    y_pred_idx = np.argmax(y_pred, axis=1)

    accuracy = np.mean(y_pred_idx == y_true_idx)
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
