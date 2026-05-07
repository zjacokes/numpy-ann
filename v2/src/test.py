"""
Test-set evaluation entry point for V2.

Loads pre-trained weights and evaluates top-1 accuracy on a labelled
image directory. Interface is identical to V1/test.py so both versions
can be evaluated the same way.

Usage:
    python test.py <imagedir> <labelfile> [--weights model_weights.json]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from data import load_label_names, load_test_images, preprocess, onehot
from model import ThreeLayerMLP
from weights import load_weights


INPUT_SIZE = 12288
HIDDEN_SIZE_1 = 512
HIDDEN_SIZE_2 = 256
HIDDEN_SIZE_3 = 128
OUTPUT_SIZE = 200
WNIDS_PATH = "tiny-imagenet-200/wnids.txt"


def load_test_images(
    image_dir: str,
    label_file: str,
    label2index: dict,
):
    """Load test images and labels from a Tiny ImageNet annotation file."""
    import os
    from PIL import Image

    imagepaths, labels = [], []
    with open(label_file) as f:
        for line in f:
            tokens = line.strip().split()
            path = os.path.join(image_dir, tokens[0])
            if os.path.exists(path):
                imagepaths.append(path)
                labels.append(label2index[tokens[1]])
            else:
                print(f"Warning: {path} not found, skipping.")

    X, Y = [], []
    for path, label in zip(imagepaths, labels):
        with Image.open(path) as img:
            X.append(np.array(img.convert("RGB"), dtype=np.uint8))
        Y.append(label)

    return np.array(X), np.array(Y, dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V2 model on a test set.")
    parser.add_argument("imagedir", type=str, help="Test image directory path.")
    parser.add_argument("labelfile", type=str, help="Test annotation file path.")
    parser.add_argument(
        "--weights", type=str, default="v2_weights_adam.json",
        help="Path to trained weights JSON file."
    )
    parser.add_argument(
        "--dtype", choices=["float32", "float64"], default="float64"
    )
    args = parser.parse_args()

    dtype = np.float32 if args.dtype == "float32" else np.float64

    _, label2index = load_label_names(WNIDS_PATH)
    X_raw, Y_raw = load_test_images(args.imagedir, args.labelfile, label2index)
    X = preprocess(X_raw, dtype=dtype)
    Y = onehot(Y_raw, OUTPUT_SIZE).astype(dtype)

    model = ThreeLayerMLP(
        input_size=INPUT_SIZE,
        hidden_size_1=HIDDEN_SIZE_1,
        hidden_size_2=HIDDEN_SIZE_2,
        hidden_size_3=HIDDEN_SIZE_3,
        output_size=OUTPUT_SIZE,
        dropout_p=0.0,  # dropout disabled at inference
        dtype=dtype,
    )
    model.params = load_weights(args.weights, dtype=dtype)

    y_pred = model.predict(X)
    acc = float(np.mean(np.argmax(y_pred, axis=1) == np.argmax(Y, axis=1)))
    print(f"Test accuracy: {100 * acc:.2f}%")


if __name__ == "__main__":
    main()
