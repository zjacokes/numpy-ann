"""
Training entry point for the three-layer neural network.

Loads Tiny ImageNet, trains the model with the same hyperparameters used
to produce the original assignment submission, and saves weights to JSON.

Usage:
    python train.py [--epochs 100] [--batch-size 50] [--subset 10000]
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from data import load_label_names, onehot
from model import HeThreeLayerNN
from ops import cross_entropy_loss
from weights import save_weights


# Architecture and training hyperparameters
INPUT_SIZE = 12288
HIDDEN_SIZE_1 = 512
HIDDEN_SIZE_2 = 256
HIDDEN_SIZE_3 = 128
OUTPUT_SIZE = 200
LEARNING_RATE = 0.001
L2_LAMBDA = 0.001

WNIDS_PATH = "tiny-imagenet-200/wnids.txt"
TRAIN_DIR = "tiny-imagenet-200/train"


def load_training_set(train_dir: str, label2index: dict) -> tuple:
    """
    Walk the Tiny ImageNet train directory and load all images.

    The directory is structured as:
        train/<wnid>/images/<wnid>_<i>.JPEG

    Returns:
        X: Image array of shape (n, 64, 64, 3).
        Y: Integer labels of shape (n,).
    """
    paths = Path(train_dir).glob("*/images/*.*")
    images = []
    labels = []
    for path in paths:
        labelname = path.parent.parent.stem
        label = label2index[labelname]
        with Image.open(path) as image:
            images.append(np.array(image.convert("RGB")))
            labels.append(label)
    return np.array(images), np.array(labels)


def train_test_split(
    X: np.ndarray,
    Y: np.ndarray,
    train_frac: float = 0.8,
    seed: int = 42,
) -> tuple:
    """Shuffle and split the dataset into train and test partitions."""
    np.random.seed(seed)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split = int(train_frac * X.shape[0])
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], Y[train_idx], X[test_idx], Y[test_idx]


def train(
    model: HeThreeLayerNN,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    epochs: int,
    batch_size: int,
) -> dict:
    """
    Train the model with mini-batch updates for the given number of epochs.

    Each epoch walks through the training set once in fixed batch order,
    calling forward + backward and then the Adam moment updater. After
    each epoch, full-train and full-test loss and accuracy are recorded.
    """
    n_samples = X_train.shape[0]
    n_batches = n_samples // batch_size

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for epoch in range(epochs):
        for i in range(n_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            X_batch = X_train[start:end]
            y_batch = Y_train[start:end]

            model.forward(X_batch)
            grads = model.backward(X_batch, y_batch)
            model.update_parameters_adam(*grads, t=epoch * n_batches + i + 1)

        # End-of-epoch evaluation
        y_pred_train = model.forward(X_train)
        train_loss = cross_entropy_loss(Y_train, y_pred_train, model, model.l2_lambda)
        train_acc = np.mean(np.argmax(Y_train, axis=1) == np.argmax(y_pred_train, axis=1))

        y_pred_test = model.forward(X_test)
        test_loss = cross_entropy_loss(Y_test, y_pred_test, model, model.l2_lambda)
        test_acc = np.mean(np.argmax(Y_test, axis=1) == np.argmax(y_pred_test, axis=1))

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {100 * train_acc:.2f}%, "
            f"Test Loss: {test_loss:.4f}, Test Acc: {100 * test_acc:.2f}%"
        )

    return history


def plot_history(history: dict, save_path: str = "training_curves.png") -> None:
    """Plot training and test loss/accuracy curves and save to disk."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(history["train_loss"], label="Train Loss")
    ax_loss.plot(history["test_loss"], label="Test Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()

    ax_acc.plot(history["train_acc"], label="Train Accuracy")
    ax_acc.plot(history["test_acc"], label="Test Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()

    fig.suptitle("Three-Layer Neural Network", fontsize=16)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    print(f"Saved training curves to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the three-layer NN on Tiny ImageNet.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument(
        "--subset",
        type=int,
        default=10000,
        help="Number of training examples to use (use 0 for the full set).",
    )
    parser.add_argument("--weights-out", type=str, default="model_weights.json")
    args = parser.parse_args()

    _, label2index = load_label_names(WNIDS_PATH)
    print("Loading training images...")
    X, Y = load_training_set(TRAIN_DIR, label2index)
    print(f"Loaded {X.shape[0]} images of shape {X.shape[1:]}")

    X_train, Y_train, X_test, Y_test = train_test_split(X, Y)

    # Flatten and normalize to [-1, 1]; one-hot encode labels
    h, w, c = X_train.shape[1:]
    X_train = (np.reshape(X_train, (-1, h * w * c)) / 128.0) - 1
    X_test = (np.reshape(X_test, (-1, h * w * c)) / 128.0) - 1
    Y_train = onehot(Y_train, OUTPUT_SIZE)
    Y_test = onehot(Y_test, OUTPUT_SIZE)

    # Optionally subset to speed up training
    if args.subset and args.subset < X_train.shape[0]:
        train_idx = np.random.permutation(X_train.shape[0])[: args.subset]
        test_idx = np.random.permutation(X_test.shape[0])[: max(args.subset // 5, 1)]
        X_train, Y_train = X_train[train_idx], Y_train[train_idx]
        X_test, Y_test = X_test[test_idx], Y_test[test_idx]
        print(f"Using subset: {X_train.shape[0]} train, {X_test.shape[0]} test")

    model = HeThreeLayerNN(
        input_size=INPUT_SIZE,
        hidden_size_1=HIDDEN_SIZE_1,
        hidden_size_2=HIDDEN_SIZE_2,
        hidden_size_3=HIDDEN_SIZE_3,
        output_size=OUTPUT_SIZE,
        learning_rate=LEARNING_RATE,
        l2_lambda=L2_LAMBDA,
    )

    history = train(
        model, X_train, Y_train, X_test, Y_test,
        epochs=args.epochs, batch_size=args.batch_size,
    )

    save_weights(model, args.weights_out)
    print(f"Saved trained weights to {args.weights_out}")
    plot_history(history)


if __name__ == "__main__":
    main()
