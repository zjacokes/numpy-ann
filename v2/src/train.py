"""
Training entry point for V2.

New in V2 vs V1:
- Optimizer selectable via --optimizer flag (adam or nag).
- Cosine learning rate schedule.
- Per-epoch index reshuffling.
- Horizontal flip augmentation per batch.
- Proper train / val / test split; early stopping on val loss.
- Best-val weights restored at end of training.
- L2 excluded from reported loss (applied in backward via gradient).
- Batch size increased to 128 (more stable gradients on full dataset).

Usage:
    python train.py --optimizer adam --epochs 150 --batch-size 128
    python train.py --optimizer nag  --epochs 150 --batch-size 128
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Make src importable when run from the repo root
sys.path.insert(0, str(Path(__file__).parent))

from data import load_label_names, load_dataset, split_dataset, preprocess, augment_batch, onehot
from model import ThreeLayerMLP
from ops import cross_entropy_loss
from optim import Adam, NAG
from schedule import cosine_decay
from weights import save_weights


# ---------------------------------------------------------------------------
# Architecture constants (match V1 for fair comparison)
# ---------------------------------------------------------------------------
INPUT_SIZE = 12288
HIDDEN_SIZE_1 = 512
HIDDEN_SIZE_2 = 256
HIDDEN_SIZE_3 = 128
OUTPUT_SIZE = 200
L2_LAMBDA = 0.001
DROPOUT_P = 0.3

WNIDS_PATH = "tiny-imagenet-200/wnids.txt"
TRAIN_DIR = "tiny-imagenet-200/train"


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: ThreeLayerMLP,
    optimizer,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    initial_lr: float,
    patience: int,
    weights_out: str,
) -> dict:
    """
    Train the model with mini-batch updates, early stopping, and LR decay.

    Shuffles training indices at the start of each epoch. Applies
    horizontal flip augmentation to each batch. Evaluates on the
    validation set after each epoch and saves best-val weights.
    Restores best weights if early stopping triggers.

    Returns a history dict of per-epoch metrics.
    """
    rng = model.rng
    n = X_train.shape[0]
    n_batches = n // batch_size

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [],   "val_acc": [],
        "lr": [],
    }

    best_val_loss = np.inf
    best_params = copy.deepcopy(model.params)
    epochs_no_improve = 0

    for epoch in range(epochs):
        lr = cosine_decay(initial_lr, epoch, epochs)
        history["lr"].append(lr)

        # Reshuffle training order each epoch (V1 bug fix #2)
        idx = rng.permutation(n)

        epoch_loss = 0.0
        epoch_correct = 0

        for i in range(n_batches):
            batch_idx = idx[i * batch_size: (i + 1) * batch_size]
            X_batch = augment_batch(X_train[batch_idx].copy(), rng)
            Y_batch = Y_train[batch_idx]

            y_pred = model.forward(X_batch, training=True)
            grads = model.backward(Y_batch, l2_lambda=L2_LAMBDA)
            optimizer.step(grads, lr=lr)

            epoch_loss += cross_entropy_loss(Y_batch, y_pred) * len(batch_idx)
            epoch_correct += int(
                np.sum(np.argmax(y_pred, axis=1) == np.argmax(Y_batch, axis=1))
            )

        train_loss = epoch_loss / n
        train_acc = epoch_correct / n

        # Validation (dropout disabled)
        val_pred = model.predict(X_val)
        val_loss = cross_entropy_loss(Y_val, val_pred)
        val_acc = float(np.mean(
            np.argmax(val_pred, axis=1) == np.argmax(Y_val, axis=1)
        ))

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1:>3}/{epochs} | lr={lr:.2e} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {100 * train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {100 * val_acc:.2f}%"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = copy.deepcopy(model.params)
            save_weights(best_params, weights_out)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1} "
                      f"(no val improvement for {patience} epochs).")
                break

    # Restore best weights
    model.params = best_params
    print(f"\nBest val loss: {best_val_loss:.4f}. Best weights restored.")
    return history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_history(history: dict, optimizer_name: str, save_path: str) -> None:
    """Save training/validation loss and accuracy curves to disk."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]], label="Train")
    axes[1].plot(epochs, [a * 100 for a in history["val_acc"]], label="Validation")
    axes[1].set_title("Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(epochs, history["lr"])
    axes[2].set_title("Learning Rate (cosine decay)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_yscale("log")

    fig.suptitle(f"V2 — Three-Layer MLP with {optimizer_name.upper()}", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    print(f"Curves saved to {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train V2 MLP on Tiny ImageNet.")
    parser.add_argument("--optimizer", choices=["adam", "nag"], default="adam")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=DROPOUT_P)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--weights-out", type=str, default=None)
    args = parser.parse_args()

    dtype = np.float32 if args.dtype == "float32" else np.float64
    weights_out = args.weights_out or f"v2_weights_{args.optimizer}.json"

    # Set thread counts before numpy does anything
    import os
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "8")

    print(f"Loading dataset from {TRAIN_DIR}...")
    t0 = time.time()
    _, label2index = load_label_names(WNIDS_PATH)
    X_raw, Y_raw = load_dataset(TRAIN_DIR, label2index)
    print(f"Loaded {len(X_raw)} images in {time.time() - t0:.1f}s")

    X_train_raw, Y_train, X_val_raw, Y_val, X_test_raw, Y_test = split_dataset(
        X_raw, Y_raw, val_frac=0.1, test_frac=0.1, seed=args.seed
    )

    print(
        f"Split: {len(X_train_raw)} train / {len(X_val_raw)} val / {len(X_test_raw)} test"
    )

    X_train = preprocess(X_train_raw, dtype=dtype)
    X_val = preprocess(X_val_raw, dtype=dtype)
    X_test = preprocess(X_test_raw, dtype=dtype)

    Y_train = onehot(Y_train, OUTPUT_SIZE).astype(dtype)
    Y_val = onehot(Y_val, OUTPUT_SIZE).astype(dtype)
    Y_test = onehot(Y_test, OUTPUT_SIZE).astype(dtype)

    model = ThreeLayerMLP(
        input_size=INPUT_SIZE,
        hidden_size_1=HIDDEN_SIZE_1,
        hidden_size_2=HIDDEN_SIZE_2,
        hidden_size_3=HIDDEN_SIZE_3,
        output_size=OUTPUT_SIZE,
        dropout_p=args.dropout,
        dtype=dtype,
        seed=args.seed,
    )

    if args.optimizer == "adam":
        optimizer = Adam(model.params, lr=args.lr)
    else:
        optimizer = NAG(model.params, lr=args.lr, momentum=0.9)

    print(f"\nTraining with {args.optimizer.upper()} | "
          f"dtype={args.dtype} | dropout={args.dropout} | "
          f"lr={args.lr} (cosine decay) | patience={args.patience}\n")

    t_start = time.time()
    history = train(
        model=model,
        optimizer=optimizer,
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        initial_lr=args.lr,
        patience=args.patience,
        weights_out=weights_out,
    )
    wall_time = time.time() - t_start

    # Final test evaluation (held-out set, never seen during training)
    test_pred = model.predict(X_test)
    test_acc = float(np.mean(
        np.argmax(test_pred, axis=1) == np.argmax(Y_test, axis=1)
    ))
    print(f"\nTest accuracy: {100 * test_acc:.2f}%")
    print(f"Total training time: {wall_time / 60:.1f} min")

    curves_path = f"v2_curves_{args.optimizer}_{args.dtype}.png"
    plot_history(history, args.optimizer, curves_path)


if __name__ == "__main__":
    main()
