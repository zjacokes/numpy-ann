"""
Floating-point precision experiment: float32 vs float64.

Trains the same model with identical hyperparameters and random seeds in
both float32 and float64, reporting wall-clock training time and final
validation accuracy for each.

This directly addresses the extra-credit item in the assignment rubric:
"you may consider using varying floating-point precisions for
computational efficiency."

The expected result: float32 trains roughly 2x faster (NumPy's BLAS
operations on float32 are substantially faster on most hardware) with
negligible accuracy difference on this task, since the gradient signal is
nowhere near the precision limit of float32.

Usage:
    python experiments/precision_comparison.py [--epochs 20] [--subset 10000]

The default is a short run (20 epochs, 10k subset) so the experiment
completes in minutes. For final reported results, run with --epochs 100
and --subset 0 (full dataset).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data import load_label_names, load_dataset, split_dataset, preprocess, augment_batch, onehot
from model import ThreeLayerMLP
from ops import cross_entropy_loss
from optim import Adam
from schedule import cosine_decay


WNIDS_PATH = "tiny-imagenet-200/wnids.txt"
TRAIN_DIR = "tiny-imagenet-200/train"
INPUT_SIZE = 12288
OUTPUT_SIZE = 200
L2_LAMBDA = 0.001


def run_one(
    dtype: np.dtype,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> dict:
    """Train one full run and return timing + accuracy results."""
    X_tr = X_train.astype(dtype)
    Y_tr = Y_train.astype(dtype)
    X_v = X_val.astype(dtype)
    Y_v = Y_val.astype(dtype)

    model = ThreeLayerMLP(
        input_size=INPUT_SIZE,
        hidden_size_1=512,
        hidden_size_2=256,
        hidden_size_3=128,
        output_size=OUTPUT_SIZE,
        dropout_p=0.3,
        dtype=dtype,
        seed=seed,
    )
    optimizer = Adam(model.params, lr=lr)
    rng = model.rng
    n = len(X_tr)
    n_batches = n // batch_size

    val_accs = []
    t_start = time.time()

    for epoch in range(epochs):
        current_lr = cosine_decay(lr, epoch, epochs)
        idx = rng.permutation(n)

        for i in range(n_batches):
            batch_idx = idx[i * batch_size: (i + 1) * batch_size]
            X_batch = augment_batch(X_tr[batch_idx].copy(), rng)
            Y_batch = Y_tr[batch_idx]
            model.forward(X_batch, training=True)
            grads = model.backward(Y_batch, l2_lambda=L2_LAMBDA)
            optimizer.step(grads, lr=current_lr)

        val_pred = model.predict(X_v)
        val_acc = float(np.mean(
            np.argmax(val_pred, axis=1) == np.argmax(Y_v, axis=1)
        ))
        val_accs.append(val_acc)
        print(f"  [{dtype}] Epoch {epoch + 1}/{epochs} - Val Acc: {100 * val_acc:.2f}%", flush=True)

    wall_time = time.time() - t_start

    return {
        "dtype": str(dtype),
        "wall_time_s": wall_time,
        "best_val_acc": max(val_accs),
        "final_val_acc": val_accs[-1],
        "val_accs": val_accs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare float32 vs float64 training speed and accuracy."
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--subset", type=int, default=10000,
        help="Number of training examples (0 = full dataset)."
    )
    args = parser.parse_args()

    print("Loading dataset...")
    _, label2index = load_label_names(WNIDS_PATH)
    X_raw, Y_raw = load_dataset(TRAIN_DIR, label2index)

    X_train_raw, Y_train, X_val_raw, Y_val, _, _ = split_dataset(
        X_raw, Y_raw, val_frac=0.1, test_frac=0.1, seed=args.seed
    )

    if args.subset and args.subset < len(X_train_raw):
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(X_train_raw))[: args.subset]
        X_train_raw = X_train_raw[idx]
        Y_train = Y_train[idx]
        print(f"Using {args.subset}-image subset for speed.")

    X_train_f64 = preprocess(X_train_raw, dtype=np.float64)
    X_val_f64 = preprocess(X_val_raw, dtype=np.float64)
    Y_train_oh = onehot(Y_train, OUTPUT_SIZE)
    Y_val_oh = onehot(Y_val, OUTPUT_SIZE)

    results = {}
    for dtype in (np.float64, np.float32):
        dtype_name = "float64" if dtype == np.float64 else "float32"
        print(f"\n--- Running {dtype_name} ---")
        results[dtype_name] = run_one(
            dtype=dtype,
            X_train=X_train_f64,
            Y_train=Y_train_oh,
            X_val=X_val_f64,
            Y_val=Y_val_oh,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )

    # Summary table
    f64 = results["float64"]
    f32 = results["float32"]
    speedup = f64["wall_time_s"] / f32["wall_time_s"]

    print("\n" + "=" * 55)
    print(f"{'':20} {'float64':>15} {'float32':>15}")
    print("-" * 55)
    print(f"{'Training time (s)':<20} {f64['wall_time_s']:>15.1f} {f32['wall_time_s']:>15.1f}")
    print(f"{'Speedup':<20} {'1.00x':>15} {speedup:>14.2f}x")
    print(f"{'Best val acc (%)':<20} {100*f64['best_val_acc']:>15.2f} {100*f32['best_val_acc']:>15.2f}")
    print(f"{'Final val acc (%)':<20} {100*f64['final_val_acc']:>15.2f} {100*f32['final_val_acc']:>15.2f}")
    print("=" * 55)
    print(f"\nConclusion: float32 is {speedup:.1f}x faster with "
          f"{abs(f64['best_val_acc'] - f32['best_val_acc']) * 100:.2f}pp "
          f"accuracy difference.")


if __name__ == "__main__":
    main()
