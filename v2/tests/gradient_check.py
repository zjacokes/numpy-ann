"""
Numerical gradient check using Autograd.

This is the test that would have caught the V1 Adam bug immediately.
It verifies that the hand-written backward pass produces gradients that
agree with numerically-computed finite differences to within a tight
tolerance.

Autograd is used here to compute reference gradients independently of
the NumPy backward pass, which is the legitimate use of the library for
this assignment. The check runs on a tiny synthetic dataset (5 examples,
10 classes) so it completes in seconds.

Usage:
    python tests/gradient_check.py

A passing run prints the relative error for each parameter and confirms
that all errors are below 1e-5. A failing run means the backward pass
has a bug.
"""

import sys
from pathlib import Path

import numpy as np
import autograd.numpy as anp
from autograd import grad

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import ThreeLayerMLP
from ops import cross_entropy_loss


# ---------------------------------------------------------------------------
# Autograd reference implementation
# ---------------------------------------------------------------------------

def autograd_loss(params: dict, X: np.ndarray, Y: np.ndarray) -> float:
    """
    Cross-entropy loss implemented in Autograd numpy so that grad() can
    differentiate through it automatically.

    This is intentionally a simple forward pass with no dropout or L2 so
    the gradient check is clean and unambiguous.
    """
    def relu(x):
        return anp.maximum(0, x)

    def softmax(x):
        shifted = x - anp.max(x, axis=1, keepdims=True)
        exp_x = anp.exp(shifted)
        return exp_x / anp.sum(exp_x, axis=1, keepdims=True)

    Z1 = X @ params["W1"] + params["b1"]
    A1 = relu(Z1)
    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = relu(Z2)
    Z3 = A2 @ params["W3"] + params["b3"]
    A3 = relu(Z3)
    Z4 = A3 @ params["W4"] + params["b4"]
    A4 = softmax(Z4)

    n = Y.shape[0]
    log_probs = anp.log(anp.clip(A4, 1e-15, 1.0))
    return -anp.sum(Y * log_probs) / n


# ---------------------------------------------------------------------------
# Relative error helper
# ---------------------------------------------------------------------------

def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """
    Standard relative error metric for gradient checks.
    Near zero when the two gradients agree, near 1 when they disagree.
    """
    return float(np.max(np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-15)))


# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------

def run_gradient_check(seed: int = 1) -> bool:
    """
    Run the gradient check. Returns True if all parameters pass.
    """
    rng = np.random.default_rng(seed)

    # Tiny problem: 5 examples, 10 features, 4 classes
    # Small enough to be fast; large enough to exercise all layers
    n_samples, n_features, n_classes = 5, 10, 4
    X = rng.standard_normal((n_samples, n_features))
    Y = np.zeros((n_samples, n_classes))
    Y[np.arange(n_samples), rng.integers(0, n_classes, n_samples)] = 1.0

    model = ThreeLayerMLP(
        input_size=n_features,
        hidden_size_1=8,
        hidden_size_2=6,
        hidden_size_3=4,
        output_size=n_classes,
        dropout_p=0.0,   # dropout off for clean gradient check
        seed=seed,
    )

    # --- Analytical gradients from our backward pass ---
    model.forward(X, training=False)
    analytical_grads = model.backward(Y, l2_lambda=0.0)

    # --- Reference gradients from Autograd ---
    grad_fn = grad(autograd_loss)
    autograd_grads = grad_fn(model.params, X, Y)

    # --- Compare ---
    print("Gradient check results:")
    print(f"{'Parameter':<8} {'Relative error':<20} {'Status'}")
    print("-" * 45)

    all_passed = True
    threshold = 1e-5

    for key in model.params:
        err = relative_error(analytical_grads[key], autograd_grads[key])
        status = "PASS" if err < threshold else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"{key:<8} {err:<20.2e} {status}")

    print()
    if all_passed:
        print("All gradients passed. Backward pass is correct.")
    else:
        print("One or more gradients FAILED. Check the backward pass.")

    return all_passed


if __name__ == "__main__":
    passed = run_gradient_check()
    sys.exit(0 if passed else 1)
