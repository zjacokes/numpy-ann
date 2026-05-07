"""
Element-wise operations, loss functions, and regularization used by the network.

All functions operate on NumPy arrays only. Dropout is implemented here
rather than in the model so it can be tested independently and reused.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation: max(0, x), elementwise."""
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU with respect to its pre-activation input."""
    return (x > 0).astype(x.dtype)


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax along axis=1.

    Subtracts the per-row max before exponentiating to prevent overflow.
    """
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------

def dropout_forward(
    x: np.ndarray,
    p: float,
    training: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inverted dropout forward pass.

    During training, randomly zeros units with probability `p` and scales
    surviving activations by 1/(1-p) so that no rescaling is needed at
    inference time.

    Args:
        x: Input activations.
        p: Drop probability (fraction of units to zero out).
        training: If False, returns x unchanged with a ones mask.
        rng: NumPy random Generator instance (for reproducibility).

    Returns:
        out: Activations after dropout.
        mask: Binary mask used (needed for the backward pass).
    """
    if not training or p == 0.0:
        return x, np.ones_like(x)
    mask = (rng.random(x.shape) > p).astype(x.dtype) / (1.0 - p)
    return x * mask, mask


def dropout_backward(dy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Backward pass through dropout: gate the gradient with the same mask."""
    return dy * mask


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def cross_entropy_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Mean cross-entropy loss over a batch.

    Args:
        y_true: One-hot labels, shape (n_samples, n_classes).
        y_pred: Predicted probabilities, shape (n_samples, n_classes).

    Returns:
        Scalar loss value.

    Note:
        L2 regularization is intentionally excluded here so that the
        reported loss is comparable between train and validation sets.
        The optimizer applies weight decay directly to the gradients.
    """
    n = y_true.shape[0]
    # Clip to avoid log(0); 1e-15 is well within float64 range
    log_probs = np.log(np.clip(y_pred, 1e-15, 1.0))
    return -np.sum(y_true * log_probs) / n
