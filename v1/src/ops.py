"""
Element-wise operations and loss functions used by the network.

All functions operate on numpy arrays only -- the assignment constraint
for this project was a from-scratch implementation without autograd.
"""

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation: max(0, x), elementwise."""
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU with respect to its input. 1 where x > 0, else 0."""
    return (x > 0).astype(float)


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Numerically-stable softmax along axis=1.

    Subtracts the per-row max before exponentiating to prevent overflow,
    and clips the shifted values to [-20, 20] for additional safety against
    underflow in extreme cases.
    """
    shifted = np.clip(x - np.max(x, axis=1, keepdims=True), -20, 20)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model,
    l2_lambda: float,
) -> float:
    """
    Cross-entropy loss with L2 regularization on all weight matrices.

    Args:
        y_true: One-hot labels, shape (n_samples, n_classes).
        y_pred: Predicted probabilities, shape (n_samples, n_classes).
        model: Model instance with weight attributes W1..W4.
        l2_lambda: L2 regularization strength.

    Returns:
        Scalar loss value (cross-entropy term + L2 penalty term).
    """
    n_samples = y_true.shape[0]
    ce_loss = -np.sum(np.log(y_pred) * y_true) / n_samples
    l2_loss = 0.5 * l2_lambda * (
        np.sum(np.square(model.W1))
        + np.sum(np.square(model.W2))
        + np.sum(np.square(model.W3))
        + np.sum(np.square(model.W4))
    )
    return ce_loss + l2_loss
