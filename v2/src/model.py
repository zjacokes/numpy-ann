"""
Three-layer feedforward neural network with dropout.

The key architectural change from V1: the model is responsible only for
the forward and backward passes. It knows nothing about optimizers or
learning rates. Weight updates are applied externally by an optimizer
instance, which makes the Adam vs NAG comparison clean and testable.

Architecture (unchanged from V1):
    input (12288) -> Dense(512) -> ReLU -> Dropout
                  -> Dense(256) -> ReLU -> Dropout
                  -> Dense(128) -> ReLU -> Dropout
                  -> Dense(200) -> Softmax
"""

import numpy as np

from ops import relu, relu_derivative, softmax, dropout_forward, dropout_backward


class ThreeLayerMLP:
    """
    Three-layer fully-connected network with He initialization and dropout.

    Parameters are stored as plain NumPy arrays. Gradients are computed in
    `backward()` and returned as a dict so any optimizer can consume them
    without the model needing to know which one is in use.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size_1: int,
        hidden_size_2: int,
        hidden_size_3: int,
        output_size: int,
        dropout_p: float = 0.3,
        dtype: np.dtype = np.float64,
        seed: int = 42,
    ):
        self.dropout_p = dropout_p
        self.dtype = dtype
        self.rng = np.random.default_rng(seed)

        # He initialization: W ~ N(0, sqrt(2 / fan_in))
        def he(fan_in, fan_out):
            return (
                self.rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)
            ).astype(dtype)

        self.params = {
            "W1": he(input_size, hidden_size_1),
            "b1": np.zeros((1, hidden_size_1), dtype=dtype),
            "W2": he(hidden_size_1, hidden_size_2),
            "b2": np.zeros((1, hidden_size_2), dtype=dtype),
            "W3": he(hidden_size_2, hidden_size_3),
            "b3": np.zeros((1, hidden_size_3), dtype=dtype),
            "W4": he(hidden_size_3, output_size),
            "b4": np.zeros((1, output_size), dtype=dtype),
        }

        # Cache for forward-pass intermediates (populated during forward())
        self._cache: dict = {}

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through the network.

        Caches all intermediate values needed for backprop. When
        `training=False`, dropout masks are ones (no units dropped).
        """
        p = self.params

        Z1 = X @ p["W1"] + p["b1"]
        A1 = relu(Z1)
        A1d, mask1 = dropout_forward(A1, self.dropout_p, training, self.rng)

        Z2 = A1d @ p["W2"] + p["b2"]
        A2 = relu(Z2)
        A2d, mask2 = dropout_forward(A2, self.dropout_p, training, self.rng)

        Z3 = A2d @ p["W3"] + p["b3"]
        A3 = relu(Z3)
        A3d, mask3 = dropout_forward(A3, self.dropout_p, training, self.rng)

        Z4 = A3d @ p["W4"] + p["b4"]
        A4 = softmax(Z4)

        self._cache = {
            "X": X,
            "Z1": Z1, "A1": A1, "A1d": A1d, "mask1": mask1,
            "Z2": Z2, "A2": A2, "A2d": A2d, "mask2": mask2,
            "Z3": Z3, "A3": A3, "A3d": A3d, "mask3": mask3,
            "A4": A4,
        }
        return A4

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self, y_true: np.ndarray, l2_lambda: float = 0.001) -> dict:
        """
        Backpropagation through the network.

        Computes gradients of cross-entropy loss with respect to all
        parameters. L2 gradient term is added to weight gradients here
        so the optimizer receives the regularized gradient directly.

        Returns a dict with the same keys as `self.params`.
        """
        c = self._cache
        p = self.params
        m = y_true.shape[0]

        # Output layer: softmax + cross-entropy gradient simplifies to (A4 - y)
        dZ4 = (c["A4"] - y_true) / m
        dW4 = c["A3d"].T @ dZ4 + l2_lambda * p["W4"]
        db4 = dZ4.sum(axis=0, keepdims=True)

        # Hidden layer 3
        dA3d = dZ4 @ p["W4"].T
        dA3 = dropout_backward(dA3d, c["mask3"])
        dZ3 = dA3 * relu_derivative(c["Z3"])
        dW3 = c["A2d"].T @ dZ3 + l2_lambda * p["W3"]
        db3 = dZ3.sum(axis=0, keepdims=True)

        # Hidden layer 2
        dA2d = dZ3 @ p["W3"].T
        dA2 = dropout_backward(dA2d, c["mask2"])
        dZ2 = dA2 * relu_derivative(c["Z2"])
        dW2 = c["A1d"].T @ dZ2 + l2_lambda * p["W2"]
        db2 = dZ2.sum(axis=0, keepdims=True)

        # Hidden layer 1
        dA1d = dZ2 @ p["W2"].T
        dA1 = dropout_backward(dA1d, c["mask1"])
        dZ1 = dA1 * relu_derivative(c["Z1"])
        dW1 = c["X"].T @ dZ1 + l2_lambda * p["W1"]
        db1 = dZ1.sum(axis=0, keepdims=True)

        return {
            "W1": dW1, "b1": db1,
            "W2": dW2, "b2": db2,
            "W3": dW3, "b3": db3,
            "W4": dW4, "b4": db4,
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return softmax probabilities at inference time (dropout disabled)."""
        return self.forward(X, training=False)

    def accuracy(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """Top-1 accuracy over a batch."""
        y_pred = self.predict(X)
        return float(np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)))
