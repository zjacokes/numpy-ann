"""
Three-layer feedforward neural network implemented from scratch in NumPy.

Architecture: input -> [hidden_1] -> [hidden_2] -> [hidden_3] -> output
Activations: ReLU on hidden layers, softmax on output.
Initialization: He initialization (sqrt(2/fan_in)).
Optimizer: Adam (with L2 regularization).
"""

import numpy as np

from ops import relu, relu_derivative, softmax


class HeThreeLayerNN:
    """
    Three-layer fully-connected network with He initialization and Adam.

    Trained with cross-entropy loss and L2 regularization on the weights.
    All forward and backward passes are implemented manually using only
    NumPy primitives.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size_1: int,
        hidden_size_2: int,
        hidden_size_3: int,
        output_size: int,
        learning_rate: float,
        l2_lambda: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        # He initialization: weights ~ N(0, 2/fan_in)
        self.W1 = np.random.randn(input_size, hidden_size_1) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size_1))
        self.W2 = np.random.randn(hidden_size_1, hidden_size_2) * np.sqrt(2.0 / hidden_size_1)
        self.b2 = np.zeros((1, hidden_size_2))
        self.W3 = np.random.randn(hidden_size_2, hidden_size_3) * np.sqrt(2.0 / hidden_size_2)
        self.b3 = np.zeros((1, hidden_size_3))
        self.W4 = np.random.randn(hidden_size_3, output_size) * np.sqrt(2.0 / hidden_size_3)
        self.b4 = np.zeros((1, output_size))

        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Adam first-moment (mean) running estimates, one per parameter
        self.mW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2)
        self.mW3 = np.zeros_like(self.W3)
        self.mb3 = np.zeros_like(self.b3)
        self.mW4 = np.zeros_like(self.W4)
        self.mb4 = np.zeros_like(self.b4)

        # Adam second-moment (variance) running estimates
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)
        self.vW3 = np.zeros_like(self.W3)
        self.vb3 = np.zeros_like(self.b3)
        self.vW4 = np.zeros_like(self.W4)
        self.vb4 = np.zeros_like(self.b4)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.

        Caches intermediate activations on `self` for use in the backward
        pass. Returns softmax probabilities over the output classes.
        """
        self.A0 = X
        self.Z1 = np.dot(self.A0, self.W1) + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = relu(self.Z2)
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = relu(self.Z3)
        self.Z4 = np.dot(self.A3, self.W4) + self.b4
        self.A4 = softmax(self.Z4)
        return self.A4

    def backward(self, X: np.ndarray, y_true: np.ndarray) -> tuple:
        """
        Backpropagation through the network.

        Computes gradients of the cross-entropy + L2 loss with respect to
        all weights and biases, applies the parameter update, and returns
        the gradients (so they can be passed to the Adam moment updater).
        """
        m = X.shape[0]

        # Output layer
        dZ4 = self.A4 - y_true
        dW4 = np.dot(self.A3.T, dZ4) / m
        db4 = np.sum(dZ4, axis=0, keepdims=True) / m

        # Hidden layer 3
        dA3 = np.dot(dZ4, self.W4.T)
        dZ3 = dA3 * relu_derivative(self.Z3)
        dW3 = np.dot(self.A2.T, dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        # Hidden layer 2
        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * relu_derivative(self.Z2)
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Hidden layer 1
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = np.dot(self.A0.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Add L2 gradient term to weight gradients
        dW1 += self.l2_lambda * self.W1
        dW2 += self.l2_lambda * self.W2
        dW3 += self.l2_lambda * self.W3
        dW4 += self.l2_lambda * self.W4

        # Parameter update
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
        self.W4 -= self.learning_rate * dW4
        self.b4 -= self.learning_rate * db4

        return dW1, db1, dW2, db2, dW3, db3, dW4, db4

    def update_parameters_adam(
        self,
        dW1, db1, dW2, db2, dW3, db3, dW4, db4,
        t: int,
    ) -> None:
        """
        Update Adam first- and second-moment estimates and compute
        bias-corrected versions of each.

        `t` is the global step count (1-indexed) used for bias correction.
        """
        # First-moment (mean) updates
        self.mW1 = self.beta1 * self.mW1 + (1 - self.beta1) * dW1
        self.mb1 = self.beta1 * self.mb1 + (1 - self.beta1) * db1
        self.mW2 = self.beta1 * self.mW2 + (1 - self.beta1) * dW2
        self.mb2 = self.beta1 * self.mb2 + (1 - self.beta1) * db2
        self.mW3 = self.beta1 * self.mW3 + (1 - self.beta1) * dW3
        self.mb3 = self.beta1 * self.mb3 + (1 - self.beta1) * db3
        self.mW4 = self.beta1 * self.mW4 + (1 - self.beta1) * dW4
        self.mb4 = self.beta1 * self.mb4 + (1 - self.beta1) * db4

        # Second-moment (variance) updates
        self.vW1 = self.beta2 * self.vW1 + (1 - self.beta2) * (dW1 ** 2)
        self.vb1 = self.beta2 * self.vb1 + (1 - self.beta2) * (db1 ** 2)
        self.vW2 = self.beta2 * self.vW2 + (1 - self.beta2) * (dW2 ** 2)
        self.vb2 = self.beta2 * self.vb2 + (1 - self.beta2) * (db2 ** 2)
        self.vW3 = self.beta2 * self.vW3 + (1 - self.beta2) * (dW3 ** 2)
        self.vb3 = self.beta2 * self.vb3 + (1 - self.beta2) * (db3 ** 2)
        self.vW4 = self.beta2 * self.vW4 + (1 - self.beta2) * (dW4 ** 2)
        self.vb4 = self.beta2 * self.vb4 + (1 - self.beta2) * (db4 ** 2)

        # Bias-corrected first-moment estimates
        mW1_corr = self.mW1 / (1 - self.beta1 ** t)
        mb1_corr = self.mb1 / (1 - self.beta1 ** t)
        mW2_corr = self.mW2 / (1 - self.beta1 ** t)
        mb2_corr = self.mb2 / (1 - self.beta1 ** t)
        mW3_corr = self.mW3 / (1 - self.beta1 ** t)
        mb3_corr = self.mb3 / (1 - self.beta1 ** t)
        mW4_corr = self.mW4 / (1 - self.beta1 ** t)
        mb4_corr = self.mb4 / (1 - self.beta1 ** t)

        # Bias-corrected second-moment estimates
        vW1_corr = self.vW1 / (1 - self.beta2 ** t)
        vb1_corr = self.vb1 / (1 - self.beta2 ** t)
        vW2_corr = self.vW2 / (1 - self.beta2 ** t)
        vb2_corr = self.vb2 / (1 - self.beta2 ** t)
        vW3_corr = self.vW3 / (1 - self.beta2 ** t)
        vb3_corr = self.vb3 / (1 - self.beta2 ** t)
        vW4_corr = self.vW4 / (1 - self.beta2 ** t)
        vb4_corr = self.vb4 / (1 - self.beta2 ** t)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return softmax probabilities for input batch X."""
        return self.forward(X)
