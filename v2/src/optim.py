"""
Gradient-based optimizers: Adam and Nesterov Accelerated Gradient (NAG).

Both are implemented as standalone classes that operate on a parameter dict
and a gradient dict with matching keys. Neither knows anything about the
model architecture, which makes the Adam vs NAG comparison clean: the
same training loop runs both without modification.

This is the assignment's "two gradient-based optimizers" requirement,
implemented properly. V1 nominally had both but Adam was a no-op.
"""

import numpy as np


class Adam:
    """
    Adam optimizer (Kingma & Ba, 2015).

    Maintains per-parameter first- and second-moment running estimates and
    applies bias-corrected updates at each step.

    Args:
        params: Dict of parameter arrays (modified in place).
        lr: Initial learning rate.
        beta1: Exponential decay for first moment (default 0.9).
        beta2: Exponential decay for second moment (default 0.999).
        epsilon: Numerical stability term (default 1e-8).
    """

    def __init__(
        self,
        params: dict,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # global step counter

        # Initialize moment estimates at zero for each parameter
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads: dict, lr: float | None = None) -> None:
        """
        Apply one Adam update step.

        Args:
            grads: Dict of gradients with the same keys as self.params.
            lr: Optional learning rate override (used by LR scheduler).
        """
        lr = lr if lr is not None else self.lr
        self.t += 1
        t = self.t

        for key in self.params:
            g = grads[key]

            # Update biased moment estimates
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g ** 2)

            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1 ** t)
            v_hat = self.v[key] / (1 - self.beta2 ** t)

            # Parameter update
            self.params[key] -= lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


class NAG:
    """
    Nesterov Accelerated Gradient (Sutskever et al., 2013).

    NAG computes the gradient at the "lookahead" position (current params
    plus the momentum step) rather than at the current params. This gives
    it better theoretical convergence properties on convex problems and
    noticeably different dynamics on non-convex ones.

    In practice for neural nets: NAG tends to converge faster in early
    training but Adam often wins on final accuracy. The comparison between
    the two on this MLP is one of the empirically honest things this
    project can show.

    Implementation uses the standard reformulation that avoids computing
    the lookahead position explicitly:
        v_t = mu * v_{t-1} + lr * grad(params - mu * v_{t-1})
        params = params - v_t

    Since we compute the gradient at current params (not the lookahead),
    we use the equivalent "Bengio" form which applies the correction after
    the gradient is already in hand:
        v_t = mu * v_{t-1} + lr * g
        params -= mu * v_t - (1 - mu) * v_{t-1}  [simplified below]

    Args:
        params: Dict of parameter arrays (modified in place).
        lr: Learning rate.
        momentum: Momentum coefficient (default 0.9).
    """

    def __init__(
        self,
        params: dict,
        lr: float = 0.01,
        momentum: float = 0.9,
    ):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocity = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads: dict, lr: float | None = None) -> None:
        """
        Apply one NAG update step.

        Args:
            grads: Dict of gradients with the same keys as self.params.
            lr: Optional learning rate override (used by LR scheduler).
        """
        lr = lr if lr is not None else self.lr
        mu = self.momentum

        for key in self.params:
            v_prev = self.velocity[key].copy()
            self.velocity[key] = mu * self.velocity[key] + lr * grads[key]
            # Nesterov correction: look ahead by one momentum step
            self.params[key] -= (1 + mu) * self.velocity[key] - mu * v_prev
