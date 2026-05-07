"""
Learning rate schedules.

Implemented as simple functions rather than classes — a schedule maps
(initial_lr, current_epoch, total_epochs) to a scalar, which is then
passed to optimizer.step(). This keeps the optimizer and schedule
independent.
"""

import numpy as np


def cosine_decay(
    initial_lr: float,
    epoch: int,
    total_epochs: int,
    min_lr: float = 1e-6,
) -> float:
    """
    Cosine annealing schedule (Loshchilov & Hutter, 2017).

    Smoothly decays the learning rate from `initial_lr` to `min_lr`
    following a half-cosine curve. Compared to step decay, cosine
    annealing avoids abrupt drops and tends to produce more stable
    late-training convergence.

    Args:
        initial_lr: Starting learning rate (at epoch 0).
        epoch: Current epoch (0-indexed).
        total_epochs: Total number of planned training epochs.
        min_lr: Floor value (learning rate never drops below this).

    Returns:
        Scheduled learning rate for the current epoch.
    """
    cos_term = np.cos(np.pi * epoch / total_epochs)
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos_term)
