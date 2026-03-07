"""
Loss functions used during training and their corresponding gradients.
Currently includes cross-entropy and mean squared error.
"""

from typing import Tuple
import numpy as np

from .activations import softmax


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error loss."""
    return np.mean((y_pred - y_true) ** 2)


def mse_grad(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Gradient of MSE loss."""
    batch_size = y_true.shape[0]
    return 2.0 * (y_pred - y_true) / batch_size


def cross_entropy_loss(
    y_true: np.ndarray,
    logits: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[float, np.ndarray]:
    """
    Cross-entropy loss for multi-class classification.
    Softmax is applied internally to convert logits into probabilities.
    """

    probs = softmax(logits)

    log_probs = np.log(probs + eps)

    loss = -np.mean(np.sum(y_true * log_probs, axis=1))

    return loss, probs


def cross_entropy_grad(y_true: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """
    Gradient of cross-entropy loss with respect to logits.
    Assumes probabilities are already obtained from softmax.
    """

    batch_size = y_true.shape[0]

    return (probs - y_true) / batch_size