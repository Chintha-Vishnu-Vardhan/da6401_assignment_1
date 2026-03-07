"""
Collection of optimization algorithms used for updating network weights.

Each optimizer modifies the parameters of the layers using the gradients
computed during backpropagation.
"""

from typing import Iterable, Dict, Tuple

import numpy as np


class BaseOptimizer:
    """
    Parent class for optimizers.
    Stores common parameters like learning rate and weight decay.
    """

    def __init__(self, learning_rate: float = 1e-3, weight_decay: float = 0.0) -> None:
        self.lr = float(learning_rate)
        self.weight_decay = float(weight_decay)

    def _apply_weight_decay(self, W: np.ndarray) -> np.ndarray:
        """Return L2 penalty term if weight decay is enabled."""
        if self.weight_decay <= 0.0:
            return 0.0
        return self.weight_decay * W

    def step(self, layers: Iterable) -> None:
        """Update parameters for all layers."""
        raise NotImplementedError


class SGD(BaseOptimizer):
    """
    Standard stochastic gradient descent.
    """

    def step(self, layers: Iterable) -> None:

        for layer in layers:

            if layer.grad_W is None or layer.grad_b is None:
                continue

            decay_term = self._apply_weight_decay(layer.W)

            layer.W -= self.lr * (layer.grad_W + decay_term)
            layer.b -= self.lr * layer.grad_b


class Momentum(BaseOptimizer):
    """
    Gradient descent with momentum.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
    ) -> None:

        super().__init__(learning_rate, weight_decay)

        self.momentum = float(momentum)
        self.velocities: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def step(self, layers: Iterable) -> None:

        for idx, layer in enumerate(layers):

            if layer.grad_W is None or layer.grad_b is None:
                continue

            decay_term = self._apply_weight_decay(layer.W)

            if idx not in self.velocities:
                v_w = np.zeros_like(layer.W)
                v_b = np.zeros_like(layer.b)
            else:
                v_w, v_b = self.velocities[idx]

            v_w = self.momentum * v_w + layer.grad_W + decay_term
            v_b = self.momentum * v_b + layer.grad_b

            layer.W -= self.lr * v_w
            layer.b -= self.lr * v_b

            self.velocities[idx] = (v_w, v_b)


class NAG(Momentum):
    """
    Nesterov Accelerated Gradient.
    Uses a look-ahead step before applying gradients.
    """

    def step(self, layers: Iterable) -> None:

        for idx, layer in enumerate(layers):

            if layer.grad_W is None or layer.grad_b is None:
                continue

            if idx not in self.velocities:
                v_w = np.zeros_like(layer.W)
                v_b = np.zeros_like(layer.b)
            else:
                v_w, v_b = self.velocities[idx]

            # Look-ahead position
            W_lookahead = layer.W - self.momentum * v_w
            b_lookahead = layer.b - self.momentum * v_b

            decay_term = self._apply_weight_decay(W_lookahead)

            v_w = self.momentum * v_w + layer.grad_W + decay_term
            v_b = self.momentum * v_b + layer.grad_b

            layer.W -= self.lr * v_w
            layer.b -= self.lr * v_b

            self.velocities[idx] = (v_w, v_b)


class RMSProp(BaseOptimizer):
    """
    RMSProp optimizer.

    Maintains a running average of squared gradients to adapt
    the learning rate of each parameter.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        rho: float = 0.9,
        eps: float = 1e-8,
    ) -> None:

        super().__init__(learning_rate, weight_decay)

        self.rho = float(rho)
        self.eps = float(eps)

        self.accumulators: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def step(self, layers: Iterable) -> None:

        for idx, layer in enumerate(layers):

            if layer.grad_W is None or layer.grad_b is None:
                continue

            decay_term = self._apply_weight_decay(layer.W)

            if idx not in self.accumulators:
                s_w = np.zeros_like(layer.W)
                s_b = np.zeros_like(layer.b)
            else:
                s_w, s_b = self.accumulators[idx]

            s_w = self.rho * s_w + (1.0 - self.rho) * (layer.grad_W + decay_term) ** 2
            s_b = self.rho * s_b + (1.0 - self.rho) * (layer.grad_b) ** 2

            layer.W -= self.lr * (layer.grad_W + decay_term) / (np.sqrt(s_w) + self.eps)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(s_b) + self.eps)

            self.accumulators[idx] = (s_w, s_b)


class Adam(BaseOptimizer):
    """
    Adam optimizer.

    Combines momentum and RMSProp ideas using first and second moment estimates.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:

        super().__init__(learning_rate, weight_decay)

        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)

        self.m: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.v: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        self.t: int = 0

    def step(self, layers: Iterable) -> None:

        self.t += 1

        for idx, layer in enumerate(layers):

            if layer.grad_W is None or layer.grad_b is None:
                continue

            decay_term = self._apply_weight_decay(layer.W)

            if idx not in self.m:
                m_w = np.zeros_like(layer.W)
                m_b = np.zeros_like(layer.b)
                v_w = np.zeros_like(layer.W)
                v_b = np.zeros_like(layer.b)
            else:
                m_w, m_b = self.m[idx]
                v_w, v_b = self.v[idx]

            g_w = layer.grad_W + decay_term
            g_b = layer.grad_b

            m_w = self.beta1 * m_w + (1.0 - self.beta1) * g_w
            m_b = self.beta1 * m_b + (1.0 - self.beta1) * g_b

            v_w = self.beta2 * v_w + (1.0 - self.beta2) * (g_w ** 2)
            v_b = self.beta2 * v_b + (1.0 - self.beta2) * (g_b ** 2)

            # Bias correction
            m_w_hat = m_w / (1.0 - self.beta1 ** self.t)
            m_b_hat = m_b / (1.0 - self.beta1 ** self.t)

            v_w_hat = v_w / (1.0 - self.beta2 ** self.t)
            v_b_hat = v_b / (1.0 - self.beta2 ** self.t)

            layer.W -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.eps)
            layer.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)

            self.m[idx] = (m_w, m_b)
            self.v[idx] = (v_w, v_b)


class Nadam(Adam):
    """
    Nadam optimizer (Adam + Nesterov momentum).
    """

    def step(self, layers: Iterable) -> None:

        self.t += 1

        for idx, layer in enumerate(layers):

            if layer.grad_W is None or layer.grad_b is None:
                continue

            decay_term = self._apply_weight_decay(layer.W)

            if idx not in self.m:
                m_w = np.zeros_like(layer.W)
                m_b = np.zeros_like(layer.b)
                v_w = np.zeros_like(layer.W)
                v_b = np.zeros_like(layer.b)
            else:
                m_w, m_b = self.m[idx]
                v_w, v_b = self.v[idx]

            g_w = layer.grad_W + decay_term
            g_b = layer.grad_b

            m_w = self.beta1 * m_w + (1.0 - self.beta1) * g_w
            m_b = self.beta1 * m_b + (1.0 - self.beta1) * g_b

            v_w = self.beta2 * v_w + (1.0 - self.beta2) * (g_w ** 2)
            v_b = self.beta2 * v_b + (1.0 - self.beta2) * (g_b ** 2)

            m_w_hat = m_w / (1.0 - self.beta1 ** self.t)
            m_b_hat = m_b / (1.0 - self.beta1 ** self.t)

            v_w_hat = v_w / (1.0 - self.beta2 ** self.t)
            v_b_hat = v_b / (1.0 - self.beta2 ** self.t)

            # Nesterov-style update
            m_w_nesterov = (
                self.beta1 * m_w_hat
                + (1.0 - self.beta1) * g_w / (1.0 - self.beta1 ** self.t)
            )

            m_b_nesterov = (
                self.beta1 * m_b_hat
                + (1.0 - self.beta1) * g_b / (1.0 - self.beta1 ** self.t)
            )

            layer.W -= self.lr * m_w_nesterov / (np.sqrt(v_w_hat) + self.eps)
            layer.b -= self.lr * m_b_nesterov / (np.sqrt(v_b_hat) + self.eps)

            self.m[idx] = (m_w, m_b)
            self.v[idx] = (v_w, v_b)


def get_optimizer(
    name: str,
    learning_rate: float,
    weight_decay: float = 0.0,
) -> BaseOptimizer:
    """
    Helper function that returns the optimizer object based on the name.
    """

    key = name.lower()

    if key == "sgd":
        return SGD(learning_rate, weight_decay)

    if key == "momentum":
        return Momentum(learning_rate, weight_decay)

    if key == "nag":
        return NAG(learning_rate, weight_decay)

    if key == "rmsprop":
        return RMSProp(learning_rate, weight_decay)

    if key == "adam":
        return Adam(learning_rate, weight_decay)

    if key == "nadam":
        return Nadam(learning_rate, weight_decay)

    raise ValueError(f"Unsupported optimizer: {name}")