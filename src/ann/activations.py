"""
Activation functions used in the neural network along with
their corresponding derivatives for backpropagation.
"""

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    x_clip = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clip))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of the sigmoid function."""
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent activation."""
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of tanh."""
    t = np.tanh(x)
    return 1.0 - t ** 2


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0.0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU."""
    return (x > 0.0).astype(x.dtype)


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax activation applied row-wise.
    A small stabilization trick is used to avoid overflow.
    """
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    sums = np.sum(exp_x, axis=1, keepdims=True)
    return exp_x / sums


# Mapping names to activation functions
ACTIVATIONS = {
    "sigmoid": sigmoid,
    "tanh": tanh,
    "relu": relu,
}


# Mapping names to derivatives
DERIVATIVES = {
    "sigmoid": sigmoid_derivative,
    "tanh": tanh_derivative,
    "relu": relu_derivative,
}