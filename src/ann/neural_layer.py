"""
Implementation of a single fully-connected (dense) neural network layer.
Each layer stores its parameters, performs forward propagation,
and computes gradients during the backward pass.
"""

from typing import Optional
import numpy as np

from .activations import ACTIVATIONS, DERIVATIVES


class NeuralLayer:
    """
    Dense layer used inside the neural network.

    Attributes expected by the autograder:
        W       : weight matrix of shape (input_dim, output_dim)
        b       : bias vector of shape (1, output_dim)
        grad_W  : gradient of loss with respect to W
        grad_b  : gradient of loss with respect to b
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: Optional[str] = None,
        weight_init: str = "xavier",
        rng: Optional[np.random.Generator] = None,
    ) -> None:

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.activation_name = activation
        self.weight_init = weight_init.lower() if weight_init is not None else "xavier"
        self.rng = rng or np.random.default_rng()

        self.W, self.b = self._init_parameters()

        # Cached values used during backpropagation
        self.X: Optional[np.ndarray] = None
        self.Z: Optional[np.ndarray] = None
        self.A: Optional[np.ndarray] = None

        # Gradients computed in backward pass
        self.grad_W: Optional[np.ndarray] = None
        self.grad_b: Optional[np.ndarray] = None

    def _init_parameters(self):
        """Initialize weights and biases according to the chosen scheme."""

        if self.weight_init == "zeros":
            W = np.zeros((self.input_dim, self.output_dim), dtype=np.float64)

        elif self.weight_init == "random":
            # Small random initialization
            W = self.rng.standard_normal((self.input_dim, self.output_dim)) * 0.01

        else:  # Xavier initialization
            std = np.sqrt(2.0 / (self.input_dim + self.output_dim))
            W = self.rng.standard_normal((self.input_dim, self.output_dim)) * std

        # Bias stored as a row vector for broadcasting
        b = np.zeros((1, self.output_dim), dtype=np.float64)

        return W, b

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation through the layer.

        Parameters
        ----------
        X : np.ndarray
            Input of shape (batch_size, input_dim)

        Returns
        -------
        np.ndarray
            Output of the layer after applying activation.
        """

        self.X = X
        self.Z = X @ self.W + self.b

        if self.activation_name is None:
            self.A = self.Z
        else:
            activation_fn = ACTIVATIONS[self.activation_name]
            self.A = activation_fn(self.Z)

        return self.A

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        Backward propagation step for the layer.

        Parameters
        ----------
        dA : np.ndarray
            Gradient of loss with respect to the layer output.

        Returns
        -------
        np.ndarray
            Gradient with respect to the layer input.
        """

        if self.X is None or self.Z is None:
            raise RuntimeError("Forward pass must be executed before backward.")

        # Apply activation derivative if the layer has an activation
        if self.activation_name is None:
            dZ = dA
        else:
            derivative_fn = DERIVATIVES[self.activation_name]
            dZ = dA * derivative_fn(self.Z)

        # Gradients for weights and bias
        self.grad_W = self.X.T @ dZ
        self.grad_b = np.sum(dZ, axis=0, keepdims=True)

        # Gradient propagated to previous layer
        dX = dZ @ self.W.T
        return dX