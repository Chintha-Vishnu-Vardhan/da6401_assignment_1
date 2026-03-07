"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .neural_layer import NeuralLayer
from .objective_functions import (
    cross_entropy_grad,
    cross_entropy_loss,
    mse_grad,
    mse_loss,
)
from .optimizers import get_optimizer


class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(
        self,
        cli_args: Any,
        input_dim: int = 784,
        num_classes: int = 10,
        rng: Optional[np.random.Generator] = None,
    ):
        self.cli_args = cli_args 
        self.rng = rng or np.random.default_rng()

        activation = getattr(cli_args, "activation", "relu")
        loss_name = getattr(cli_args, "loss", "cross_entropy")
        optimizer_name = getattr(cli_args, "optimizer", "sgd")
        learning_rate = getattr(cli_args, "learning_rate", getattr(cli_args, "lr", 1e-3))
        weight_decay = getattr(cli_args, "weight_decay", getattr(cli_args, "wd", 0.0))
        weight_init = getattr(cli_args, "weight_init", getattr(cli_args, "wi", "xavier"))

        hidden_sizes = None
        if hasattr(cli_args, "hidden_size") and getattr(cli_args, "hidden_size") is not None:
            hidden_sizes = cli_args.hidden_size
        elif hasattr(cli_args, "hidden_layers") and getattr(cli_args, "hidden_layers") is not None:
            hidden_sizes = cli_args.hidden_layers
        elif hasattr(cli_args, "num_neurons") and getattr(cli_args, "num_neurons") is not None:
            hidden_sizes = cli_args.num_neurons

        if isinstance(hidden_sizes, int):
            num_layers = getattr(cli_args, "num_layers", getattr(cli_args, "nhl", 1))
            hidden_sizes = [hidden_sizes] * int(num_layers)
        elif hidden_sizes is None:
            hidden_sizes = [128, 128]

        self.hidden_sizes: List[int] = [int(h) for h in hidden_sizes]
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.activation_name = activation.lower()
        self.loss_name = loss_name.lower()

        self.layers: List[NeuralLayer] = []

        in_dim = self.input_dim
        for h in self.hidden_sizes:
            self.layers.append(
                NeuralLayer(
                    input_dim=in_dim,
                    output_dim=h,
                    activation=self.activation_name,
                    weight_init=weight_init,
                    rng=self.rng,
                )
            )
            in_dim = h

        self.layers.append(
            NeuralLayer(
                input_dim=in_dim,
                output_dim=self.num_classes,
                activation=None,
                weight_init=weight_init,
                rng=self.rng,
            )
        )

        self.optimizer = get_optimizer(
            optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        self.last_logits: Optional[np.ndarray] = None
        self.last_probs: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        self.last_logits = out
        return out

    def compute_loss_and_output(self,y_true: np.ndarray):

        if self.last_logits is None:
            raise RuntimeError("Must call forward() before computing loss.")

        logits = self.last_logits
        m = logits.shape[0]
        n = self.num_classes

        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            y_int = y_true.flatten().astype(int)
            y_onehot = np.zeros((m,n))
            y_onehot[np.arange(m),y_int] = 1.0
        else:
            y_onehot = y_true

        if self.loss_name in ("cross_entropy","crossentropy","ce"):
            loss,probs = cross_entropy_loss(y_onehot,logits)
            self.last_probs = probs
            return loss,probs

        if self.loss_name in ("mse","mean_squared_error"):
            preds = logits
            loss = mse_loss(y_onehot,preds)
            return loss,preds

        raise ValueError(f"Unsupported loss function: {self.loss_name}")

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray = None):
        if self.last_logits is None:
            raise RuntimeError("Forward must be called before backward.")

        m = self.last_logits.shape[0]
        n = self.num_classes

        # Autograder safeguard: convert integer labels to one-hot if necessary
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            y_int = y_true.flatten().astype(int)
            y_onehot = np.zeros((m, n))
            y_onehot[np.arange(m), y_int] = 1.0
        else:
            y_onehot = y_true

        if self.loss_name in ("cross_entropy", "crossentropy", "ce"):
            from .activations import softmax as _softmax
            probs = _softmax(self.last_logits)
            d_out = cross_entropy_grad(y_onehot, probs)
        elif self.loss_name in ("mse", "mean_squared_error"):
            d_out = mse_grad(y_onehot, self.last_logits)
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_name}")

        grad = d_out
        grad_W_list = []
        grad_b_list = []

        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)

        # Gradescope expects numpy object arrays
        grad_w_arr = np.empty(len(grad_W_list), dtype=object)
        grad_b_arr = np.empty(len(grad_b_list), dtype=object)

        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            grad_w_arr[i] = gw
            grad_b_arr[i] = gb

        return grad_w_arr, grad_b_arr

    def update_weights(self) -> None:
        self.optimizer.step(self.layers)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
        batch_size: int,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        wandb_run: Optional[Any] = None,
    ) -> Dict[str, List[float]]:
        num_samples = X_train.shape[0]
        history: Dict[str, List[float]] = {
            "train_loss": [], "train_accuracy": [],
            "val_loss": [], "val_accuracy": [],
        }

        best_val_acc = -1.0
        best_weights = None

        for epoch in range(epochs):
            grad_norms_layer0: List[float] = []
            sparsity_layer0: List[float] = []

            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]
                if X_batch.shape[0] == 0:
                    continue

                logits = self.forward(X_batch)

                first_layer = self.layers[0]
                if first_layer.A is not None:
                    zero_frac = float(np.mean(first_layer.A <= 0.0))
                    sparsity_layer0.append(zero_frac)

                loss, y_out = self.compute_loss_and_output(y_batch)
                self.backward(y_batch, y_out)

                if first_layer.grad_W is not None:
                    grad_norm = float(np.linalg.norm(first_layer.grad_W))
                    grad_norms_layer0.append(grad_norm)

                self.update_weights()

            train_loss, train_acc = self.evaluate(X_train, y_train)
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc)

            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    # Save as dict for compatibility
                    best_weights = self.get_weights()

            if grad_norms_layer0:
                history.setdefault("grad_norm_layer0", []).append(float(np.mean(grad_norms_layer0)))
            if sparsity_layer0:
                history.setdefault("activation_sparsity_layer0", []).append(float(np.mean(sparsity_layer0)))

            if wandb_run is not None:
                log_data = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                }
                if X_val is not None and y_val is not None:
                    log_data["val_loss"] = history["val_loss"][-1]
                    log_data["val_accuracy"] = history["val_accuracy"][-1]
                if "grad_norm_layer0" in history:
                    log_data["grad_norm_layer0"] = history["grad_norm_layer0"][-1]
                if "activation_sparsity_layer0" in history:
                    log_data["activation_sparsity_layer0"] = history["activation_sparsity_layer0"][-1]
                wandb_run.log(log_data)

            print(f"Epoch [{epoch+1}/{epochs}]  "
                  f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
                  + (f"  val_loss={history['val_loss'][-1]:.4f}"
                     f"  val_acc={history['val_accuracy'][-1]:.4f}"
                     if X_val is not None else ""))

        if best_weights is not None:
            self.set_weights(best_weights)
            print(f"Restored best weights with Val Accuracy: {best_val_acc:.4f}")

        return history

    def evaluate(self,X: np.ndarray,y: np.ndarray):

        logits = self.forward(X)
        loss,y_pred_for_loss = self.compute_loss_and_output(y)

        if y.ndim == 2:
            y_true_labels = np.argmax(y,axis=1)
        else:
            y_true_labels = y

        if self.loss_name in ("cross_entropy","crossentropy","ce"):
            probs = self.last_probs if self.last_probs is not None else y_pred_for_loss
            y_pred_labels = np.argmax(probs,axis=1)
        else:
            y_pred_labels = np.argmax(y_pred_for_loss,axis=1)

        accuracy = float(np.mean(y_pred_labels == y_true_labels))

        return float(loss),accuracy

    # ── FIX: get_weights returns dict so np.load(...).item() gives a dict ──
    def get_weights(self) -> Dict[str, np.ndarray]:
        """
        Return weights as a dict: {'W0': W, 'b0': b, 'W1': W, 'b1': b, ...}

        This format is required by the autograder:
            data = np.load('best_model.npy', allow_pickle=True).item()
            model.set_weights(data)  # expects dict
        """
        d = {}
        for i, layer in enumerate(self.layers):
            d[f'W{i}'] = layer.W.copy()
            d[f'b{i}'] = layer.b.copy()
        return d

    def set_weights(self, weights) -> None:
        """
        Set weights for all layers. Dynamically rebuilds layers if the 
        injected weights don't match the current initialized architecture.
        """
        if isinstance(weights, dict):
            # Count how many layers are in the weight dict
            n_layers = sum(1 for k in weights if k.startswith("W"))
            if n_layers == 0:
                return
            
            # Check if architecture needs to be rebuilt
            needs_rebuild = (n_layers != len(self.layers))
            if not needs_rebuild:
                for i, layer in enumerate(self.layers):
                    if weights[f"W{i}"].shape != layer.W.shape:
                        needs_rebuild = True
                        break
            
            if needs_rebuild:
                # Rebuild layers to exactly match the weight dict shapes
                activation = getattr(self.cli_args, "activation", self.activation_name)
                weight_init = getattr(self.cli_args, "weight_init", "xavier")
                self.layers = []
                for i in range(n_layers):
                    W = weights[f"W{i}"]
                    in_size, out_size = W.shape
                    is_last = (i == n_layers - 1)
                    act = None if is_last else activation
                    self.layers.append(
                        NeuralLayer(
                            input_dim=in_size,
                            output_dim=out_size,
                            activation=act,
                            weight_init=weight_init,
                            rng=self.rng
                        )
                    )

            # Apply the weights
            for i, layer in enumerate(self.layers):
                layer.W = np.array(weights[f"W{i}"], dtype=np.float64)
                layer.b = np.array(weights[f"b{i}"], dtype=np.float64)
            return

