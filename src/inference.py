"""
Inference Script

This script loads a trained neural network model and evaluates it on the
test portion of the dataset. The stored model weights and configuration
are used to reconstruct the network before performing inference.
"""

import argparse
import json
import os
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


def parse_arguments():
    """
    Parse command line arguments required for running inference.
    The arguments mirror the training configuration so the model
    architecture can be reconstructed properly.
    """
    parser = argparse.ArgumentParser(description="Inference for Neural Network")

    parser.add_argument("-d", "--dataset", type=str, default="mnist")

    parser.add_argument("-e", "--epochs", type=int, default=30)
    parser.add_argument("-b", "--batch_size", type=int, default=128)

    parser.add_argument("-o", "--optimizer", type=str, default="rmsprop")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0015)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0001)

    parser.add_argument("-nhl", "--num_layers", type=int, default=3)
    parser.add_argument("-sz", "--hidden_size", type=str, nargs="+",
                        default=["128", "128", "128"])

    parser.add_argument("-a", "--activation", type=str, default="tanh")
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy")
    parser.add_argument("-wi", "--weight_init", type=str, default="xavier")

    parser.add_argument("-w_p", "--wandb_project", type=str, default=None)

    # Locations of the saved model and configuration file
    parser.add_argument("--model_path", type=str, default="best_model.npy")
    parser.add_argument("--config_save_path", type=str, default="best_config.json")

    parser.add_argument("--config_path", type=str, default=None)

    args = parser.parse_args()

    # Allow overriding the config file path if provided explicitly
    if args.config_path is not None:
        args.config_save_path = args.config_path

    return args


def normalize_hidden_sizes(args):
    """
    Ensure hidden layer sizes are stored as a list of integers.
    This helps avoid issues when values are read from JSON or CLI.
    """
    hs = getattr(args, "hidden_size", [128, 128, 128])

    if isinstance(hs, str):
        hs = [int(x) for x in hs.replace("[", "").replace("]", "").split(",")]

    elif isinstance(hs, list):
        hs = [int(x) for x in hs]

    args.hidden_size = hs
    return args


def load_model_from_disk(model_path: str, config_path: str, args: Any) -> NeuralNetwork:
    """
    Load saved model weights and configuration from disk, then
    rebuild the neural network using the stored parameters.
    """

    # Load saved configuration if available
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            saved_config = json.load(f)

        for key, value in saved_config.items():
            setattr(args, key, value)

    args = normalize_hidden_sizes(args)

    weights = np.load(model_path, allow_pickle=True)

    # np.save sometimes stores dicts inside a 0-dim ndarray
    if isinstance(weights, np.ndarray) and weights.ndim == 0:
        weights = weights.item()

    model = NeuralNetwork(args, input_dim=784, num_classes=10)
    model.set_weights(weights)

    return model


def evaluate_model(model: NeuralNetwork,
                   X_test: np.ndarray,
                   y_test,
                   batch_size: int = 512):
    """
    Run forward passes on the test data and compute evaluation metrics.
    """

    n = X_test.shape[0]
    logits_list = []

    # Process data in batches to avoid large memory usage
    for start in range(0, n, batch_size):
        end = start + batch_size
        logits_list.append(model.forward(X_test[start:end]))

    logits = np.vstack(logits_list)

    y_pred = np.argmax(logits, axis=1)

    # Handle both one-hot labels and integer labels
    if y_test.ndim == 2:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    # Attempt to compute loss using the model’s internal loss function
    try:
        model.last_logits = logits
        loss, _ = model.compute_loss_and_output(y_test)
    except Exception:
        loss = 0.0

    return {
        "logits": logits,
        "loss": float(loss),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


def main():
    """
    Main inference pipeline:
    1. Load saved model
    2. Load test dataset
    3. Run evaluation
    4. Print metrics
    """

    args = parse_arguments()

    model = load_model_from_disk(
        args.model_path,
        args.config_save_path,
        args
    )

    data = load_dataset(args.dataset)

    X_test = data[4]
    y_test = data[5]

    results = evaluate_model(
        model,
        X_test,
        y_test,
        batch_size=args.batch_size
    )

    print(
        f"Loss: {results['loss']:.4f}\n"
        f"Accuracy: {results['accuracy']:.4f}\n"
        f"Precision: {results['precision']:.4f}\n"
        f"Recall: {results['recall']:.4f}\n"
        f"F1-score: {results['f1']:.4f}"
    )

    return results


if __name__ == "__main__":
    main()