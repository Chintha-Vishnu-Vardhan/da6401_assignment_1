"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import json
from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train/Inference for Neural Network")

    # Required by Guidelines: Best configuration as defaults
    parser.add_argument("-d", "--dataset", type=str, default="mnist", choices=["mnist", "fashion", "fashion_mnist"])
    parser.add_argument("-e", "--epochs", type=int, default=12)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-o", "--optimizer", type=str, default="nadam", 
                        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.003)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-nhl", "--num_layers", type=int, default=3)
    parser.add_argument("-sz", "--hidden_size", type=str, nargs="+", default=["128", "128", "128"])
    parser.add_argument("-a", "--activation", type=str, default="sigmoid", choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("-l", "--loss", type=str, default="mse", choices=["cross_entropy", "mse"])
    parser.add_argument("-wi", "--weight_init", type=str, default="xavier", choices=["random", "xavier", "zeros"])
    
    # Revised Guideline additions
    parser.add_argument("-w_p", "--wandb_project", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="best_model.npy")
    parser.add_argument("--config_save_path", type=str, default="config.json")

    return parser.parse_args()


def load_model_from_disk(model_path: str, args: Any) -> NeuralNetwork:
    """
    Load trained model from disk as per revised guidelines.
    """
    data = np.load(model_path, allow_pickle=True)
    # Depending on numpy version, it might be an object array or list
    if isinstance(data, np.ndarray) and data.shape == ():
        weights = data.item()
    else:
        weights = list(data)
        
    model = NeuralNetwork(args, input_dim=784, num_classes=10)
    model.set_weights(weights)
    return model


def evaluate_model(model: NeuralNetwork, X_test: np.ndarray, y_test_onehot: np.ndarray, y_test_labels: np.ndarray):
    """
    Evaluate model on test data.

    Returns:
        Dictionary with keys: logits, loss, accuracy, f1, precision, recall
    """
    logits = model.forward(X_test)
    loss, _ = model.compute_loss_and_output(y_test_onehot)

    y_pred_labels = np.argmax(model.last_probs if model.last_probs is not None else logits, axis=1)

    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_labels,
        y_pred_labels,
        average="macro",
        zero_division=0,
    )

    return {
        "logits": logits,
        "loss": float(loss),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def main():
    """
    Main inference function.

    Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()

    # Load config if available
    try:
        with open(args.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        # Minimal fallback config; assumes default training args were used
        config = {
            "dataset": args.dataset,
            "epochs": 0,
            "batch_size": args.batch_size,
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "num_layers": 2,
            "hidden_size": [128, 128],
            "activation": "relu",
            "loss": "cross_entropy",
            "weight_init": "xavier",
        }

    # Ensure dataset consistency
    config["dataset"] = args.dataset

    # Load data (test split only)
    (
        _X_train,
        _y_train,
        _X_val,
        _y_val,
        X_test,
        y_test_onehot,
        y_test_labels,
    ) = load_dataset(args.dataset)

    # Build and load model
    model = load_model(args.model_path, config)

    results = evaluate_model(model, X_test, y_test_onehot, y_test_labels)

    print(
        f"Test Loss: {results['loss']:.4f}, "
        f"Accuracy: {results['accuracy']:.4f}, "
        f"F1: {results['f1']:.4f}, "
        f"Precision: {results['precision']:.4f}, "
        f"Recall: {results['recall']:.4f}"
    )

    return results


if __name__ == "__main__":
    main()

