"""Section 2.10: Fashion-MNIST Transfer Challenge
Based on MNIST learnings, pick 3 best configurations and test on Fashion-MNIST.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
from src.utils.data_loader import load_dataset
from src.ann.neural_network import NeuralNetwork

data = load_dataset("fashion_mnist")
X_train, y_train, X_val, y_val = data[0], data[1], data[2], data[3]

# 3 configs chosen strictly from MNIST learnings:
# Config 1: Best MNIST config (RMSProp, 3x128, ReLU) — direct transfer
# Config 2: Deeper network (RMSProp, 4x128, ReLU) — more capacity for complex patterns
# Config 3: RMSProp, 3x128, Tanh — test if Tanh's smoothness helps on harder dataset
configs = [
    {
        "name": "2.10_config1_rmsprop_3x128_relu",
        "optimizer": "rmsprop", "learning_rate": 0.001,
        "num_layers": 3, "hidden_size": [128, 128, 128],
        "activation": "relu", "epochs": 20,
        "rationale": "Best MNIST config — direct transfer baseline"
    },
    {
        "name": "2.10_config2_rmsprop_4x128_relu",
        "optimizer": "rmsprop", "learning_rate": 0.001,
        "num_layers": 4, "hidden_size": [128, 128, 128, 128],
        "activation": "relu", "epochs": 20,
        "rationale": "Deeper network for higher complexity of fashion items"
    },
    {
        "name": "2.10_config3_rmsprop_3x128_tanh",
        "optimizer": "rmsprop", "learning_rate": 0.001,
        "num_layers": 3, "hidden_size": [128, 128, 128],
        "activation": "tanh", "epochs": 20,
        "rationale": "Tanh avoids dead neurons, smoother gradients on harder task"
    },
]

for cfg in configs:

    class Args:
        dataset = "fashion_mnist"
        batch_size = 128
        weight_decay = 0.0
        loss = "cross_entropy"
        weight_init = "xavier"
        wandb_project = "DA6401__Intro_to_DL_Assignment1"
        model_path = "src/temp_model.npy"
        config_save_path = "src/temp_config.json"

    args = Args()
    args.optimizer     = cfg["optimizer"]
    args.learning_rate = cfg["learning_rate"]
    args.num_layers    = cfg["num_layers"]
    args.hidden_size   = cfg["hidden_size"]
    args.activation    = cfg["activation"]
    args.epochs        = cfg["epochs"]

    run = wandb.init(
        project="DA6401__Intro_to_DL_Assignment1",
        name=cfg["name"],
        group="2.10_Fashion_MNIST_Transfer",
        config={
            "optimizer":      cfg["optimizer"],
            "learning_rate":  cfg["learning_rate"],
            "num_layers":     cfg["num_layers"],
            "hidden_size":    str(cfg["hidden_size"]),
            "activation":     cfg["activation"],
            "dataset":        "fashion_mnist",
            "rationale":      cfg["rationale"],
        },
        reinit=True
    )

    model = NeuralNetwork(args, input_dim=784, num_classes=10)
    history = model.train(
        X_train, y_train,
        epochs=cfg["epochs"],
        batch_size=128,
        X_val=X_val, y_val=y_val,
        wandb_run=run
    )

    best_val = max(history["val_accuracy"])
    print(f"\n✅ {cfg['name']}: Best Val Accuracy = {best_val:.4f}\n")
    run.finish()