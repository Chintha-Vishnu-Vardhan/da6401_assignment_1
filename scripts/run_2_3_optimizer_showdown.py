"""Section 2.3: The Optimizer Showdown"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import load_dataset
from src.ann.neural_network import NeuralNetwork
import wandb

data = load_dataset("mnist")
X_train, y_train, X_val, y_val = data[0], data[1], data[2], data[3]

for opt in ["sgd", "momentum", "nag", "rmsprop"]:
    class Args:
        dataset = "mnist"
        epochs = 10
        batch_size = 128
        learning_rate = 0.001
        weight_decay = 0.0
        num_layers = 3
        hidden_size = [128, 128, 128]
        activation = "relu"
        loss = "cross_entropy"
        weight_init = "xavier"
        wandb_project = "DA6401__Intro_to_DL_Assignment1"
        model_path = "src/temp_model.npy"
        config_save_path = "src/temp_config.json"
    
    args = Args()
    args.optimizer = opt

    run = wandb.init(
        project="DA6401__Intro_to_DL_Assignment1",
        name=f"2.3_{opt}",
        group="2.3_Optimizer_Showdown",
        config={
            "optimizer": opt,
            "learning_rate": 0.001,
            "architecture": "3x128_ReLU",
            "loss": "cross_entropy"
        },
        reinit=True
    )

    model = NeuralNetwork(args, input_dim=784, num_classes=10)
    model.train(X_train, y_train, epochs=10, batch_size=128,
                X_val=X_val, y_val=y_val, wandb_run=run)
    run.finish()
    print(f"✅ Done: {opt}\n")