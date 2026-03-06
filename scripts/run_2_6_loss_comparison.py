"""Section 2.6: Loss Function Comparison — MSE vs Cross-Entropy"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import wandb
from src.utils.data_loader import load_dataset
from src.ann.neural_network import NeuralNetwork

data = load_dataset("mnist")
X_train, y_train, X_val, y_val = data[0], data[1], data[2], data[3]

configs = [
    ("cross_entropy", "2.6_cross_entropy"),
    ("mse",           "2.6_mse"),
]

for loss_fn, run_name in configs:

    class Args:
        dataset      = "mnist"
        epochs       = 15
        batch_size   = 128
        optimizer    = "rmsprop"
        learning_rate = 0.001
        weight_decay = 0.0
        num_layers   = 3
        hidden_size  = [128, 128, 128]
        activation   = "relu"
        weight_init  = "xavier"
        wandb_project = "DA6401__Intro_to_DL_Assignment1"
        model_path   = "src/temp_model.npy"
        config_save_path = "src/temp_config.json"

    args = Args()
    args.loss = loss_fn

    run = wandb.init(
        project="DA6401__Intro_to_DL_Assignment1",
        name=run_name,
        group="2.6_Loss_Comparison",
        config={
            "loss": loss_fn,
            "optimizer": "rmsprop",
            "learning_rate": 0.001,
            "architecture": "3x128_ReLU",
        },
        reinit=True
    )

    model = NeuralNetwork(args, input_dim=784, num_classes=10)
    num_samples = X_train.shape[0]

    for epoch in range(args.epochs):
        idx = np.random.permutation(num_samples)
        X_shuf, y_shuf = X_train[idx], y_train[idx]

        for start in range(0, num_samples, args.batch_size):
            X_batch = X_shuf[start:start+args.batch_size]
            y_batch = y_shuf[start:start+args.batch_size]
            if len(X_batch) == 0:
                continue
            model.forward(X_batch)
            loss, y_out = model.compute_loss_and_output(y_batch)
            model.backward(y_batch, y_out)
            model.update_weights()

        train_loss, train_acc = model.evaluate(X_train, y_train)
        val_loss,   val_acc   = model.evaluate(X_val,   y_val)

        print(f"[{run_name}] Epoch {epoch+1} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        run.log({
            "epoch":          epoch + 1,
            "train_loss":     train_loss,
            "train_accuracy": train_acc,
            "val_loss":       val_loss,
            "val_accuracy":   val_acc,
        })

    run.finish()
    print(f"✅ Done: {run_name}\n")