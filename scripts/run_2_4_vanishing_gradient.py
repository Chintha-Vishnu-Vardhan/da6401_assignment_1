"""Section 2.4: Vanishing Gradient Analysis
Fixes optimizer to RMSProp. Compares Sigmoid vs ReLU across different 
network depths. Logs gradient norms for the first hidden layer.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import wandb
from src.utils.data_loader import load_dataset
from src.ann.neural_network import NeuralNetwork

data = load_dataset("mnist")
X_train, y_train, X_val, y_val = data[0], data[1], data[2], data[3]

configs = [
    # (activation, num_layers, hidden_size, run_name)
    ("sigmoid", 3, [128, 128, 128],       "2.4_sigmoid_3layers"),
    ("relu",    3, [128, 128, 128],       "2.4_relu_3layers"),
    ("sigmoid", 5, [128, 128, 128, 128, 128], "2.4_sigmoid_5layers"),
    ("relu",    5, [128, 128, 128, 128, 128], "2.4_relu_5layers"),
]

for activation, num_layers, hidden_size, run_name in configs:

    class Args:
        dataset = "mnist"
        epochs = 10
        batch_size = 128
        optimizer = "rmsprop"
        learning_rate = 0.001
        weight_decay = 0.0
        loss = "cross_entropy"
        weight_init = "xavier"
        wandb_project = "DA6401__Intro_to_DL_Assignment1"
        model_path = "src/temp_model.npy"
        config_save_path = "src/temp_config.json"

    args = Args()
    args.activation = activation
    args.num_layers = num_layers
    args.hidden_size = hidden_size

    run = wandb.init(
        project="DA6401__Intro_to_DL_Assignment1",
        name=run_name,
        group="2.4_Vanishing_Gradient",
        config={
            "activation": activation,
            "num_layers": num_layers,
            "optimizer": "rmsprop",
            "learning_rate": 0.001,
        },
        reinit=True
    )

    model = NeuralNetwork(args, input_dim=784, num_classes=10)

    num_samples = X_train.shape[0]

    for epoch in range(args.epochs):
        # Shuffle
        idx = np.random.permutation(num_samples)
        X_shuf, y_shuf = X_train[idx], y_train[idx]

        epoch_grad_norms = []

        for start in range(0, num_samples, args.batch_size):
            X_batch = X_shuf[start:start+args.batch_size]
            y_batch = y_shuf[start:start+args.batch_size]
            if len(X_batch) == 0:
                continue

            model.forward(X_batch)
            loss, y_out = model.compute_loss_and_output(y_batch)
            model.backward(y_batch, y_out)
            model.update_weights()

            # Gradient norm of FIRST hidden layer
            if model.layers[0].grad_W is not None:
                norm = float(np.linalg.norm(model.layers[0].grad_W))
                epoch_grad_norms.append(norm)

        train_loss, train_acc = model.evaluate(X_train, y_train)
        val_loss, val_acc     = model.evaluate(X_val,   y_val)
        avg_grad_norm = float(np.mean(epoch_grad_norms)) if epoch_grad_norms else 0.0

        print(f"[{run_name}] Epoch {epoch+1} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_acc={val_acc:.4f} | grad_norm_layer0={avg_grad_norm:.6f}")

        run.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "grad_norm_layer0": avg_grad_norm,
        })

    run.finish()
    print(f"✅ Done: {run_name}\n")