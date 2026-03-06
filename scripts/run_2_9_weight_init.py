"""Section 2.9: Weight Initialization & Symmetry Breaking
Compares Zeros vs Xavier initialization.
Logs gradients of 5 specific neurons in layer 0 for first 50 iterations.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import wandb
from src.utils.data_loader import load_dataset
from src.ann.neural_network import NeuralNetwork

data = load_dataset("mnist")
X_train, y_train = data[0], data[1]
X_val,   y_val   = data[2], data[3]

# We track exactly 5 neurons (columns) in layer 0's weight matrix
TRACKED_NEURONS = [0, 1, 2, 3, 4]
MAX_ITERATIONS  = 50   # first 50 gradient steps

for init_name in ["zeros", "xavier"]:

    class Args:
        dataset       = "mnist"
        epochs        = 10
        batch_size    = 128
        optimizer     = "sgd"
        learning_rate = 0.01
        weight_decay  = 0.0
        num_layers    = 3
        hidden_size   = [128, 128, 128]
        activation    = "relu"
        loss          = "cross_entropy"
        wandb_project = "DA6401__Intro_to_DL_Assignment1"
        model_path    = "src/temp_model.npy"
        config_save_path = "src/temp_config.json"

    args = Args()
    args.weight_init = init_name

    run = wandb.init(
        project="DA6401__Intro_to_DL_Assignment1",
        name=f"2.9_{init_name}_init",
        group="2.9_Weight_Init_Symmetry",
        config={
            "weight_init": init_name,
            "optimizer": "sgd",
            "learning_rate": 0.01,
            "architecture": "3x128_ReLU",
        },
        reinit=True
    )

    model = NeuralNetwork(args, input_dim=784, num_classes=10)
    num_samples = X_train.shape[0]
    iteration   = 0
    stop        = False

    for epoch in range(args.epochs):
        if stop:
            break
        idx    = np.random.permutation(num_samples)
        X_shuf = X_train[idx]
        y_shuf = y_train[idx]

        for start in range(0, num_samples, args.batch_size):
            if iteration >= MAX_ITERATIONS:
                stop = True
                break

            X_batch = X_shuf[start:start + args.batch_size]
            y_batch = y_shuf[start:start + args.batch_size]
            if len(X_batch) == 0:
                continue

            model.forward(X_batch)
            loss, y_out = model.compute_loss_and_output(y_batch)
            model.backward(y_batch, y_out)

            # grad_W shape: (784, 128) — track gradient norm for 5 output neurons
            grad_W = model.layers[0].grad_W  # (784, 128)

            log_data = {"iteration": iteration + 1, "train_loss": float(loss)}

            for n in TRACKED_NEURONS:
                # L2 norm of the gradient column for neuron n
                neuron_grad_norm = float(np.linalg.norm(grad_W[:, n]))
                log_data[f"neuron_{n}_grad_norm"] = neuron_grad_norm

            # Also log overall grad norm
            log_data["grad_norm_layer0"] = float(np.linalg.norm(grad_W))

            print(f"[{init_name}] iter={iteration+1:3d} | loss={loss:.4f} | "
                  + " | ".join([f"n{n}={log_data[f'neuron_{n}_grad_norm']:.6f}"
                                 for n in TRACKED_NEURONS]))

            run.log(log_data)
            model.update_weights()
            iteration += 1

    run.finish()
    print(f"✅ Done: {init_name}\n")