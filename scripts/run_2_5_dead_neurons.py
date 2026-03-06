"""Section 2.5: The Dead Neuron Investigation
ReLU + high LR (0.1) vs Tanh + high LR (0.1).
Monitors activation sparsity (fraction of zero activations = dead neurons).
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
    ("relu",  0.1,   "2.5_relu_high_lr"),
    ("tanh",  0.1,   "2.5_tanh_high_lr"),
    ("relu",  0.001, "2.5_relu_normal_lr"),  # control run
]

for activation, lr, run_name in configs:

    class Args:
        dataset = "mnist"
        epochs = 15
        batch_size = 128
        optimizer = "sgd"
        weight_decay = 0.0
        num_layers = 3
        hidden_size = [128, 128, 128]
        loss = "cross_entropy"
        weight_init = "xavier"
        wandb_project = "DA6401__Intro_to_DL_Assignment1"
        model_path = "src/temp_model.npy"
        config_save_path = "src/temp_config.json"

    args = Args()
    args.activation = activation
    args.learning_rate = lr

    run = wandb.init(
        project="DA6401__Intro_to_DL_Assignment1",
        name=run_name,
        group="2.5_Dead_Neurons",
        config={
            "activation": activation,
            "learning_rate": lr,
            "optimizer": "sgd",
            "num_layers": 3,
        },
        reinit=True
    )

    model = NeuralNetwork(args, input_dim=784, num_classes=10)
    num_samples = X_train.shape[0]

    for epoch in range(args.epochs):
        idx = np.random.permutation(num_samples)
        X_shuf, y_shuf = X_train[idx], y_train[idx]

        batch_sparsity_l1, batch_sparsity_l2, batch_sparsity_l3 = [], [], []
        batch_grad_norms = []

        for start in range(0, num_samples, args.batch_size):
            X_batch = X_shuf[start:start+args.batch_size]
            y_batch = y_shuf[start:start+args.batch_size]
            if len(X_batch) == 0:
                continue

            model.forward(X_batch)

            # Measure dead neurons: fraction of activations <= 0
            if model.layers[0].A is not None:
                batch_sparsity_l1.append(float(np.mean(model.layers[0].A <= 0)))
            if model.layers[1].A is not None:
                batch_sparsity_l2.append(float(np.mean(model.layers[1].A <= 0)))
            if model.layers[2].A is not None:
                batch_sparsity_l3.append(float(np.mean(model.layers[2].A <= 0)))

            loss, y_out = model.compute_loss_and_output(y_batch)
            model.backward(y_batch, y_out)

            if model.layers[0].grad_W is not None:
                batch_grad_norms.append(float(np.linalg.norm(model.layers[0].grad_W)))

            model.update_weights()

        train_loss, train_acc = model.evaluate(X_train, y_train)
        val_loss, val_acc     = model.evaluate(X_val,   y_val)

        log_data = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "dead_neuron_frac_layer1": np.mean(batch_sparsity_l1) if batch_sparsity_l1 else 0,
            "dead_neuron_frac_layer2": np.mean(batch_sparsity_l2) if batch_sparsity_l2 else 0,
            "dead_neuron_frac_layer3": np.mean(batch_sparsity_l3) if batch_sparsity_l3 else 0,
            "grad_norm_layer0": np.mean(batch_grad_norms) if batch_grad_norms else 0,
        }

        print(f"[{run_name}] Epoch {epoch+1} | "
              f"val_acc={val_acc:.4f} | "
              f"dead_l1={log_data['dead_neuron_frac_layer1']:.3f} | "
              f"dead_l2={log_data['dead_neuron_frac_layer2']:.3f} | "
              f"dead_l3={log_data['dead_neuron_frac_layer3']:.3f} | "
              f"grad_norm={log_data['grad_norm_layer0']:.4f}")

        run.log(log_data)

    run.finish()
    print(f"✅ Done: {run_name}\n")