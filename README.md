# DA6401 Assignment 1 - Multi-Layer Perceptron from Scratch

**Roll No:** CE23B005  
**Name:** Chintha Vishnu Vardhan  

---

## Links

- **W&B Report:** https://wandb.ai/chinthavishnuvardhan4-indian-institute-of-technology-madras/DA6401__Intro_to_DL_Assignment1/reports/DA6401-Assignment-1-Multi-Layer-Perceptron--VmlldzoxNjEwNTQ1MA?accessToken=qnw35kh50yseg0ww5gzzfuxqhgmp4zmao8jj5vs8rzq9oqvlgr4tyngf65j1qwg7
- **GitHub:** https://github.com/Chintha-Vishnu-Vardhan/da6401_assignment_1

---

## Overview

This is a fully hand-coded multi-layer perceptron using only NumPy — no PyTorch, no TensorFlow. The assignment asked us to build backprop from scratch, implement multiple optimizers, and run experiments on MNIST and Fashion-MNIST. Training, inference, weight saving and loading all work through command-line scripts.

---

## Folder structure

```
da6401_assignment_1/
│
├── src/
│   ├── ann/
│   │   ├── __init__.py
│   │   ├── activations.py           # sigmoid, tanh, relu, softmax + derivatives
│   │   ├── neural_layer.py          # single dense layer with forward/backward
│   │   ├── neural_network.py        # full MLP: build, train, evaluate, save/load
│   │   ├── objective_functions.py   # cross-entropy and MSE losses + gradients
│   │   └── optimizers.py            # SGD, Momentum, NAG, RMSProp, Adam, Nadam
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── data_loader.py           # loads MNIST / Fashion-MNIST, splits, one-hot
│   │
│   ├── train.py                     # training entry point (full CLI)
│   ├── inference.py                 # load saved weights and evaluate on test set
│   ├── best_model.npy               # saved best weights (dict format)
│   └── best_config.json             # hyperconfig that produced best_model.npy
│
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/Chintha-Vishnu-Vardhan/da6401_assignment_1.git
cd da6401_assignment_1

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Training

Run from the **repo root**:

```bash
python src/train.py
```

This uses the defaults (RMSProp, lr=0.0015, tanh, 3×128, cross-entropy, 30 epochs) and saves weights to `src/best_model.npy` and config to `src/best_config.json`.

You can override anything via CLI:

```bash
python src/train.py \
  --dataset mnist \
  --epochs 30 \
  --batch_size 128 \
  --optimizer rmsprop \
  --learning_rate 0.0015 \
  --weight_decay 0.0001 \
  --num_layers 3 \
  --hidden_size "[128,128,128]" \
  --activation tanh \
  --loss cross_entropy \
  --weight_init xavier
```

### All CLI arguments

| Argument | Short | Default | Options |
|---|---|---|---|
| `--dataset` | `-d` | `mnist` | `mnist`, `fashion_mnist` |
| `--epochs` | `-e` | `30` | any int |
| `--batch_size` | `-b` | `128` | any int |
| `--optimizer` | `-o` | `rmsprop` | `sgd`, `momentum`, `nag`, `rmsprop` |
| `--learning_rate` | `-lr` | `0.0015` | any float |
| `--weight_decay` | `-wd` | `0.0001` | any float |
| `--num_layers` | `-nhl` | `3` | any int |
| `--hidden_size` | `-sz` | `[128,128,128]` | list of ints |
| `--activation` | `-a` | `tanh` | `relu`, `tanh`, `sigmoid` |
| `--loss` | `-l` | `cross_entropy` | `cross_entropy`, `mse` |
| `--weight_init` | `-wi` | `xavier` | `random`, `xavier`, `zeros` |
| `--wandb_project` | `-w_p` | `None` | any string |
| `--model_path` | | `best_model.npy` | path |
| `--config_save_path` | | `best_config.json` | path |

---

## Inference

```bash
python src/inference.py
```

This loads `src/best_model.npy` and `src/best_config.json` (relative to where the script runs from), rebuilds the exact architecture from the saved config, and prints Accuracy, Precision, Recall and F1 on the test set.

You can point it at a different model:

```bash
python src/inference.py \
  --model_path src/best_model.npy \
  --config_save_path src/best_config.json \
  --dataset mnist
```

---

## How the code is structured internally

### `utils/data_loader.py`

`load_dataset(dataset)` downloads MNIST or Fashion-MNIST via `keras.datasets`, normalises pixels to [0, 1], flattens to 784 features, and returns a **7-tuple**:

```python
X_train, y_train, X_val, y_val, X_test, y_test_onehot, y_test_labels = load_dataset("mnist")
```

The train/val split is 90/10, stratified, fixed seed=42. `y_train` and `y_val` are one-hot encoded; `y_test_labels` is raw integer labels (needed for sklearn metrics).

---

### `ann/activations.py`

Contains plain functions plus two dicts:

```python
ACTIVATIONS = {"sigmoid": sigmoid, "tanh": tanh, "relu": relu}
DERIVATIVES = {"sigmoid": sigmoid_derivative, "tanh": tanh_derivative, "relu": relu_derivative}
```

`softmax` is defined separately and used only in loss computation, not inside layers. The output layer has `activation=None`, meaning it outputs raw logits — the autograder expects this.

---

### `ann/neural_layer.py`

A single dense layer. Key attributes:

```python
layer.W        # (input_dim, output_dim)
layer.b        # (1, output_dim)
layer.grad_W   # populated after backward()
layer.grad_b   # populated after backward()
```

`forward(X)` caches `X`, `Z` (pre-activation), and `A` (post-activation). `backward(dA)` computes `dZ = dA * activation_derivative(Z)`, then:

```
grad_W = X.T @ dZ
grad_b = sum(dZ, axis=0)
return dZ @ W.T      # passes gradient to previous layer
```

No division by batch size here — that's handled entirely in the loss gradient.

---

### `ann/objective_functions.py`

Two losses and their gradients:

**Cross-entropy** takes raw logits and applies softmax internally:
```python
loss, probs = cross_entropy_loss(y_true, logits)
```

The gradient is the fused softmax + CE gradient w.r.t. logits:
```python
return (probs - y_true) / batch_size
```
Note the parentheses — this was a bug in an earlier version where `probs - y_true / batch_size` was written, which due to operator precedence only divided `y_true`, making gradients ~128x too large and preventing convergence entirely.

**MSE** is straightforward:
```python
return 2.0 * (y_pred - y_true) / batch_size
```

---

### `ann/neural_network.py`

The main class. Constructor takes `cli_args` (an `argparse.Namespace`) plus `input_dim` and `num_classes`, builds all layers up front, and wires up the optimizer.

**`forward(X)`** runs the input through all layers and returns logits. The output is stored in `self.last_logits`.

**`backward(y_true)`** computes the loss gradient at the output and propagates it backwards through all layers. Returns `(grad_w_arr, grad_b_arr)` as numpy object arrays — each element is the gradient for one layer, ordered from last to first.

**`get_weights()`** returns a plain dict:
```python
{'W0': ..., 'b0': ..., 'W1': ..., 'b1': ..., ...}
```
This is what gets saved to `.npy`. When loaded, `np.load('best_model.npy', allow_pickle=True).item()` gives back the same dict.

**`set_weights(weights)`** accepts the dict format and handles the case where the loaded architecture doesn't match (it rebuilds layers from the weight shapes).

**`train()`** takes `X_train, y_train, X_val, y_val` directly (the val split is already done in `data_loader`). It tracks the best val accuracy across epochs and restores those weights at the end.

**`evaluate(X, y)`** returns `(loss, accuracy)` as a tuple.

---

### `ann/optimizers.py`

All optimizers inherit from `BaseOptimizer` and implement a `step(layers)` method. They check for `layer.grad_W is None` before updating, so calling step before any backward pass is safe. The factory function:

```python
optimizer = get_optimizer("rmsprop", learning_rate=0.001, weight_decay=0.0)
optimizer.step(layers)
```

RMSProp and Momentum both take a `momentum`/`rho` parameter (default 0.9). Adam and Nadam are also included but weren't the focus of the assignment experiments.

---

## Best model results

| Metric | Value |
|---|---|
| Test Accuracy | 97.93% |
| Test F1 | 97.90% |
| Optimizer | RMSProp |
| Architecture | 3 hidden layers × 128 neurons |
| Activation | Tanh |
| Learning rate | 0.0015 |
| Weight decay | 0.0001 |
| Weight init | Xavier |
| Loss | Cross-Entropy |

---

## Hyperparameter sweep

A 170-run Bayesian sweep was done through W&B. The sweep config is in `sweep_config.yaml`. Main takeaways:

- RMSProp consistently outperformed SGD by a large margin; NAG and Momentum were somewhere in between
- Learning rate 0.001–0.003 was the sweet spot; anything above 0.01 tended to diverge with RMSProp
- Tanh vs ReLU was close on MNIST; tanh did slightly better on Fashion-MNIST
- Xavier init was noticeably better than random at deeper configs (4+ layers)

To reproduce a sweep:
```bash
wandb sweep sweep_config.yaml
wandb agent <SWEEP_ID>
```

---

## Notes on the implementation

A few things worth calling out that aren't obvious from just reading the code:

**Why logits and not probabilities?** The autograder calls `forward()` and expects raw logits back. Softmax is only applied inside `cross_entropy_loss` and inside `backward()`. Don't call `softmax` on the output yourself before passing it anywhere in the test pipeline.

**Weight saving format:** `get_weights()` returns a dict, not a list. This was changed from the original version which returned a list of `(W, b)` tuples, because `np.load(...).item()` on a dict-saved array returns the dict directly — exactly what `set_weights` expects.

**The 7-tuple from data_loader:** `train.py` unpacks as `data[0], data[1], data[2], data[3]` for the train/val split. `inference.py` uses `data[4]` and `data[5]` for the test set. The 7th element `data[6]` is integer labels, used only for sklearn metrics.

**Running from repo root:** Both `train.py` and `inference.py` default model paths to `best_model.npy` (relative, no `src/` prefix), because the autograder runs scripts with `src/` as the working directory. If you're running locally from the repo root, just use `--model_path src/best_model.npy`.

---

## Requirements

```
numpy>=1.21.0
keras>=2.7.0
scikit-learn>=0.24.2
wandb>=0.12.0
matplotlib>=3.4.0
```
