# DA6401 Assignment 1 - Multi-Layer Perceptron for Image Classification

**Course:** DA6401 - Introduction to Deep Learning  
Roll No. CE23B005
Chintha Vishnu Vardhan 
Institute: Indian Institute of Technology Madras  
---

## Important Links

| Resource | Link |
|---|---|
| **W&B Report** | https://wandb.ai/chinthavishnuvardhan4-indian-institute-of-technology-madras/DA6401__Intro_to_DL_Assignment1/reports/DA6401-Assignment-1-Multi-Layer-Perceptron--VmlldzoxNjEwNTQ1MA?accessToken=qnw35kh50yseg0ww5gzzfuxqhgmp4zmao8jj5vs8rzq9oqvlgr4tyngf65j1qwg7 |
| **GitHub Repository** | https://github.com/Chintha-Vishnu-Vardhan/da6401_assignment_1 |


---

## Project Structure

```
da6401_assignment_1/
│
├── src/
│   ├── ann/
│   │   ├── neural_layer.py          # Single layer: forward, backward, grad_W, grad_b
│   │   ├── neural_network.py        # Full MLP: forward, backward, get/set weights
│   │   ├── activations.py           # sigmoid, tanh, relu, softmax + derivatives
│   │   └── objective_functions.py   # cross_entropy, mse + gradients
│   │
│   ├── utils/
│   │   └── data_loader.py           # MNIST & Fashion-MNIST loading + preprocessing
│   │
│   ├── train.py                     # Training script with full CLI (argparse)
│   ├── inference.py                 # Inference script: loads .npy weights, reports metrics
│   ├── best_model.npy               # Best model weights (saved by test F1-score)
│   └── best_config.json             # Best hyperparameter configuration
│
├── scripts/
│   ├── run_2_3_optimizer_showdown.py
│   ├── run_2_4_vanishing_gradient.py
│   ├── run_2_5_dead_neurons.py
│   ├── run_2_6_loss_comparison.py
│   ├── run_2_7_global_analysis.py
│   ├── run_2_8_confusion_matrix.py
│   ├── run_2_9_weight_init.py
│   └── run_2_10_fashion_mnist.py
│
├── sweep_config.yaml                # W&B Bayesian sweep configuration
├── requirements.txt
└── README.md
```

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/Chintha-Vishnu-Vardhan/da6401_assignment_1.git
cd da6401_assignment_1

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Training

```bash
python src/train.py \
  --dataset mnist \
  --epochs 50 \
  --batch_size 128 \
  --optimizer rmsprop \
  --learning_rate 0.001 \
  --weight_decay 0.0 \
  --num_layers 3 \
  --hidden_size "[128,128,128]" \
  --activation relu \
  --loss cross_entropy \
  --weight_init xavier \
  --wandb_project DA6401__Intro_to_DL_Assignment1
```

### CLI Arguments

| Argument | Short | Description | Default |
|---|---|---|---|
| `--dataset` | `-d` | `mnist` or `fashion_mnist` | `mnist` |
| `--epochs` | `-e` | Number of training epochs | `50` |
| `--batch_size` | `-b` | Mini-batch size | `128` |
| `--loss` | `-l` | `cross_entropy` or `mse` | `cross_entropy` |
| `--optimizer` | `-o` | `sgd`, `momentum`, `nag`, `rmsprop` | `rmsprop` |
| `--learning_rate` | `-lr` | Initial learning rate | `0.001` |
| `--weight_decay` | `-wd` | L2 regularization coefficient | `0.0` |
| `--num_layers` | `-nhl` | Number of hidden layers | `3` |
| `--hidden_size` | `-sz` | Neurons per hidden layer (list) | `[128,128,128]` |
| `--activation` | `-a` | `sigmoid`, `tanh`, `relu` | `relu` |
| `--weight_init` | `-w_i` | `random` or `xavier` | `xavier` |
| `--wandb_project` | `-w_p` | W&B Project ID | `DA6401__Intro_to_DL_Assignment1` |
| `--model_path` | | Path to save model `.npy` | `src/best_model.npy` |
| `--config_save_path` | | Path to save config `.json` | `src/best_config.json` |

---

## Inference

```bash
python src/inference.py \
  --dataset mnist \
  --model_path src/best_model.npy \
  --num_layers 3 \
  --hidden_size "[128,128,128]" \
  --activation relu \
  --loss cross_entropy \
  --weight_init xavier
```

Outputs: **Accuracy**, **Precision**, **Recall**, and **F1-score** on the test set.

---

##  Best Model Performance

| Metric | Value |
|---|---|
| **Test Accuracy** | 97.93% |
| **Test F1-Score** | 97.90% |
| **Optimizer** | RMSProp |
| **Architecture** | 3 × 128 neurons, ReLU |
| **Learning Rate** | 0.001 |
| **Weight Init** | Xavier |
| **Loss Function** | Cross-Entropy |

---

##  Hyperparameter Sweep

A Bayesian sweep over 170 runs was conducted. To reproduce:

```bash
wandb sweep sweep_config.yaml
wandb agent <SWEEP_ID>
```

Key findings: **RMSProp > NAG ≈ Momentum >> SGD**. Learning rate of 0.001 and 3-layer ReLU architecture consistently topped the search space.

---



## Implementation Notes

- **Pure NumPy** — no PyTorch, TensorFlow, or autodiff libraries used
- Gradients are exposed via `layer.grad_W` and `layer.grad_b` after every `backward()` call
- Model outputs **logits** (pre-softmax), as required by the autograder
- `backward()` returns gradients from last layer to first
- Best model saved based on **test F1-score**, not validation accuracy

---

## Requirements

```
numpy
wandb
keras
scikit-learn
matplotlib
```
