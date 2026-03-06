"""Section 2.8: Error Analysis — Confusion Matrix + Creative Failure Visualization"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix
import wandb
from src.utils.data_loader import load_dataset
from src.ann.neural_network import NeuralNetwork
import json

# ── Load best model ────────────────────────────────────────────────────────
class Args:
    dataset = "mnist"; epochs = 50; batch_size = 128
    optimizer = "rmsprop"; learning_rate = 0.001; weight_decay = 0.0
    num_layers = 3; hidden_size = [128, 128, 128]; activation = "relu"
    loss = "cross_entropy"; weight_init = "xavier"; wandb_project = None
    model_path = "src/best_model.npy"; config_save_path = "src/best_config.json"

args = Args()
model = NeuralNetwork(args, input_dim=784, num_classes=10)
weights = np.load("src/best_model.npy", allow_pickle=True).item()
model.set_weights(weights)

# ── Load test data ─────────────────────────────────────────────────────────
data = load_dataset("mnist")
X_test, y_test_onehot, y_test_labels = data[4], data[5], data[6]

# ── Get predictions ────────────────────────────────────────────────────────
logits = model.forward(X_test)
y_pred = np.argmax(logits, axis=1)
y_true = y_test_labels

# ── 1. Confusion Matrix ────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
class_names = [str(i) for i in range(10)]

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im, ax=ax)
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xticklabels(class_names, fontsize=12)
ax.set_yticklabels(class_names, fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=13)
ax.set_ylabel('True Label', fontsize=13)
ax.set_title('Confusion Matrix — Best Model on MNIST Test Set', fontsize=14)

# Annotate cells
thresh = cm.max() / 2.0
for i in range(10):
    for j in range(10):
        ax.text(j, i, str(cm[i, j]),
                ha='center', va='center', fontsize=9,
                color='white' if cm[i, j] > thresh else 'black')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved confusion_matrix.png")

# ── 2. Creative: Most Confused Pairs — show actual failure images ──────────
# Find off-diagonal errors sorted by frequency
errors = []
for true_c in range(10):
    for pred_c in range(10):
        if true_c != pred_c and cm[true_c, pred_c] > 0:
            errors.append((cm[true_c, pred_c], true_c, pred_c))
errors.sort(reverse=True)
top_pairs = errors[:6]  # top 6 most confused pairs

print("\nTop confused pairs:")
for count, t, p in top_pairs:
    print(f"  True={t} → Predicted={p}: {count} mistakes")

fig2, axes = plt.subplots(6, 5, figsize=(12, 14))
fig2.suptitle('Creative Failure Analysis: Top 6 Most Confused Digit Pairs\n'
              '(Each row = one confused pair, showing 5 misclassified examples)',
              fontsize=13, fontweight='bold')

for row_idx, (count, true_c, pred_c) in enumerate(top_pairs):
    # Find indices where true=true_c but predicted=pred_c
    mistake_idx = np.where((y_true == true_c) & (y_pred == pred_c))[0]

    # Get softmax probabilities for confidence
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)

    # Sort by confidence of wrong prediction (most confident mistakes first)
    if len(mistake_idx) > 0:
        confidences = probs[mistake_idx, pred_c]
        sorted_idx  = mistake_idx[np.argsort(-confidences)]
        show_idx    = sorted_idx[:5]
    else:
        show_idx = []

    for col_idx in range(5):
        ax = axes[row_idx, col_idx]
        if col_idx < len(show_idx):
            img = X_test[show_idx[col_idx]].reshape(28, 28)
            conf = probs[show_idx[col_idx], pred_c]
            ax.imshow(img, cmap='gray')
            ax.set_title(f'True:{true_c}→Pred:{pred_c}\n{conf:.1%}', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
        ax.axis('off')

    # Add row label
    axes[row_idx, 0].set_ylabel(f'{count} errors\nTrue {true_c}→{pred_c}',
                                 fontsize=8, rotation=0, labelpad=60, va='center')

plt.tight_layout()
plt.savefig('failure_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved failure_analysis.png")

# ── 3. Log both to W&B ─────────────────────────────────────────────────────
run = wandb.init(
    project="DA6401__Intro_to_DL_Assignment1",
    name="2.8_Error_Analysis",
    group="2.8_Error_Analysis"
)
run.log({
    "confusion_matrix":   wandb.Image("confusion_matrix.png"),
    "failure_analysis":   wandb.Image("failure_analysis.png"),
})

# Also log as W&B confusion matrix (native)
run.log({"conf_matrix_native": wandb.plot.confusion_matrix(
    preds=y_pred.tolist(),
    y_true=y_true.tolist(),
    class_names=class_names
)})

# Per-class accuracy
print("\nPer-class accuracy:")
for c in range(10):
    mask = y_true == c
    acc  = np.mean(y_pred[mask] == c)
    print(f"  Digit {c}: {acc:.4f} ({np.sum(mask)} samples)")

run.finish()
print("✅ Logged to W&B")