"""
Run once after updating neural_network.py.
Embeds '_activation' into best_model.npy so set_weights() always works.

Run from repo root:
    python scripts/resave_best_model.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import numpy as np
import json

MODEL_PATH  = "src/best_model.npy"
CONFIG_PATH = "src/best_config.json"

# Load existing weights
weights = np.load(MODEL_PATH, allow_pickle=True).item()
print("Keys before fix:", [k for k in weights.keys()])

# Load config to find the correct activation
with open(CONFIG_PATH) as f:
    cfg = json.load(f)
activation = cfg.get("activation", "relu")
print(f"Activation from config: {activation}")

# Embed activation and resave
weights["_activation"] = activation
np.save(MODEL_PATH, weights)
print(f"\nResaved {MODEL_PATH} with '_activation'='{activation}'")

# ── Verify the fix ────────────────────────────────────────────────────────
from ann.neural_network import NeuralNetwork

class WrongCli:
    """Simulate autograder passing wrong activation"""
    activation  = "sigmoid"   # WRONG on purpose
    loss        = "cross_entropy"
    optimizer   = "rmsprop"
    learning_rate = 0.001
    weight_decay  = 0.0
    num_layers    = 3
    hidden_size   = [128, 128, 128]
    weight_init   = "xavier"

model = NeuralNetwork(WrongCli(), input_dim=784, num_classes=10)
loaded = np.load(MODEL_PATH, allow_pickle=True).item()
model.set_weights(loaded)

print(f"\nActivation after set_weights : {model.activation_name}")
print(f"Expected                     : {activation}")
assert model.activation_name == activation, "FIX FAILED — activation still wrong!"
print("✅ Fix verified — activation is correctly restored from saved weights")

# Quick forward pass sanity check
X = np.random.randn(10, 784)
logits = model.forward(X)
preds  = np.argmax(logits, axis=1)
print(f"\nForward pass OK — logits shape: {logits.shape}")
print(f"Predictions (should vary): {preds}")
print(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")

if logits.max() - logits.min() < 0.1:
    print("⚠️  WARNING: logits range is suspiciously small — check your model!")
else:
    print("✅ Logits look healthy")

print("\nDone. Commit and push src/best_model.npy + src/best_config.json")