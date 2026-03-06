"""Section 2.7: Global Performance Analysis - Fixed for sweep run naming"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
import numpy as np
import matplotlib.pyplot as plt

api = wandb.Api()
entity  = "chinthavishnuvardhan4-indian-institute-of-technology-madras"
project = "DA6401__Intro_to_DL_Assignment1"

print("Fetching all runs...")
runs = api.runs(f"{entity}/{project}", per_page=300)

train_accs, val_accs, run_names, is_overfit = [], [], [], []
skipped = 0

for run in runs:
    if run.state != "finished":
        skipped += 1
        continue

    s = run.summary

    # Try multiple possible key names for train accuracy
    ta = (s.get("train_accuracy") or
          s.get("train_acc") or
          s.get("best_train_accuracy") or
          None)

    # Try multiple possible key names for val accuracy
    va = (s.get("val_accuracy") or
          s.get("val_acc") or
          s.get("best_val_accuracy") or
          None)

    if ta is None or va is None:
        skipped += 1
        continue

    try:
        ta = float(ta)
        va = float(va)
    except (TypeError, ValueError):
        skipped += 1
        continue

    # Sanity check — must be valid accuracy values
    if not (0.0 < ta <= 1.0 and 0.0 < va <= 1.0):
        skipped += 1
        continue

    train_accs.append(ta)
    val_accs.append(va)
    run_names.append(run.name)
    gap = ta - va
    is_overfit.append(gap > 0.05)

print(f"Runs included : {len(train_accs)}")
print(f"Runs skipped  : {skipped}")

if len(train_accs) == 0:
    print("\nNo runs found! Checking what keys exist in a sample run...")
    for run in api.runs(f"{entity}/{project}", per_page=5):
        if run.state == "finished":
            print(f"\nRun: {run.name}")
            print(f"Summary keys: {list(run.summary.keys())}")
            break
    sys.exit(1)

train_accs = np.array(train_accs)
val_accs   = np.array(val_accs)
is_overfit = np.array(is_overfit)

# ── Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 9))

normal_mask  = ~is_overfit
overfit_mask = is_overfit

ax.scatter(train_accs[normal_mask],  val_accs[normal_mask],
           c='steelblue', alpha=0.6, s=70,
           label=f'Normal runs: {normal_mask.sum()}', zorder=3)

if overfit_mask.sum() > 0:
    ax.scatter(train_accs[overfit_mask], val_accs[overfit_mask],
               c='crimson', alpha=0.75, s=90, marker='D',
               label=f'Overfit runs (gap > 5%): {overfit_mask.sum()}', zorder=4)

# y = x line
lo = max(min(train_accs.min(), val_accs.min()) - 0.03, 0)
hi = min(max(train_accs.max(), val_accs.max()) + 0.02, 1.01)
ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, alpha=0.5,
        label='Perfect generalization (y=x)')
ax.fill_between([lo, hi], [lo, hi], [lo, lo],
                alpha=0.04, color='red', label='Overfitting zone')

# Annotate best val run
best_idx = int(np.argmax(val_accs))
ax.annotate(
    f'Best val: {run_names[best_idx][:25]}\nVal={val_accs[best_idx]:.4f}',
    xy=(train_accs[best_idx], val_accs[best_idx]),
    xytext=(train_accs[best_idx] - 0.1, val_accs[best_idx] - 0.06),
    fontsize=8, color='darkgreen',
    arrowprops=dict(arrowstyle='->', color='darkgreen')
)

# Annotate worst overfit run
if overfit_mask.sum() > 0:
    gaps = train_accs - val_accs
    wo_idx = int(np.argmax(gaps[overfit_mask]))
    wo_ta  = train_accs[overfit_mask][wo_idx]
    wo_va  = val_accs[overfit_mask][wo_idx]
    wo_nm  = np.array(run_names)[overfit_mask][wo_idx]
    ax.annotate(
        f'Most overfit: {wo_nm[:20]}\nGap={wo_ta-wo_va:.3f}',
        xy=(wo_ta, wo_va),
        xytext=(wo_ta - 0.12, wo_va + 0.04),
        fontsize=8, color='crimson',
        arrowprops=dict(arrowstyle='->', color='crimson')
    )

ax.set_xlabel('Training Accuracy', fontsize=13)
ax.set_ylabel('Validation Accuracy', fontsize=13)
ax.set_title(
    f'Training vs Validation Accuracy — All {len(train_accs)} Runs\n'
    f'Red diamonds = overfit runs (train − val gap > 5%)',
    fontsize=13
)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim([lo, hi])
ax.set_ylim([max(val_accs.min()-0.03, 0), hi])
plt.tight_layout()
plt.savefig('global_performance.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved global_performance.png")

# ── Terminal stats ─────────────────────────────────────────────────────────
gaps = train_accs - val_accs
print(f"\nTotal runs analyzed : {len(train_accs)}")
print(f"Overfit runs        : {overfit_mask.sum()}")
print(f"Best val accuracy   : {val_accs.max():.4f}  ({run_names[np.argmax(val_accs)]})")
print(f"Mean val accuracy   : {val_accs.mean():.4f}")
print(f"Mean train-val gap  : {gaps.mean():.4f}")

top5 = np.argsort(-gaps)[:5]
print("\nTop 5 most overfit runs:")
for i in top5:
    print(f"  {run_names[i]:<35} train={train_accs[i]:.4f}  "
          f"val={val_accs[i]:.4f}  gap={gaps[i]:.4f}")

# ── Log to W&B ─────────────────────────────────────────────────────────────
log_run = wandb.init(
    project="DA6401__Intro_to_DL_Assignment1",
    name="2.7_Global_Performance_Analysis",
    group="2.7_Global_Analysis"
)
log_run.log({"global_performance_plot": wandb.Image("global_performance.png")})

table = wandb.Table(columns=["run_name", "train_accuracy",
                              "val_accuracy", "gap", "overfit"])
for i in range(len(train_accs)):
    table.add_data(run_names[i], float(train_accs[i]),
                   float(val_accs[i]), float(gaps[i]), bool(is_overfit[i]))
log_run.log({"all_runs_table": table})
log_run.finish()
print("✅ Logged to W&B")