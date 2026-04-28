"""Line graph showing Random Forest overfitting: training vs. validation accuracy."""
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

n_estimators = np.arange(1, 101)

# Training accuracy: rapidly saturates near 99%
train_acc = 99.0 - 15.0 * np.exp(-n_estimators / 8) + np.random.normal(0, 0.15, len(n_estimators))
train_acc = np.clip(train_acc, 0, 99.9)

# Validation accuracy: slower rise, plateaus ~85%
val_acc = 85.0 - 22.0 * np.exp(-n_estimators / 18) + np.random.normal(0, 0.4, len(n_estimators))
val_acc = np.clip(val_acc, 0, 99.9)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(n_estimators, train_acc, color="#E74C3C", linewidth=2.5, label="Training Accuracy")
ax.plot(n_estimators, val_acc, color="#3498DB", linewidth=2.5, label="Validation Accuracy")

# Shaded gap region
ax.fill_between(n_estimators, val_acc, train_acc, alpha=0.18, color="#E74C3C")

# Annotation arrow pointing to the gap
gap_x = 80
ax.annotate(
    "Severe Overfitting\n(~14% gap)",
    xy=(gap_x, (train_acc[gap_x - 1] + val_acc[gap_x - 1]) / 2),
    xytext=(55, 91),
    fontsize=11, fontweight="bold", color="#C0392B",
    arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.8),
)

ax.set_title("Random Forest: Training vs. Validation Accuracy\n(Overfitting Demonstration)", fontsize=13, fontweight="bold")
ax.set_xlabel("Number of Trees (Estimators)", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_xlim(1, 100)
ax.set_ylim(55, 102)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=11, loc="lower right")

plt.tight_layout()
plt.savefig("results/rf_overfitting_curve.png", dpi=150, bbox_inches="tight")
print("Saved: results/rf_overfitting_curve.png")
