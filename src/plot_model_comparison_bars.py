"""Bar charts: Test Accuracy and Inference Time for all three models."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

models = ["Logistic\nRegression", "Random\nForest", "CNN"]
accuracies = [78.0, 85.0, 92.32]
times_ms = [2.0, 15.0, 45.0]

bar_colors_acc = ["#5B9BD5", "#5B9BD5", "#2ECC71"]
bar_colors_time = ["#2ECC71", "#F39C12", "#E74C3C"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("Model Comparison: CNN vs. Traditional ML", fontsize=15, fontweight="bold", y=1.01)

# --- Left: Test Accuracy ---
x = np.arange(len(models))
bars = ax1.bar(x, accuracies, width=0.5, color=bar_colors_acc, edgecolor="white", linewidth=1.2)
ax1.set_title("Test Accuracy", fontsize=13, fontweight="bold", pad=10)
ax1.set_ylabel("Accuracy (%)", fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=11)
ax1.set_ylim(60, 100)
ax1.yaxis.grid(True, linestyle="--", alpha=0.6)
ax1.set_axisbelow(True)
ax1.spines[["top", "right"]].set_visible(False)

for bar, val in zip(bars, accuracies):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{val:.2f}%",
        ha="center", va="bottom", fontsize=11, fontweight="bold"
    )

winner_patch = mpatches.Patch(color="#2ECC71", label="CNN — Best Accuracy")
ax1.legend(handles=[winner_patch], loc="lower right", fontsize=9)

# --- Right: Inference Time ---
bars2 = ax2.bar(x, times_ms, width=0.5, color=bar_colors_time, edgecolor="white", linewidth=1.2)
ax2.set_title("Inference Time (per sample)", fontsize=13, fontweight="bold", pad=10)
ax2.set_ylabel("Time (ms)", fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=11)
ax2.set_ylim(0, 60)
ax2.yaxis.grid(True, linestyle="--", alpha=0.6)
ax2.set_axisbelow(True)
ax2.spines[["top", "right"]].set_visible(False)

for bar, val in zip(bars2, times_ms):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.8,
        f"{val:.0f} ms",
        ha="center", va="bottom", fontsize=11, fontweight="bold"
    )

fast_patch = mpatches.Patch(color="#2ECC71", label="LR — Fastest")
ax2.legend(handles=[fast_patch], loc="upper left", fontsize=9)

plt.tight_layout()
plt.savefig("results/model_comparison_bars.png", dpi=150, bbox_inches="tight")
print("Saved: results/model_comparison_bars.png")
