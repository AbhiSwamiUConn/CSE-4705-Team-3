"""Conceptual diagram: 2D spatial grid (CNN input) vs. 1D flattened vector (traditional ML)."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig = plt.figure(figsize=(13, 8))
fig.patch.set_facecolor("#F8F9FA")

# ── 2D Grid (top half) ────────────────────────────────────────────────────────
ax_2d = fig.add_axes([0.05, 0.52, 0.55, 0.42])
ax_2d.set_xlim(0, 32)
ax_2d.set_ylim(0, 32)
ax_2d.set_aspect("equal")
ax_2d.set_facecolor("#EBF5FB")
ax_2d.spines[:].set_color("#2980B9")
for spine in ax_2d.spines.values():
    spine.set_linewidth(2)

# Draw grid lines
for i in range(0, 33, 4):
    ax_2d.axhline(i, color="#AED6F1", lw=0.5)
    ax_2d.axvline(i, color="#AED6F1", lw=0.5)

# Draw a simple music note shape on the grid
note_col = "#1A5276"
# Note head (ellipse-ish filled squares at ~(14,6) area)
for r, c in [(6, 14), (6, 15), (7, 13), (7, 14), (7, 15), (7, 16), (8, 14), (8, 15)]:
    rect = patches.Rectangle((c, r), 1, 1, facecolor=note_col, edgecolor="none")
    ax_2d.add_patch(rect)
# Stem
for r in range(8, 22):
    rect = patches.Rectangle((16, r), 1, 1, facecolor=note_col, edgecolor="none")
    ax_2d.add_patch(rect)
# Flag
for c in range(16, 22):
    rect = patches.Rectangle((c, 22 - (c - 16)), 1, 1, facecolor=note_col, edgecolor="none")
    ax_2d.add_patch(rect)

ax_2d.set_xticks([])
ax_2d.set_yticks([])
ax_2d.set_title("CNN Input: Preserved 2D Spatial Context (32×32)", fontsize=12, fontweight="bold",
                 color="#1A5276", pad=8)

# ── 1D Vector (bottom half) ───────────────────────────────────────────────────
ax_1d = fig.add_axes([0.05, 0.08, 0.90, 0.14])
ax_1d.set_xlim(0, 128)
ax_1d.set_ylim(0, 1)
ax_1d.set_aspect("auto")
ax_1d.set_facecolor("#FDFEFE")
ax_1d.spines[:].set_color("#E74C3C")
for spine in ax_1d.spines.values():
    spine.set_linewidth(2)

rng = np.random.default_rng(7)
colors_1d = rng.choice(["#1A5276", "#D6EAF8"], size=128, p=[0.18, 0.82])
for i, col in enumerate(colors_1d):
    rect = patches.Rectangle((i, 0.05), 0.85, 0.9, facecolor=col, edgecolor="#AED6F1", lw=0.3)
    ax_1d.add_patch(rect)

ax_1d.set_xticks([])
ax_1d.set_yticks([])
ax_1d.set_title(
    "Traditional ML Input: Flattened 1D Vector  (1,024 independent pixels — spatial structure destroyed)",
    fontsize=11, fontweight="bold", color="#922B21", pad=8
)

# ── Breakdown arrows (2D → 1D) ────────────────────────────────────────────────
ax_arrow = fig.add_axes([0.0, 0.0, 1.0, 1.0])
ax_arrow.set_xlim(0, 1)
ax_arrow.set_ylim(0, 1)
ax_arrow.axis("off")

arrow_props = dict(arrowstyle="-|>", color="#7D3C98", lw=2,
                   mutation_scale=18, connectionstyle="arc3,rad=0.3")

# Three arrows fanning from the bottom of the 2D grid to the top of the 1D strip
for start_x, end_x in [(0.20, 0.10), (0.33, 0.50), (0.50, 0.88)]:
    ax_arrow.annotate(
        "", xy=(end_x, 0.23), xytext=(start_x, 0.52),
        arrowprops=arrow_props,
    )

ax_arrow.text(0.65, 0.38, "Spatial\nstructure\ndestroyed", ha="center", va="center",
              fontsize=10, color="#7D3C98", fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5EEF8", edgecolor="#7D3C98"))

fig.suptitle("Why CNNs Win: 2D Spatial Context vs. Flattened 1D Input",
             fontsize=14, fontweight="bold", y=0.99)

plt.savefig("results/1d_vs_2d_diagram.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Saved: results/1d_vs_2d_diagram.png")
