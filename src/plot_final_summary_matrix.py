"""Render a visually styled summary comparison table as a PNG."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

columns = ["Spatial\nContext?", "Overfitting\nRisk", "Inference\nSpeed", "Final\nAccuracy", "Verdict"]
rows = ["Logistic\nRegression", "Random\nForest", "CNN"]

cell_data = [
    ["No",  "Low",    "Fast\n(~2 ms)",  "78.0%", "Baseline — fast\nbut limited"],
    ["No",  "HIGH",   "Medium\n(~15 ms)", "85.0%", "Better accuracy,\nsevere overfitting"],
    ["Yes", "Low",    "Slow\n(~45 ms)",  "92.3%", "Best overall —\nunderstands structure"],
]

# Cell background colours
default_bg = "#FFFFFF"
cnn_bg = "#D5F5E3"        # light green for CNN row
lr_speed_bg = "#D6EAF8"   # light blue for LR inference speed cell

fig, ax = plt.subplots(figsize=(13, 4.5))
ax.axis("off")
fig.patch.set_facecolor("#F8F9FA")

n_rows, n_cols = len(rows), len(columns)
col_w = 1.0 / (n_cols + 1)   # extra col for row labels
row_h = 1.0 / (n_rows + 1)   # extra row for header

header_bg = "#2C3E50"
header_fg = "white"
row_label_bg = "#34495E"

def draw_cell(ax, x, y, w, h, text, bg, fg="black", fontsize=10, bold=False):
    rect = patches.FancyBboxPatch(
        (x + 0.003, y + 0.01), w - 0.006, h - 0.02,
        boxstyle="round,pad=0.01", facecolor=bg, edgecolor="#BDC3C7", linewidth=1.2,
        transform=ax.transAxes, clip_on=False
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize,
        color=fg, fontweight="bold" if bold else "normal",
        transform=ax.transAxes
    )

# Header row
draw_cell(ax, 0, 1 - row_h, col_w, row_h, "Model", header_bg, header_fg, bold=True)
for j, col in enumerate(columns):
    draw_cell(ax, col_w + j * col_w, 1 - row_h, col_w, row_h, col, header_bg, header_fg, bold=True)

# Data rows
for i, (row_label, row_vals) in enumerate(zip(rows, cell_data)):
    y = 1 - (i + 2) * row_h
    is_cnn = (i == 2)
    row_bg = cnn_bg if is_cnn else default_bg
    draw_cell(ax, 0, y, col_w, row_h, row_label, row_label_bg, header_fg, bold=True)
    for j, val in enumerate(row_vals):
        # Highlight LR inference speed cell
        bg = row_bg
        if i == 0 and j == 2:
            bg = lr_speed_bg
        # Bold/red for HIGH overfitting
        bold = (val == "HIGH")
        fg = "#C0392B" if val == "HIGH" else ("black" if not is_cnn else "#1A5276")
        draw_cell(ax, col_w + j * col_w, y, col_w, row_h, val, bg, fg, bold=bold)

# Legend
ax.text(0.01, 0.02,
        "  Light green = CNN (overall winner)    Light blue = Logistic Regression (fastest inference)",
        ha="left", va="bottom", fontsize=9, color="#555555", transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="#EAECEE", edgecolor="#BDC3C7"))

ax.set_title("Final Model Comparison Summary", fontsize=14, fontweight="bold",
             color="#2C3E50", pad=12)

plt.savefig("results/final_summary_matrix.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved: results/final_summary_matrix.png")
