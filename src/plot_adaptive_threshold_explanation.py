"""
Explain why Adaptive Gaussian Thresholding beats a global (Otsu) threshold.

Uses a real HOMUS quarter-note stroke file, adds a synthetic uneven-lighting
gradient, then shows three panels:
  1. Original (shadowed canvas)
  2. Global Otsu threshold  → fails: shadow background becomes a white blob
  3. Adaptive Gaussian      → succeeds: note extracted cleanly despite shadow
"""
from pathlib import Path
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Rendering helpers (mirrors preprocess.py exactly) ─────────────────────────
CANVAS_SIZE = 128
STROKE_THICKNESS = 3
PADDING = 10


def parse_strokes(filepath):
    lines = Path(filepath).read_text(errors="ignore").strip().split("\n")
    strokes = []
    for line in lines[1:]:
        points = []
        for pair in line.strip(";").split(";"):
            pair = pair.strip()
            if "," in pair:
                parts = pair.split(",")
                try:
                    points.append((int(parts[0]), int(parts[1])))
                except ValueError:
                    continue
        if points:
            strokes.append(points)
    return strokes


def render_to_canvas(strokes):
    canvas = np.full((CANVAS_SIZE, CANVAS_SIZE), 255, dtype=np.uint8)
    all_points = [pt for s in strokes for pt in s]
    if not all_points:
        return canvas
    xs, ys = [p[0] for p in all_points], [p[1] for p in all_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    draw_area = CANVAS_SIZE - 2 * PADDING
    scale = draw_area / max(x_max - x_min or 1, y_max - y_min or 1)
    x_off = PADDING + (draw_area - (x_max - x_min) * scale) / 2
    y_off = PADDING + (draw_area - (y_max - y_min) * scale) / 2
    for stroke in strokes:
        pts = np.array(
            [[int((x - x_min) * scale + x_off), int((y - y_min) * scale + y_off)]
             for x, y in stroke], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, 0, STROKE_THICKNESS)
        else:
            cv2.circle(canvas, tuple(pts[0]), STROKE_THICKNESS, 0, -1)
    return canvas


# ── 1. Render a real quarter note (dense 2-stroke file) ──────────────────────
raw_file = PROJECT_ROOT / "HOMUS" / "HOMUS" / "71" / "71-88.txt"
strokes = parse_strokes(raw_file)
canvas = render_to_canvas(strokes)   # 128×128, white bg, black strokes

# ── 2. Add a synthetic bottom-right shadow ───────────────────────────────────
#   gradient: 0 at top-left → 200 at bottom-right (power curve for sharp onset)
yy, xx = np.mgrid[0:CANVAS_SIZE, 0:CANVAS_SIZE]
gradient = ((xx / (CANVAS_SIZE - 1)) * (yy / (CANVAS_SIZE - 1))) ** 0.65 * 210
gradient = gradient.astype(np.int32)

# Subtract from the canvas (white background drops to ~45; black strokes stay 0)
shadowed = np.clip(canvas.astype(np.int32) - gradient, 0, 255).astype(np.uint8)

# ── 3. Global Otsu threshold (BINARY_INV: strokes → white, bg → black) ───────
_, global_thresh = cv2.threshold(
    shadowed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)
otsu_val = cv2.threshold(shadowed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[0]

# ── 4. Adaptive Gaussian threshold (exactly as in preprocess.py) ─────────────
adaptive_thresh = cv2.adaptiveThreshold(
    shadowed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
)

# ── 5. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), facecolor="#F0F4F8")
fig.suptitle(
    "Why Adaptive Gaussian Thresholding?\n"
    "A Global Threshold Fails Under Uneven Lighting",
    fontsize=14, fontweight="bold", color="#2C3E50", y=1.02,
)

panel_cfg = [
    (shadowed,        "Original with Uneven Lighting",        "gray",    "#2C3E50", None),
    (global_thresh,   "Global Threshold\n(Fails in Shadows)", "binary",  "#C0392B", "#FDEDEC"),
    (adaptive_thresh, "Adaptive Gaussian\n(Preserves Details)","binary", "#1A5276", "#EBF5FB"),
]

for ax, (img, title, cmap, title_color, bg) in zip(axes, panel_cfg):
    if bg:
        ax.set_facecolor(bg)
    ax.imshow(img, cmap=cmap, vmin=0, vmax=255, interpolation="nearest")
    ax.set_title(title, fontsize=12, fontweight="bold", color=title_color,
                 pad=10, linespacing=1.4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color(title_color)

# Annotation on the global panel: point at the blob in the bottom-right
axes[1].annotate(
    f"Shadow background\nmisclassified as\nforeground (blob!)\n[Otsu T={otsu_val:.0f}]",
    xy=(100, 100), xytext=(48, 42),
    fontsize=8.5, color="#922B21", fontweight="bold",
    arrowprops=dict(arrowstyle="-|>", color="#C0392B", lw=1.5),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDEDEC", edgecolor="#C0392B", alpha=0.92),
)

# Annotation on the adaptive panel: highlight clean result
axes[2].annotate(
    "Note cleanly\nextracted despite\nlocal shadow!",
    xy=(72, 95), xytext=(6, 40),
    fontsize=8.5, color="#1A5276", fontweight="bold",
    arrowprops=dict(arrowstyle="-|>", color="#2980B9", lw=1.5),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#EBF5FB", edgecolor="#2980B9", alpha=0.92),
)

# Add a subtle gradient bar below the left image as a legend
cbar_ax = fig.add_axes([0.065, -0.04, 0.27, 0.025])
grad_bar = np.linspace(255, 45, 256).reshape(1, -1)
cbar_ax.imshow(grad_bar, cmap="gray", aspect="auto", vmin=0, vmax=255)
cbar_ax.set_xticks([0, 255])
cbar_ax.set_xticklabels(["Bright (top-left)", "Shadow (bottom-right)"], fontsize=8)
cbar_ax.set_yticks([])
cbar_ax.set_title("Lighting gradient applied", fontsize=8, pad=3, color="#555")
for spine in cbar_ax.spines.values():
    spine.set_linewidth(0.8)

plt.tight_layout(pad=1.5)
out_path = PROJECT_ROOT / "results" / "adaptive_threshold_explanation.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out_path}")
