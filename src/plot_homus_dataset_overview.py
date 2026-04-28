"""HOMUS Dataset Overview: raw stroke panel + 8-class rendered image grid."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Reuse the exact rendering pipeline from preprocess.py ────────────────────
CANVAS_SIZE = 128
IMG_SIZE = 32
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
    all_points = [pt for stroke in strokes for pt in stroke]
    if not all_points:
        return canvas
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
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


def process_canvas(canvas):
    binary = cv2.adaptiveThreshold(
        canvas, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return cv2.resize(binary, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


# ── Class definitions ─────────────────────────────────────────────────────────
HOMUS_DIR = PROJECT_ROOT / "HOMUS" / "HOMUS"

CLASSES = [
    ("Whole-Note",   0, "Whole\nNote"),
    ("Half-Note",    1, "Half\nNote"),
    ("Quarter-Note", 2, "Quarter\nNote"),
    ("G-Clef",       3, "G Clef"),
    ("F-Clef",       4, "F Clef"),
    ("Quarter-Rest", 5, "Quarter\nRest"),
    ("Eighth-Rest",  6, "Eighth\nRest"),
    ("Common-Time",  7, "Common\nTime"),
]

# ── Pick one clean example per class from the test set ───────────────────────
X_test = np.load(PROJECT_ROOT / "data/processed/X_test_cnn.npy")   # (800,32,32,1)
y_test = np.load(PROJECT_ROOT / "data/processed/y_test.npy")

def pick_example(label_idx):
    """Return a 32×32 image (float 0-1) for the given class label."""
    idxs = np.where(y_test == label_idx)[0]
    imgs = X_test[idxs, :, :, 0]
    # Pick the sample closest to the median darkness (avoids blank or ultra-thick extremes)
    darkness = imgs.mean(axis=(1, 2))
    median_pos = np.argsort(darkness)[len(idxs) // 2]
    return imgs[median_pos]

# ── Raw stroke file for left panel: G-Clef (3 strokes, visually distinctive) ─
RAW_STROKE_FILE = HOMUS_DIR / "1" / "1-77.txt"
raw_strokes = parse_strokes(RAW_STROKE_FILE)

# ── Layout ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 7), facecolor="#F8F9FA")
fig.suptitle("HOMUS Dataset Overview", fontsize=17, fontweight="bold",
             color="#2C3E50", y=1.01)

outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.55], wspace=0.06)

# ── LEFT: Raw stroke plot ──────────────────────────────────────────────────────
ax_stroke = fig.add_subplot(outer[0])
ax_stroke.set_facecolor("#EBF5FB")
ax_stroke.set_aspect("equal")

stroke_palette = ["#E74C3C", "#2980B9", "#27AE60", "#8E44AD", "#F39C12"]

all_pts = [pt for s in raw_strokes for pt in s]
xs_all = [p[0] for p in all_pts]
ys_all = [p[1] for p in all_pts]
x_min_r, y_min_r = min(xs_all), min(ys_all)

for i, stroke in enumerate(raw_strokes):
    col = stroke_palette[i % len(stroke_palette)]
    sx = [p[0] - x_min_r for p in stroke]
    sy = [-(p[1] - y_min_r) for p in stroke]   # flip Y so top is top
    ax_stroke.plot(sx, sy, color=col, linewidth=2.2, solid_capstyle="round",
                   solid_joinstyle="round", label=f"Stroke {i + 1}")
    # Mark pen-down start
    ax_stroke.scatter(sx[0], sy[0], color=col, s=60, zorder=5, edgecolors="white", linewidths=0.8)

ax_stroke.set_title("Raw Data: Coordinate-Based Strokes\n(G Clef — 3 pen strokes)",
                     fontsize=12, fontweight="bold", color="#1A5276", pad=9)
ax_stroke.set_xlabel("X coordinate (pixels)", fontsize=10)
ax_stroke.set_ylabel("Y coordinate (pixels, flipped)", fontsize=10)
ax_stroke.legend(fontsize=9, loc="lower right", framealpha=0.85)
ax_stroke.spines[["top", "right"]].set_visible(False)
ax_stroke.tick_params(labelsize=9)

# ── RIGHT: 2×4 image grid ─────────────────────────────────────────────────────
right_gs = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=outer[1],
                                            hspace=0.55, wspace=0.18)

for idx, (raw_label, class_id, display_name) in enumerate(CLASSES):
    row, col = divmod(idx, 4)
    ax = fig.add_subplot(right_gs[row, col])

    img = pick_example(class_id)
    ax.imshow(img, cmap="binary", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(display_name, fontsize=9.5, fontweight="bold",
                 color="#1A5276", pad=4, linespacing=1.3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#2980B9")
        spine.set_linewidth(1.5)

# Section label above the grid
fig.text(0.57, 1.005, "The 8 Classes in Scope  (rendered 32×32)",
         ha="center", va="bottom", fontsize=12, fontweight="bold", color="#1A5276")

plt.savefig(PROJECT_ROOT / "results/homus_dataset_overview.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Saved: results/homus_dataset_overview.png")
