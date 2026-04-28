"""
Visualize the three-step binarization and inversion preprocessing pipeline:
  Panel 1 — Raw 32×32 grayscale (with simulated scan noise + gradient)
  Panel 2 — Global binary threshold: noise eliminated, black note on white bg
  Panel 3 — Inverted binary: white note on black bg, 0.0/1.0 scale for CNN
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from preprocess import parse_strokes, render_to_canvas, CANVAS_SIZE, IMG_SIZE

# ── Render a real, dense quarter-note stroke ──────────────────────────────────
raw_file = PROJECT_ROOT / "HOMUS" / "HOMUS" / "71" / "71-88.txt"
canvas = render_to_canvas(parse_strokes(raw_file))   # 128×128, uint8

# ── Step 1: Add simulated scan artifacts before resizing ──────────────────────
#   • Gaussian noise  → mimics scanner sensor noise / compression artefacts
#   • Subtle gradient → uneven ink absorption across the page
rng = np.random.default_rng(42)
noise = rng.normal(0, 20, canvas.shape).astype(np.int16)

yy, xx = np.mgrid[0:CANVAS_SIZE, 0:CANVAS_SIZE]
gradient = ((xx / (CANVAS_SIZE - 1)) * 18 + (yy / (CANVAS_SIZE - 1)) * 14).astype(np.int16)

noisy_canvas = np.clip(canvas.astype(np.int16) + noise + gradient, 0, 255).astype(np.uint8)

# Resize to 32×32 with INTER_AREA — area averaging creates genuine grayscale
# intermediate values at stroke edges, giving the "grayscale artifacts" look
raw_32 = cv2.resize(noisy_canvas, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

# ── Step 2: Binarization — global Otsu, THRESH_BINARY ────────────────────────
#   THRESH_BINARY: pixels darker than T  → 0 (black)  = ink/stroke stays dark
#                  pixels brighter than T → 255 (white) = background stays light
#   Noise is eliminated because every pixel snaps to exactly 0 or 255
otsu_t, binary_32 = cv2.threshold(raw_32, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# ── Step 3: Inversion — THRESH_BINARY_INV (equivalent to 255 - binary_32) ───
#   Strokes become 255 (=1.0 when normalised to float)  → High Activation
#   Background becomes 0  (=0.0)                         → Null / Silent
inverted_32 = cv2.bitwise_not(binary_32)

# Locate annotation target points inside the inverted image
stroke_ys, stroke_xs = np.where(inverted_32 == 255)
bg_ys,     bg_xs     = np.where(inverted_32 == 0)

# Note centroid (annotation target for "1.0")
note_cx = int(np.median(stroke_xs))
note_cy = int(np.median(stroke_ys))

# Background corner far from the note
bg_cx, bg_cy = 2, 2   # top-left is reliably empty background

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 5), facecolor="#F0F4F8")
fig.suptitle(
    "Preprocessing Pipeline: Binarization & Inversion",
    fontsize=14, fontweight="bold", color="#2C3E50", y=1.03,
)

# ---- Panel 1: Raw grayscale --------------------------------------------------
axes[0].imshow(raw_32, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
axes[0].set_title("Raw Grayscale\n(Artifacts Present)",
                  fontsize=12, fontweight="bold", color="#555555", pad=9, linespacing=1.4)

# Pixel-value colorbar strip beneath panel 1 (shows the 0-255 range)
cax1 = axes[0].inset_axes([0.0, -0.12, 1.0, 0.07])
cax1.imshow(np.linspace(0, 255, 256).reshape(1, -1), cmap="gray",
            aspect="auto", vmin=0, vmax=255)
cax1.set_xticks([0, 128, 255])
cax1.set_xticklabels(["0", "128", "255"], fontsize=7.5)
cax1.set_yticks([])
cax1.set_title("Pixel value range", fontsize=7.5, pad=2, color="#666")

# Annotate one noisy background patch
axes[0].annotate(
    "Mixed\ngray values\n(noise)",
    xy=(1, 1), xytext=(9, 10),
    fontsize=7.5, color="#7D6608",
    arrowprops=dict(arrowstyle="-|>", color="#B7950B", lw=1.2),
    bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEF9E7", edgecolor="#F1C40F", alpha=0.9),
)

# ---- Panel 2: Binary (black note on white bg) --------------------------------
axes[1].imshow(binary_32, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
axes[1].set_title(f"Strict Binary\n(Noise Eliminated,  T={otsu_t:.0f})",
                  fontsize=12, fontweight="bold", color="#1A5276", pad=9, linespacing=1.4)
axes[1].set_facecolor("#EBF5FB")

# Annotate the eliminated noise region
axes[1].annotate(
    "Only 0 or 255\n— no grey noise",
    xy=(1, 1), xytext=(7, 10),
    fontsize=7.5, color="#1A5276",
    arrowprops=dict(arrowstyle="-|>", color="#2980B9", lw=1.2),
    bbox=dict(boxstyle="round,pad=0.25", facecolor="#EBF5FB", edgecolor="#2980B9", alpha=0.9),
)

# ---- Panel 3: Inverted (white note on black bg) ------------------------------
axes[2].imshow(inverted_32, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
axes[2].set_title("Inverted\n(Optimized for CNN)",
                  fontsize=12, fontweight="bold", color="#1D6A39", pad=9, linespacing=1.4)
axes[2].set_facecolor("#EAFAF1")

# Annotation: stroke → 1.0
offset_x = 10 if note_cx < 16 else -10
text_x    = note_cx + (8 if note_cx < 16 else -2)
axes[2].annotate(
    "1.0\n(High Activation)",
    xy=(note_cx, note_cy), xytext=(text_x, note_cy - 9),
    fontsize=7.5, color="#145A32", fontweight="bold",
    arrowprops=dict(arrowstyle="-|>", color="#27AE60", lw=1.4),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#EAFAF1", edgecolor="#27AE60", alpha=0.92),
)

# Annotation: background → 0.0
axes[2].annotate(
    "0.0\n(Null)",
    xy=(bg_cx, bg_cy), xytext=(bg_cx + 6, bg_cy + 10),
    fontsize=7.5, color="#4A235A",
    arrowprops=dict(arrowstyle="-|>", color="#8E44AD", lw=1.4),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5EEF8", edgecolor="#8E44AD", alpha=0.92),
)

# ---- Shared axis formatting --------------------------------------------------
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False)

# Colour the spines to match each panel's theme
spine_colors = ["#F0B912", "#2980B9", "#27AE60"]
for ax, sc in zip(axes, spine_colors):
    for spine in ax.spines.values():
        spine.set_linewidth(2.2)
        spine.set_color(sc)

# Step-number badges centred above each panel
badge_cfg = [("①", "#7D6608"), ("②", "#1A5276"), ("③", "#145A32")]
for ax, (badge, col) in zip(axes, badge_cfg):
    ax.text(0.5, 1.155, badge, transform=ax.transAxes,
            ha="center", va="center", fontsize=20, color=col, fontweight="bold")

plt.tight_layout(pad=1.8)
out_path = PROJECT_ROOT / "results" / "binarization_inversion_pipeline.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out_path}")
print(f"  Otsu threshold used: {otsu_t:.0f}")
print(f"  Note annotation at pixel ({note_cx}, {note_cy})")
