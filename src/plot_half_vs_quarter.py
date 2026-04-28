"""Find and display a thick Half note vs. a Quarter note from real test data."""
import numpy as np
import matplotlib.pyplot as plt

X_test = np.load("data/processed/X_test_cnn.npy")   # (800, 32, 32, 1)
y_test = np.load("data/processed/y_test.npy")        # (800,)  0-7

# Class indices: half-note=1, quarter-note=2
half_idx = np.where(y_test == 1)[0]
quarter_idx = np.where(y_test == 2)[0]

# Find the "thickest" half-note: one where the dark pixel ratio is highest
# (thick strokes leave little white space → high mean intensity)
half_images = X_test[half_idx, :, :, 0]
darkness = half_images.mean(axis=(1, 2))
thick_half_pos = half_idx[np.argmax(darkness)]

# Pick a median-darkness quarter note as a representative standard sample
quarter_images = X_test[quarter_idx, :, :, 0]
q_darkness = quarter_images.mean(axis=(1, 2))
mid_quarter_pos = quarter_idx[np.argsort(q_darkness)[len(quarter_idx) // 2]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
fig.suptitle(
    "Confusion Case: Thick Half Note vs. Quarter Note\n"
    "(32×32 resolution makes hollow center nearly invisible)",
    fontsize=12, fontweight="bold"
)

ax1.imshow(X_test[thick_half_pos, :, :, 0], cmap="gray_r", vmin=0, vmax=1)
ax1.set_title("Thick Half Note  (Target: Half)", fontsize=11, fontweight="bold", color="#C0392B")
ax1.axis("off")

ax2.imshow(X_test[mid_quarter_pos, :, :, 0], cmap="gray_r", vmin=0, vmax=1)
ax2.set_title("Quarter Note  (Target: Quarter)", fontsize=11, fontweight="bold", color="#1A5276")
ax2.axis("off")

plt.tight_layout()
plt.savefig("results/half_vs_quarter_resolution_issue.png", dpi=150, bbox_inches="tight")
print(f"Saved: results/half_vs_quarter_resolution_issue.png")
print(f"  Half note sample index : {thick_half_pos}  (mean px={darkness.max():.3f})")
print(f"  Quarter note sample idx: {mid_quarter_pos}")
