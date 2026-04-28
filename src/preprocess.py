from pathlib import Path
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CANVAS_SIZE = 128
IMG_SIZE = 32
STROKE_THICKNESS = 3
PADDING = 10

CLASS_LABELS = {
    # Original 3 note classes
    "whole-note":   0,
    "half-note":    1,
    "quarter-note": 2,
    # New 5 classes
    "g-clef":       3,
    "f-clef":       4,
    "quarter-rest": 5,
    "eighth-rest":  6,
    "common-time":  7,
}


def parse_strokes(filepath: Path) -> list[list[tuple[int, int]]]:
    lines = filepath.read_text(errors="ignore").strip().split("\n")
    strokes = []
    for line in lines[1:]:  # skip class-label line
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


def render_to_canvas(strokes: list[list[tuple[int, int]]]) -> np.ndarray:
    canvas = np.full((CANVAS_SIZE, CANVAS_SIZE), 255, dtype=np.uint8)

    all_points = [pt for stroke in strokes for pt in stroke]
    if not all_points:
        return canvas

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    draw_area = CANVAS_SIZE - 2 * PADDING
    x_range = x_max - x_min or 1
    y_range = y_max - y_min or 1
    scale = draw_area / max(x_range, y_range)

    # Center within the padded draw area
    x_off = PADDING + (draw_area - x_range * scale) / 2
    y_off = PADDING + (draw_area - y_range * scale) / 2

    for stroke in strokes:
        pts = np.array(
            [
                [int((x - x_min) * scale + x_off), int((y - y_min) * scale + y_off)]
                for x, y in stroke
            ],
            dtype=np.int32,
        )
        if len(pts) >= 2:
            cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, 0, STROKE_THICKNESS)
        else:
            cv2.circle(canvas, tuple(pts[0]), STROKE_THICKNESS, 0, -1)

    return canvas


def process_canvas(canvas: np.ndarray) -> np.ndarray:
    # Adaptive Gaussian threshold: dark strokes → white (255), background → black (0)
    binary = cv2.adaptiveThreshold(
        canvas, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    return cv2.resize(binary, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def load_all_samples() -> tuple[np.ndarray, np.ndarray]:
    images, labels = [], []
    total = sum(len(list((RAW_DIR / cls).glob("*.txt"))) for cls in CLASS_LABELS)
    processed = 0

    for cls, label in CLASS_LABELS.items():
        cls_dir = RAW_DIR / cls
        if not cls_dir.exists():
            print(f"  WARNING: {cls_dir} not found, skipping.")
            continue

        files = sorted(cls_dir.glob("*.txt"))
        print(f"  Loading {len(files)} {cls} files...")

        for filepath in files:
            strokes = parse_strokes(filepath)
            if not strokes:
                continue
            canvas = render_to_canvas(strokes)
            img = process_canvas(canvas)
            images.append(img)
            labels.append(label)

            processed += 1
            if processed % 100 == 0:
                print(f"    {processed}/{total} files processed...")

    return np.array(images, dtype=np.float32) / 255.0, np.array(labels, dtype=np.int32)


def ascii_preview(img_2d: np.ndarray, threshold: float = 0.3) -> str:
    chars = " .:#"
    rows = []
    for row in img_2d:
        line = ""
        for val in row:
            if val > 0.7:
                line += chars[3]
            elif val > 0.3:
                line += chars[2]
            elif val > 0.05:
                line += chars[1]
            else:
                line += chars[0]
        rows.append(line)
    return "\n".join(rows)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and rendering samples...")
    X, y = load_all_samples()

    print(f"\nRaw dataset: {X.shape[0]} samples, class counts: "
          + ", ".join(f"{cls}={np.sum(y==lbl)}" for cls, lbl in CLASS_LABELS.items()))

    # Stratified 80/20 split to maintain class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4D arrays for CNN: (samples, 32, 32, 1)
    X_train_cnn = X_train[..., np.newaxis]
    X_test_cnn = X_test[..., np.newaxis]

    # 2D arrays for flat models: (samples, 1024)
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)

    print("\nSaving to data/processed/...")
    np.save(PROCESSED_DIR / "X_train_cnn.npy", X_train_cnn)
    np.save(PROCESSED_DIR / "X_test_cnn.npy", X_test_cnn)
    np.save(PROCESSED_DIR / "X_train_flat.npy", X_train_flat)
    np.save(PROCESSED_DIR / "X_test_flat.npy", X_test_flat)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)

    print("\n=== Array Shapes ===")
    print(f"  X_train_cnn  (CNN 4D):  {X_train_cnn.shape}   dtype={X_train_cnn.dtype}")
    print(f"  X_test_cnn   (CNN 4D):  {X_test_cnn.shape}    dtype={X_test_cnn.dtype}")
    print(f"  X_train_flat (flat 2D): {X_train_flat.shape}  dtype={X_train_flat.dtype}")
    print(f"  X_test_flat  (flat 2D): {X_test_flat.shape}   dtype={X_test_flat.dtype}")
    print(f"  y_train:                {y_train.shape}")
    print(f"  y_test:                 {y_test.shape}")
    print(f"\n  Pixel value range: [{X_train_cnn.min():.3f}, {X_train_cnn.max():.3f}]")

    print("\n=== Train class distribution ===")
    for cls, lbl in CLASS_LABELS.items():
        print(f"  {cls:8s} (label {lbl}): train={np.sum(y_train==lbl):4d}  test={np.sum(y_test==lbl):4d}")

    print("\n=== ASCII preview — one sample per class ===")
    for cls, lbl in CLASS_LABELS.items():
        idx = np.where(y_train == lbl)[0][0]
        print(f"\n--- {cls.upper()} NOTE (train[{idx}]) ---")
        print(ascii_preview(X_train[idx]))


if __name__ == "__main__":
    main()
