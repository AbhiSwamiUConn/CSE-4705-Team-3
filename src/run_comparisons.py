"""
Master evaluation script: trains/loads all three models on the shared test split,
captures Accuracy, Macro F1, and Inference Time, then writes a CSV and bar-chart.
"""

import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR   = PROJECT_ROOT / "results"
CNN_MODEL_PATH = PROJECT_ROOT / "models" / "cnn" / "music_note_cnn.h5"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "Whole Note", "Half Note", "Quarter Note",
    "G-Clef", "F-Clef",
    "Quarter Rest", "Eighth Rest",
    "Common Time",
]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_flat_data():
    X_train = np.load(PROCESSED_DIR / "X_train_flat.npy")
    X_test  = np.load(PROCESSED_DIR / "X_test_flat.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test  = np.load(PROCESSED_DIR / "y_test.npy")
    return X_train, X_test, y_train, y_test


def load_cnn_data():
    X_train = np.load(PROCESSED_DIR / "X_train_cnn.npy")
    X_test  = np.load(PROCESSED_DIR / "X_test_cnn.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test  = np.load(PROCESSED_DIR / "y_test.npy")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Model runners — each returns (accuracy, macro_f1, inference_seconds)
# ---------------------------------------------------------------------------

def run_logistic_regression(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 50)
    print("  LOGISTIC REGRESSION")
    print("=" * 50)

    scaler      = StandardScaler()
    X_train_s   = scaler.fit_transform(X_train)
    X_test_s    = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42, multi_class="auto")
    print("  Training...")
    model.fit(X_train_s, y_train)

    t0      = time.perf_counter()
    y_pred  = model.predict(X_test_s)
    inf_sec = time.perf_counter() - t0

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro")
    print(f"  Test Accuracy:  {acc:.4f}")
    print(f"  Macro F1-Score: {f1:.4f}")
    print(f"  Inference Time: {inf_sec * 1000:.2f} ms")
    return acc, f1, inf_sec


def run_random_forest(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 50)
    print("  RANDOM FOREST")
    print("=" * 50)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    print("  Training...")
    model.fit(X_train, y_train)

    t0      = time.perf_counter()
    y_pred  = model.predict(X_test)
    inf_sec = time.perf_counter() - t0

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro")
    print(f"  Test Accuracy:  {acc:.4f}")
    print(f"  Macro F1-Score: {f1:.4f}")
    print(f"  Inference Time: {inf_sec * 1000:.2f} ms")
    return acc, f1, inf_sec


def run_cnn(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 50)
    print("  CNN")
    print("=" * 50)

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    num_classes = len(CLASS_NAMES)
    needs_train = True
    if CNN_MODEL_PATH.exists():
        candidate = keras.models.load_model(str(CNN_MODEL_PATH))
        if candidate.output_shape[-1] == num_classes:
            print(f"  Loading saved model from {CNN_MODEL_PATH}")
            model = candidate
            needs_train = False
        else:
            print(f"  Saved model has {candidate.output_shape[-1]} output classes "
                  f"but need {num_classes} — retraining from scratch.")

    if needs_train:
        print("  Training CNN from scratch (20 epochs)...")
        tf.random.set_seed(42)
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(0.4),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ], name="music_note_cnn")
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)
        model.save(str(CNN_MODEL_PATH))
        print(f"  Model saved → {CNN_MODEL_PATH}")

    t0      = time.perf_counter()
    y_pred  = np.argmax(model.predict(X_test, verbose=0), axis=1)
    inf_sec = time.perf_counter() - t0

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro")
    print(f"  Test Accuracy:  {acc:.4f}")
    print(f"  Macro F1-Score: {f1:.4f}")
    print(f"  Inference Time: {inf_sec * 1000:.2f} ms")
    return acc, f1, inf_sec


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def save_csv(results):
    csv_path = RESULTS_DIR / "final_comparison_table.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Model", "Test Accuracy", "Macro F1-Score", "Inference Time (ms)"])
        for name, acc, f1, t in results:
            writer.writerow([name, f"{acc:.4f}", f"{f1:.4f}", f"{t * 1000:.2f}"])
    print(f"\nCSV saved → {csv_path}")


def save_plot(results):
    models = [r[0] for r in results]
    accs   = [r[1] for r in results]
    f1s    = [r[2] for r in results]

    x     = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    bars_acc = ax.bar(x - width / 2, accs, width, label="Test Accuracy",  color="#4C72B0")
    bars_f1  = ax.bar(x + width / 2, f1s,  width, label="Macro F1-Score", color="#DD8452")

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Test Accuracy & Macro F1-Score")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for bar in (*bars_acc, *bars_f1):
        ax.annotate(
            f"{bar.get_height():.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=9,
        )

    out = RESULTS_DIR / "model_comparison_plot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 50)
    print("  MASTER MODEL COMPARISON")
    print("=" * 50)
    print(f"\nShared test split: {PROCESSED_DIR}")
    print("All three models are evaluated on the identical y_test array.\n")

    # Load data — y_test is the same array for all models (same split, random_state=42)
    X_train_flat, X_test_flat, y_train_flat, y_test = load_flat_data()
    X_train_cnn,  X_test_cnn,  y_train_cnn,  _      = load_cnn_data()

    print(f"Test set size: {len(y_test)} samples")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {(y_test == i).sum()} samples")

    results = []

    acc, f1, t = run_logistic_regression(X_train_flat, X_test_flat, y_train_flat, y_test)
    results.append(("Logistic Regression", acc, f1, t))

    acc, f1, t = run_random_forest(X_train_flat, X_test_flat, y_train_flat, y_test)
    results.append(("Random Forest", acc, f1, t))

    acc, f1, t = run_cnn(X_train_cnn, X_test_cnn, y_train_cnn, y_test)
    results.append(("CNN", acc, f1, t))

    save_csv(results)
    save_plot(results)

    # Print markdown-ready summary
    print("\n" + "=" * 65)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 65)
    header = f"{'Model':<22} {'Accuracy':>12} {'Macro F1':>12} {'Infer (ms)':>12}"
    print(header)
    print("-" * 60)
    for name, acc, f1, t in results:
        print(f"{name:<22} {acc:>12.4f} {f1:>12.4f} {t * 1000:>12.2f}")


if __name__ == "__main__":
    main()
