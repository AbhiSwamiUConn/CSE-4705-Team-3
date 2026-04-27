from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless — saves to disk instead of opening a window
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR   = PROJECT_ROOT / "results"
MODEL_OUT      = Path(__file__).resolve().parent / "music_note_cnn.h5"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "Whole Note", "Half Note", "Quarter Note",
    "G-Clef", "F-Clef",
    "Quarter Rest", "Eighth Rest",
    "Common Time",
]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SHAPE   = (32, 32, 1)
EPOCHS      = 20
BATCH_SIZE  = 32
RANDOM_SEED = 42

# GPU setup

def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU detected: {[g.name for g in gpus]}")
        print("Memory growth enabled — GPU will be used.")
    else:
        print("No GPU detected — training on CPU.")

# Data

def load_data():
    X_train = np.load(PROCESSED_DIR / "X_train_cnn.npy")
    X_test  = np.load(PROCESSED_DIR / "X_test_cnn.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test  = np.load(PROCESSED_DIR / "y_test.npy")
    print(f"\nData loaded:  X_train={X_train.shape}  X_test={X_test.shape}")
    print(f"              y_train={y_train.shape}  y_test={y_test.shape}")
    return X_train, X_test, y_train, y_test

# Model

def build_model(input_shape, num_classes):
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ], name="music_note_cnn")
    return model

# Plots

def plot_accuracy_curves(history):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.history["accuracy"],     label="Train accuracy")
    ax.plot(history.history["val_accuracy"], label="Val accuracy", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training vs. Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = RESULTS_DIR / "cnn_accuracy_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Accuracy curve saved → {out}")


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — CNN")
    out = RESULTS_DIR / "cnn_confusion_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved → {out}")
    return cm

# Main

def main():
    configure_gpu()
    tf.random.set_seed(RANDOM_SEED)

    X_train, X_test, y_train, y_test = load_data()

    model = build_model(IMG_SHAPE, NUM_CLASSES)
    model.summary()

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"\nTraining for {EPOCHS} epochs, batch size {BATCH_SIZE}...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1,
    )

    # Eval

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest loss:     {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}  ({test_acc*100:.2f}%)")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    f1_per_class = f1_score(y_test, y_pred, average=None)
    f1_macro     = f1_score(y_test, y_pred, average="macro")
    f1_weighted  = f1_score(y_test, y_pred, average="weighted")

    print("\n=== F1 Scores ===")
    for cls, score in zip(CLASS_NAMES, f1_per_class):
        print(f"  {cls:8s}: {score:.4f}")
    print(f"  {'Macro':8s}: {f1_macro:.4f}")
    print(f"  {'Weighted':8s}: {f1_weighted:.4f}")

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    cm = plot_confusion_matrix(y_test, y_pred)
    print("\n=== Confusion Matrix ===")
    print(f"{'':10s}", "  ".join(f"{n:>8}" for n in CLASS_NAMES))
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:10s}", "  ".join(f"{v:>8d}" for v in row))

    plot_accuracy_curves(history)

    # save model
    
    model.save(str(MODEL_OUT))
    print(f"\nModel saved → {MODEL_OUT}")


if __name__ == "__main__":
    main()
