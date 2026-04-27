import sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT/"src"))
import preprocess

PROCESSED_DIR = PROJECT_ROOT/"data"/"processed"
RESULTS_DIR = PROJECT_ROOT/"results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_FILES = ["X_train_flat.npy", "X_test_flat.npy", "y_train.npy", "y_test.npy"]

def ensure_processed_data():
    """Check that all required .npy files exist; run preprocessing if any are missing."""
    missing = [f for f in REQUIRED_FILES if not (PROCESSED_DIR / f).exists()]
    if missing:
        print(f"Missing {len(missing)} processed file(s). Running preprocessing...")
        preprocess.main()
        print("Preprocessing complete.\n")
    else:
        print("All processed data files found.\n")

def load_data():
    """Load flattened feature arrays and labels from the processed data directory."""
    X_train = np.load(PROCESSED_DIR/"X_train_flat.npy")
    X_test  = np.load(PROCESSED_DIR/"X_test_flat.npy")
    y_train = np.load(PROCESSED_DIR/"y_train.npy")
    y_test  = np.load(PROCESSED_DIR/"y_test.npy")

    return X_train, X_test, y_train, y_test

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Random Forest")
    out = RESULTS_DIR / "rf_confusion_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved → {out}")

def main():
    ensure_processed_data()

    print("Loading datasets...")
    X_train, X_test, y_train, y_test = load_data()

    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators = 100, random_state = 42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n" + "="*30)
    print("   RANDOM FOREST RESULTS")
    print("="*30)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    target_names = [name.capitalize() for name in preprocess.CLASS_LABELS.keys()]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    plot_confusion_matrix(y_test, y_pred, target_names)

    # Training accuracy to check for overfitting
    train_acc = accuracy_score(y_train, model.predict(X_train))
    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:     {accuracy_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    main()
