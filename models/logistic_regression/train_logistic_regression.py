import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))
import preprocess

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REQUIRED_FILES = ["X_train_flat.npy", "X_test_flat.npy", "y_train.npy", "y_test.npy"]

def ensure_data_exists():
    """Checks for processed files and triggers preprocess if missing anything"""
    missing = [f for f in REQUIRED_FILES if not (PROCESSED_DIR / f).exists()]
    if missing:
        print(f"Data missing: {missing}")
        print("Executing preprocess.main()...")
        preprocess.main()
        print("Preprocessing complete.\n")
    else:
        print("Processed data found")

def load_flat_data():
    """Loads flattened arrays"""
    X_train = np.load(PROCESSED_DIR / 'X_train_flat.npy')
    X_test = np.load(PROCESSED_DIR / 'X_test_flat.npy')
    y_train = np.load(PROCESSED_DIR / 'y_train.npy')
    y_test = np.load(PROCESSED_DIR / 'y_test.npy')
    return X_train, X_test, y_train, y_test

def main():
    #Data Check
    ensure_data_exists()
    #Load Flattened Data
    print("Loading datasets...")
    X_train, X_test, y_train, y_test = load_flat_data()
    #Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)    
    #Training
    #max_iter=1000 ensures convergence for the 1024 input features
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42, multi_class='auto')
    model.fit(X_train_scaled, y_train)
    #Eval
    y_pred = model.predict(X_test_scaled)
    print("\n" + "="*30)
    print("   LOGISTIC REGRESSION RESULTS")
    print("="*30)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    #Map labels dynamically from preprocess.py configuration
    target_names = [name.capitalize() for name in preprocess.CLASS_LABELS.keys()]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
