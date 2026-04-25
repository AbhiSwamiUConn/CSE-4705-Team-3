import numpy as np

# Import this into top of every model

def load_team_data():
    # Load the shared processed data
    X_train = np.load('data/processed/X_train.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')

    # flatten data
    # (samples, 32, 32, 1) -> (samples, 1024)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    print(f"Data Loaded: {X_train_flat.shape[0]} training samples, {X_test_flat.shape[0]} test samples.")
    return X_train_flat, X_test_flat, y_train, y_test