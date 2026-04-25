import numpy as np
import matplotlib.pyplot as plt

# Load the CNN-ready data (4D)
X_train = np.load('data/processed/X_train_cnn.npy')
y_train = np.load('data/processed/y_train.npy')

# Map IDs back to names for the title
class_names = {0: 'Whole-note', 1: 'Half-note', 2: 'Quarter-note'}

# Set up a grid to view 5 random samples
plt.figure(figsize=(12, 4))
indices = np.random.choice(len(X_train), 5, replace=False)

for i, idx in enumerate(indices):
    plt.subplot(1, 5, i + 1)
    # Remove the channel dimension for plotting: (32, 32, 1) -> (32, 32)
    plt.imshow(X_train[idx].reshape(32, 32), cmap='gray')
    plt.title(f"Label: {class_names[y_train[idx]]}")
    plt.axis('off')

plt.tight_layout()
plt.show()
