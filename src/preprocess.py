import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ------------------------------
# Configuration
# ------------------------------
DATASET_PATH = r"C:\Users\mohamed aziz\PycharmProjects\eye-detection1\data\raw"
IMG_SIZE = 96
CLASSES = ["normal", "sleep", "yawn"]

# ------------------------------
# Load Dataset
# ------------------------------
def load_dataset(test_size=0.2, random_state=42):
    images = []
    labels = []

    for idx, category in enumerate(CLASSES):
        folder = os.path.join(DATASET_PATH, category)
        if not os.path.exists(folder):
            print(f"[ERROR] Folder does not exist: {folder}")
            continue

        # Only load image files
        files = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if len(files) == 0:
            print(f"[WARNING] No images found in {folder}")
            continue

        print(f"[INFO] Loading {category} images ({len(files)} files)...")

        for file in files:
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARNING] Cannot read image: {img_path}")
                continue

            # Convert BGR → RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Resize
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            # Normalize
            img = img / 255.0

            images.append(img)
            labels.append(idx)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    print(f"[INFO] Total images loaded: {len(images)}")

    # One-hot encode labels
    labels = to_categorical(labels, num_classes=len(CLASSES))

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=test_size, random_state=random_state, shuffle=True
    )

    print(f"[INFO] Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    return X_train, X_val, y_train, y_val

# ------------------------------
# Optional: Visualize samples
# ------------------------------
def visualize_samples(X, y, num_samples=5):
    plt.figure(figsize=(12, 4))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(X[i])
        plt.title(f"Class: {CLASSES[np.argmax(y[i])]}")
        plt.axis('off')
    plt.show()

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    X_train, X_val, y_train, y_val = load_dataset()
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

    # Optional: visualize first few training images
    visualize_samples(X_train, y_train, num_samples=5)
