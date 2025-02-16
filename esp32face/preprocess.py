import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Define dataset paths
dataset_path = "dataset/"  # Ensure this path is correct
mask_path = os.path.join(dataset_path, "with_mask")
no_mask_path = os.path.join(dataset_path, "without_mask")

X = []
Y = []

# Function to load and process images
def load_images_from_folder(folder, label):
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"Warning: Unable to read {img_path}")
            continue  # Skip unreadable images

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        image = cv2.resize(image, (96, 96))  # Resize to 96x96
        X.append(image)
        Y.append(label)

# Load data
load_images_from_folder(mask_path, 1)  # 1 = Mask
load_images_from_folder(no_mask_path, 0)  # 0 = No Mask

# Convert lists to NumPy arrays
X = np.array(X).reshape(-1, 96, 96, 1) / 255.0  # Normalize & reshape
Y = np.array(Y)

# Split dataset into training (80%) and validation (20%) sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Save the preprocessed data
np.save("X_train.npy", X_train)
np.save("X_val.npy", X_val)
np.save("Y_train.npy", Y_train)
np.save("Y_val.npy", Y_val)

print(f"Dataset processed! {len(X_train)} training images, {len(X_val)} validation images.")
