import cv2
import numpy as np
import os
from PIL import Image

# Path to dataset folder
DATASET_PATH = "dataset"  # Change this to your dataset folder path
CATEGORIES = ["with_mask", "without_mask"]

def train_classifier():
    faces = []
    labels = []

    # Iterate over each category (with_mask and without_mask)
    for label, category in enumerate(CATEGORIES):
        category_path = os.path.join(DATASET_PATH, category)

        # Loop through each image in the category folder
        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            imageNp = np.array(img, 'uint8')  # Convert to NumPy array

            faces.append(imageNp)  # Add face data
            labels.append(label)   # Add label (0=with_mask, 1=without_mask)

            cv2.imshow("Training", imageNp)
            cv2.waitKey(1)

    labels = np.array(labels)

    # Create LBPH Face Recognizer and train
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, labels)

    # Save the trained model
    recognizer.write("classifier.xml")

    cv2.destroyAllWindows()
    print("✅ Training Completed! Classifier saved as 'classifier.xml'.")

train_classifier()
