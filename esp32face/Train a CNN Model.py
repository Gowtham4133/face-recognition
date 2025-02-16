import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load the preprocessed dataset
X_train = np.load("X_train.npy")
X_val = np.load("X_val.npy")
Y_train = np.load("Y_train.npy")
Y_val = np.load("Y_val.npy")

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification output
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, epochs=10, validation_data=(X_val, Y_val), batch_size=32)

# Save the trained model
model.save("face_mask_model.h5")

print("Model trained and saved successfully!")
