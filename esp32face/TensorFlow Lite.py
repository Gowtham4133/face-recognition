import tensorflow as tf

# Convert the trained model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(tf.keras.models.load_model("face_mask_model.h5"))
tflite_model = converter.convert()

# Save the TFLite model
with open("face_mask_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model converted to TFLite!")
