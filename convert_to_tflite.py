import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("D:/ai safety monitor/model/helmet_detector.h5")

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
tflite_model_path = "D:/ai safety monitor/model/helmet_detector.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"✅ TFLite model saved at: {tflite_model_path}")
