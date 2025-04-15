
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('efficient_profile_detector.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open("profile_detector_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model saved!")
