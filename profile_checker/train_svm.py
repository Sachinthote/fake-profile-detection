import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Dataset path
DATASET_PATH = r"D:\Profile Checking\fake_profile_detection\fake_profile_detection\dataset\train"
CATEGORIES = ["real", "fake"]
IMG_SIZE = (224, 224)

# Load pre-trained VGG16 model (without classifier layers)
feature_extractor = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

def extract_features(image_path):
    """Extract deep learning features using VGG16."""
    img = load_img(image_path, target_size=IMG_SIZE)  # Load image
    img_array = img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions
    img_array = preprocess_input(img_array)  # Preprocess for VGG16
    
    features = feature_extractor.predict(img_array)  # Extract features
    return features.flatten()  # Flatten to 1D vector

# Prepare dataset
X, y = [], []
for label, category in enumerate(CATEGORIES):
    folder_path = os.path.join(DATASET_PATH, category)
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Dataset folder not found: {folder_path}")
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        features = extract_features(img_path)
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Train SVM model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svm_model = SVC(kernel="linear", probability=True)
svm_model.fit(X_train, y_train)

# Save the trained models
MODEL_PATH = r"D:/Profile Checking/fake_profile_detection/profile_checker/"
os.makedirs(MODEL_PATH, exist_ok=True)

joblib.dump(svm_model, os.path.join(MODEL_PATH, "svm_image_model.pkl"))
joblib.dump(feature_extractor, os.path.join(MODEL_PATH, "feature_extractor.pkl"))

print("✅ Image-based SVM model trained and saved successfully!")
