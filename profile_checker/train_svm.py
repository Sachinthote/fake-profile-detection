import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# ✅ Suppress TensorFlow logs and disable GPU (Render doesn’t support GPU)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Dataset path and categories
DATASET_PATH = r"D:\Profile Checking\fake_profile_detection\fake_profile_detection\dataset\train"
CATEGORIES = ["real", "fake"]
IMG_SIZE = (224, 224)

# ✅ Load pre-trained VGG16 without top layers (feature extractor)
feature_extractor = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# ✅ Manual VGG16-style preprocessing (matches views.py)
def manual_preprocess(img_array):
    img_array = img_array.astype(np.float32)
    img_array = img_array[..., ::-1]  # RGB to BGR
    mean = [103.939, 116.779, 123.68]
    img_array[..., 0] -= mean[0]
    img_array[..., 1] -= mean[1]
    img_array[..., 2] -= mean[2]
    return img_array

# ✅ Feature extractor
def extract_features(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = manual_preprocess(img_array)
    features = feature_extractor.predict(img_array)
    return features.flatten()

# ✅ Load dataset and extract features
X, y = [], []
for label, category in enumerate(CATEGORIES):
    folder_path = os.path.join(DATASET_PATH, category)
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Dataset folder not found: {folder_path}")
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        try:
            features = extract_features(img_path)
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"⚠️ Skipping {img_path}: {e}")

X = np.array(X)
y = np.array(y)

# ✅ Train SVM classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svm_model = SVC(kernel="linear", probability=True)
svm_model.fit(X_train, y_train)

# ✅ Save models
MODEL_PATH = r"D:/Profile Checking/fake_profile_detection/profile_checker/"
os.makedirs(MODEL_PATH, exist_ok=True)

joblib.dump(svm_model, os.path.join(MODEL_PATH, "svm_image_model.pkl"))
joblib.dump(feature_extractor, os.path.join(MODEL_PATH, "feature_extractor.pkl"))

print("✅ Image-based SVM model trained and saved successfully!")
