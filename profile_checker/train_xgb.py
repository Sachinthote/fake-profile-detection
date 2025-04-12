import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from sklearn.model_selection import train_test_split
import xgboost as xgb  # ✅ XGBoost classifier

# ✅ Disable unnecessary logs and GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ✅ Paths and settings
DATASET_PATH = r"D:\Profile Checking\fake_profile_detection\fake_profile_detection\dataset\train"
CATEGORIES = ["real", "fake"]
IMG_SIZE = (224, 224)

# ✅ Load MobileNetV2 (as feature extractor)
feature_extractor = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3), pooling='avg')

def extract_features(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = mobilenet_preprocess(img_array)
    features = feature_extractor.predict(img_array)
    return features.flatten()

# ✅ Extract features from dataset
X, y = [], []
for label, category in enumerate(CATEGORIES):
    folder_path = os.path.join(DATASET_PATH, category)
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        try:
            features = extract_features(img_path)
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"⚠️ Skipping {filename}: {e}")

X = np.array(X)
y = np.array(y)

# ✅ Train XGBoost
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# ✅ Save model
MODEL_PATH = r"D:/Profile Checking/fake_profile_detection/profile_checker/"
os.makedirs(MODEL_PATH, exist_ok=True)
joblib.dump(xgb_model, os.path.join(MODEL_PATH, "xgb_image_model.pkl"))
joblib.dump(feature_extractor, os.path.join(MODEL_PATH, "feature_extractor.pkl"))

print("✅ XGBoost model with MobileNetV2 features trained and saved successfully!")
