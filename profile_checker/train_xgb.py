import os
import numpy as np
import joblib
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving graphs
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
import xgboost as xgb
from tqdm import tqdm
import time
import pandas as pd

# Suppress TensorFlow logs and disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow messages
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

# Configure paths and settings
DATASET_PATH = r"D:\Profile Checking\fake_profile_detection\fake_profile_detection\dataset\train"
VALID_PATH = r"D:/Profile Checking/fake_profile_detection/fake_profile_detection/dataset/validate"
MODEL_PATH = r"D:/Profile Checking/fake_profile_detection/profile_checker/"
CATEGORIES = ["fake", "real"]  # Ensure the order matches the label assignment
IMG_SIZE = (224, 224)
BATCH_SIZE = 32  # Process images in batches

# Create model directory and figures directory
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(os.path.join(MODEL_PATH, "figures"), exist_ok=True)

# Load MobileNetV2 as a feature extractor
print("Loading MobileNetV2 feature extractor...")
feature_extractor = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling='avg'
)
feature_extractor.trainable = False  # Freeze the model

# Function to process image batches
def process_image_batch(image_paths):
    """Process multiple images at once for better performance."""
    batch_images = []
    valid_paths = []

    for path in image_paths:
        try:
            img = load_img(path, target_size=IMG_SIZE)
            img_array = img_to_array(img)
            batch_images.append(img_array)
            valid_paths.append(path)
        except Exception as e:
            print(f"⚠️ Error loading {os.path.basename(path)}: {e}")

    if len(batch_images) == 0:  # Check if list is empty
        return np.array([]), []  # Return empty array and list

    batch_array = np.array(batch_images)
    batch_array = preprocess_input(batch_array)

    try:
        features = feature_extractor.predict(batch_array, verbose=0)
        return features, valid_paths
    except Exception as e:
        print(f"⚠️ Error in feature extraction: {e}")
        return np.array([]), []

# Function to extract features from the dataset
def extract_dataset_features(dataset_path):
    X, y, image_filenames = [], [], []

    start_time = time.time()
    for label, category in enumerate(CATEGORIES):
        folder_path = os.path.join(dataset_path, category)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Get all image files
        image_paths = [
            os.path.join(folder_path, filename)
            for filename in os.listdir(folder_path)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        print(f"Processing {len(image_paths)} {category} images...")

        # Process in batches
        for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
            batch_paths = image_paths[i:i + BATCH_SIZE]
            batch_features, valid_paths = process_image_batch(batch_paths)

            if len(batch_features) > 0:  # Check if features were extracted
                X.extend(batch_features)
                y.extend([label] * len(batch_features))
                image_filenames.extend([os.path.basename(p) for p in valid_paths])

    processing_time = time.time() - start_time
    print(f"Feature extraction completed in {processing_time:.2f} seconds")
    print(f"Extracted features from {len(X)} images")

    if len(X) == 0:
        raise ValueError("No features could be extracted from images. Please check your dataset path and image files.")

    return np.array(X), np.array(y), image_filenames

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CATEGORIES, yticklabels=CATEGORIES)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(title)
        plt.tight_layout()

        save_path = os.path.join(MODEL_PATH, "figures", f"{title.lower().replace(' ', '_')}.png")
        print(f"Saving confusion matrix to: {save_path}")
        plt.savefig(save_path, dpi=300)
        plt.close()
    except Exception as e:
        print(f"⚠️ Error plotting confusion matrix: {e}")

# Main execution
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("FAKE PROFILE DETECTION MODEL TRAINING")
    print("=" * 50 + "\n")

    try:
        # Step 1: Extract features from training dataset
        print("\n[STEP 1] Extracting features from training dataset...")
        X_train_full, y_train_full, train_filenames = extract_dataset_features(DATASET_PATH)

        # Step 2: Split data into train/test sets
        print("\n[STEP 2] Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test, train_files, test_files = train_test_split(
            X_train_full, y_train_full, train_filenames, test_size=0.2, random_state=42
        )

        # Step 3: Train XGBoost model
        print("\n[STEP 3] Training XGBoost model...")
        xgb_model = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric=['logloss', 'error'],
            tree_method='hist',
            n_jobs=-1,
            learning_rate=0.1,
            max_depth=6,
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),  # Handle imbalance
            min_child_weight=1
        )

        # Fit the model
        xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)

        # Step 4: Evaluate the model
        print("\n[STEP 4] Evaluating model performance...")
        y_pred = xgb_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")

        # Debug predictions
        print(f"Sample Predictions: {y_pred[:10]}")
        print(f"Mapped Predictions: {[CATEGORIES[p] for p in y_pred[:10]]}")

        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred, "Confusion Matrix")

        # Step 5: Save the model
        print("\n[STEP 5] Saving the model...")
        joblib.dump(xgb_model, os.path.join(MODEL_PATH, "xgb_image_model.pkl"))
        print("Model saved successfully!")

    except Exception as e:
        print(f"\n❌ Error during training process: {e}")
        import traceback
        traceback.print_exc()