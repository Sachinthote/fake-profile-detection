import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
from tensorflow.keras import layers, models, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.applications import EfficientNetV2S, ConvNeXtSmall
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2
from datetime import datetime
import albumentations as A
from tqdm import tqdm
import pandas as pd
import shutil
import cv2
import time
from keras_cv.models import TwoWayTransformer
import timm

print(f"TensorFlow version: {tf.__version__}")

# Configuration - Paths and hyperparameters
TRAIN_PATH = r"D:/Profile Checking/fake_profile_detection/fake_profile_detection/dataset/train"
VALID_PATH = r"D:/Profile Checking/fake_profile_detection/fake_profile_detection/dataset/validate"
OUTPUT_DIR = r"D:/Profile Checking/fake_profile_detection/profile_checker"
MODEL_PATH = os.path.join(OUTPUT_DIR, "profile_detector.h5")
TFLITE_PATH = os.path.join(OUTPUT_DIR, "profile_detector_model.tflite")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
MODEL_TYPE = "hybrid"  # Options: "efficientnet", "convnext", "vit", "hybrid"
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
MIXED_PRECISION = True
USE_CACHE = True
ENSEMBLE = True
SEED = 42

# Set random seeds for reproducibility
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Enable mixed precision if selected
if MIXED_PRECISION:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("🚀 Mixed precision enabled")

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Data analysis and visualization
def analyze_dataset(train_path, valid_path):
    """Analyze and print dataset information"""
    print(f"🔍 Analyzing dataset...")
    train_classes = sorted(os.listdir(train_path))
    val_classes = sorted(os.listdir(valid_path))
    
    train_samples = sum([len(os.listdir(os.path.join(train_path, class_name))) for class_name in train_classes])
    val_samples = sum([len(os.listdir(os.path.join(valid_path, class_name))) for class_name in val_classes])
    
    print(f"Classes: {train_classes}")
    print(f"Training samples: {train_samples}, Validation samples: {val_samples}")
    
    class_distribution = {
        'train': {},
        'validation': {}
    }
    
    for class_name in train_classes:
        count = len(os.listdir(os.path.join(train_path, class_name)))
        percentage = count/train_samples*100
        class_distribution['train'][class_name] = {'count': count, 'percentage': percentage}
        print(f"  - Training '{class_name}': {count} images ({percentage:.1f}%)")

    for class_name in val_classes:
        count = len(os.listdir(os.path.join(valid_path, class_name)))
        percentage = count/val_samples*100
        class_distribution['validation'][class_name] = {'count': count, 'percentage': percentage}
        print(f"  - Validation '{class_name}': {count} images ({percentage:.1f}%)")
    
    # Plot class distribution
    labels = train_classes
    train_counts = [class_distribution['train'][c]['count'] for c in labels]
    val_counts = [class_distribution['validation'][c]['count'] for c in labels]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    width = 0.35
    
    ax.bar(x - width/2, train_counts, width, label='Train')
    ax.bar(x + width/2, val_counts, width, label='Validation')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Number of samples')
    ax.set_title('Dataset Distribution')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "dataset_distribution.png"))
    plt.close()
    
    # Check if the dataset is balanced
    is_balanced = True
    for split in ['train', 'validation']:
        counts = [class_distribution[split][c]['count'] for c in train_classes]
        max_count = max(counts)
        min_count = min(counts)
        
        if max_count / min_count > 1.2:  # 20% tolerance
            is_balanced = False
            break
    
    print(f"Dataset balance check: {'✅ Balanced' if is_balanced else '⚠️ Imbalanced'}")
    
    return class_distribution, train_classes, is_balanced

# Custom data generator with Albumentations for advanced augmentation
class ProfileDataGenerator:
    """Custom data generator with advanced augmentations using Albumentations"""
    def __init__(self, directory, batch_size=32, image_size=(256, 256), 
                 augment=False, shuffle=True, cache_dir=None, class_mode='binary'):
        self.directory = directory
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        self.shuffle = shuffle
        self.class_mode = class_mode
        self.cache_dir = cache_dir
        
        # Get class and file information
        self.classes = sorted(os.listdir(directory))
        self.class_indices = {cls: i for i, cls in enumerate(self.classes)}
        self.files = []
        self.labels = []
        
        # Create augmentation pipeline
        if self.augment:
            self.transform = A.Compose([
                # Spatial transformations
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
                
                # Color transformations
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                ], p=0.5),
                
                # Noise and quality transformations
                A.OneOf([
                    A.GaussNoise(var_limit=(10, 50)),
                    A.GaussianBlur(blur_limit=3),
                    A.ImageCompression(quality_lower=75, quality_upper=100),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5))
                ], p=0.3),
                
                # Distortion and occlusion
                A.OneOf([
                    A.GridDistortion(distort_limit=0.1),
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50),
                    A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1)
                ], p=0.2),
                
                A.CoarseDropout(max_holes=8, max_height=8, max_width=8, min_height=4, min_width=4, p=0.3),
                
                # Normalization (match model expectations)
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        # Load file paths and labels
        for class_name in self.classes:
            class_dir = os.path.join(directory, class_name)
            class_files = os.listdir(class_dir)
            
            for file in class_files:
                self.files.append(os.path.join(class_dir, file))
                self.labels.append(self.class_indices[class_name])
        
        self.indices = np.arange(len(self.files))
        self.n_samples = len(self.files)
        self.n_batches = int(np.ceil(self.n_samples / self.batch_size))
        self.on_epoch_end()
        
        # Cache dataset if directory is provided
        if self.cache_dir is not None:
            self.cache_dataset()
    
    def on_epoch_end(self):
        """Shuffle dataset at the end of each epoch if shuffle is True"""
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def cache_dataset(self):
        """Preprocess and cache all images to disk for faster training"""
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Check if already cached
        if len(os.listdir(self.cache_dir)) >= len(self.files):
            print(f"Dataset already cached in {self.cache_dir}")
            return
        
        print(f"Caching dataset to {self.cache_dir}...")
        for i, idx in enumerate(tqdm(range(len(self.files)))):
            img_path = self.files[idx]
            label = self.labels[idx]
            
            # Read and transform image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.image_size)
            transformed = self.transform(image=img)
            img_processed = transformed['image']
            
            # Save processed image and label
            cache_file = os.path.join(self.cache_dir, f"img_{i:06d}.npz")
            np.savez_compressed(cache_file, image=img_processed, label=label)
    
    def __len__(self):
        """Return the number of batches per epoch"""
        return self.n_batches
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indices for this batch
        batch_indices = self.indices[index * self.batch_size:min((index + 1) * self.batch_size, self.n_samples)]
        
        # Initialize batch arrays
        batch_x = np.empty((len(batch_indices), *self.image_size, 3), dtype=np.float32)
        batch_y = np.empty(len(batch_indices), dtype=np.float32)
        
        # Fill batch arrays
        for i, idx in enumerate(batch_indices):
            if self.cache_dir is not None and USE_CACHE:
                # Load from cache
                cache_file = os.path.join(self.cache_dir, f"img_{idx:06d}.npz")
                if os.path.exists(cache_file):
                    data = np.load(cache_file)
                    batch_x[i] = data['image']
                    batch_y[i] = data['label']
                else:
                    # Fallback to loading from disk
                    img_path = self.files[idx]
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.image_size)
                    transformed = self.transform(image=img)
                    batch_x[i] = transformed['image']
                    batch_y[i] = self.labels[idx]
            else:
                # Load from disk and apply transformations
                img_path = self.files[idx]
                img = cv2.imread(img_path)
                if img is None:
                    # Use an empty black image as fallback
                    img = np.zeros((*self.image_size, 3), dtype=np.uint8)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.image_size)
                
                transformed = self.transform(image=img)
                batch_x[i] = transformed['image']
                batch_y[i] = self.labels[idx]
        
        return batch_x, batch_y

# Create TensorFlow datasets from generators
def create_tf_datasets(train_gen, val_gen):
    """Convert custom generators to TensorFlow datasets"""
    def gen_train():
        for i in range(len(train_gen)):
            yield train_gen[i]
    
    def gen_val():
        for i in range(len(val_gen)):
            yield val_gen[i]
    
    output_signature = (
        tf.TensorSpec(shape=(None, *IMG_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
    
    train_ds = tf.data.Dataset.from_generator(
        gen_train,
        output_signature=output_signature
    )
    
    val_ds = tf.data.Dataset.from_generator(
        gen_val,
        output_signature=output_signature
    )
    
    # Optimize dataset performance
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds

# Build EfficientNetV2 model
def build_efficientnet_model(img_size=(256, 256)):
    """Create EfficientNetV2S model for binary classification"""
    # Input layer
    inputs = Input(shape=(*img_size, 3))
    
    # Base model with pretrained weights
    base_model = EfficientNetV2S(
        weights='imagenet',
        include_top=False,
        input_tensor=inputs
    )
    
    # Global pooling to reduce dimensions
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    
    # Feature normalization
    x = layers.BatchNormalization()(x)
    
    # First dense block with regularization
    x = layers.Dense(512, kernel_regularizer=regularizers.l2(0.0005))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(0.5)(x)
    
    # Second dense block
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.0005))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(0.4)(x)
    
    # Attention mechanism
    attention = layers.Dense(256, activation='tanh')(x)
    attention = layers.Dense(256, activation='sigmoid')(attention)
    x = layers.Multiply()([x, attention])
    
    # Output layer with sigmoid for binary classification
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    # Create the full model
    model = Model(inputs=inputs, outputs=outputs, name='efficientnet_classifier')
    
    return model

# Build ConvNeXt model
def build_convnext_model(img_size=(256, 256)):
    """Create ConvNeXt model for binary classification"""
    # Input layer
    inputs = Input(shape=(*img_size, 3))
    
    # Base model with pretrained weights
    base_model = ConvNeXtSmall(
        weights='imagenet',
        include_top=False,
        input_tensor=inputs
    )
    
    # Global pooling to reduce dimensions
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    
    # Feature normalization
    x = layers.BatchNormalization()(x)
    
    # Feature extraction block
    x = layers.Dense(512)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation('gelu')(x)
    x = layers.Dropout(0.4)(x)
    
    # Feature reduction block
    x = layers.Dense(256)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation('gelu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer with sigmoid for binary classification
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    # Create the full model
    model = Model(inputs=inputs, outputs=outputs, name='convnext_classifier')
    
    return model

# Build Vision Transformer model
def build_vit_model(img_size=(256, 256)):
    """Create Vision Transformer model for binary classification using timm"""
    # Input layer
    inputs = Input(shape=(*img_size, 3))
    
    # Use timm to create a pretrained ViT model
    # First convert the input to the format expected by timm (B, C, H, W)
    x = layers.Permute((3, 1, 2))(inputs)  # Convert from (B, H, W, C) to (B, C, H, W)
    
    # Create a wrapper for the timm model
    vit = timm.create_model('vit_base_patch16_224', pretrained=True)
    
    # Create a functional API wrapper for the timm model
    # This is just a placeholder for how you might incorporate timm
    # In a real implementation, you'd need to properly integrate the models
    
    # Since we can't directly use timm in TF Keras functional API,
    # we can use a Lambda layer to wrap it or use a custom TF implementation
    # Here we'll use a placeholder implementation with standard TF layers
    
    # Patch embedding
    x = layers.Conv2D(768, kernel_size=16, strides=16, padding="valid")(inputs)
    # Reshape patches to sequence
    batch_size = tf.shape(x)[0]
    x = layers.Reshape((-1, 768))(x)
    # Add position embedding
    x = layers.Dense(768)(x)
    
    # Transformer blocks
    for _ in range(12):  # 12 transformer blocks
        # Attention block
        skip = x
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.MultiHeadAttention(num_heads=12, key_dim=64)(x, x)
        x = layers.Add()([x, skip])
        
        # MLP block
        skip = x
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dense(3072, activation='gelu')(x)
        x = layers.Dense(768)(x)
        x = layers.Add()([x, skip])
    
    # Classification head
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(512, activation='gelu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    # Create the full model
    model = Model(inputs=inputs, outputs=outputs, name='vit_classifier')
    
    return model

# Build hybrid ensemble model
def build_hybrid_model(img_size=(256, 256)):
    """Create a hybrid model combining CNN and attention features for binary classification"""
    # Input layer
    inputs = Input(shape=(*img_size, 3))
    
    # EfficientNetV2S path
    base_efficient = EfficientNetV2S(
        weights='imagenet',
        include_top=False,
        input_tensor=inputs
    )
    
    # ConvNext path
    base_convnext = ConvNeXtSmall(
        weights='imagenet',
        include_top=False,
        input_tensor=inputs
    )
    
    # Get features from each model
    features_efficient = base_efficient.output
    features_efficient = layers.GlobalAveragePooling2D(name='gap_efficient')(features_efficient)
    features_efficient = layers.BatchNormalization(name='bn_efficient')(features_efficient)
    
    features_convnext = base_convnext.output
    features_convnext = layers.GlobalAveragePooling2D(name='gap_convnext')(features_convnext)
    features_convnext = layers.BatchNormalization(name='bn_convnext')(features_convnext)
    
    # Combine features
    combined_features = layers.Concatenate(name='combined_features')([features_efficient, features_convnext])
    
    # Attention mechanism
    attention = layers.Dense(512, activation='tanh', name='attention_1')(combined_features)
    attention = layers.Dense(512, activation='sigmoid', name='attention_2')(attention)
    attentional_features = layers.Multiply(name='attention_mul')([combined_features, attention])
    
    # Feature extraction
    x = layers.Dense(512, kernel_regularizer=l2(0.0003), name='dense_1')(attentional_features)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Activation('swish', name='swish_1')(x)
    x = layers.Dropout(0.4, name='dropout_1')(x)
    
    x = layers.Dense(256, kernel_regularizer=l2(0.0003), name='dense_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Activation('swish', name='swish_2')(x)
    x = layers.Dropout(0.3, name='dropout_2')(x)
    
    # Output layer with sigmoid for binary classification
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    # Create the full model
    model = Model(inputs=inputs, outputs=outputs, name='hybrid_classifier')
    
    return model

# Build selected model based on configuration
def build_model(model_type='efficientnet'):
    """Build and compile selected model architecture"""
    print(f"🏗️ Building {model_type.upper()} model...")
    
    if (model_type == 'efficientnet'):
        model = build_efficientnet_model(IMG_SIZE)
    elif (model_type == 'convnext'):
        model = build_convnext_model(IMG_SIZE)
    elif (model_type == 'vit'):
        model = build_vit_model(IMG_SIZE)
    elif (model_type == 'hybrid'):
        model = build_hybrid_model(IMG_SIZE)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Compile the model with AdamW optimizer
    optimizer = AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Use binary focal loss for better handling of hard examples
    loss = tf.keras.losses.BinaryFocalCrossentropy(
        alpha=0.25,
        gamma=2.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.F1Score(name='f1')
        ]
    )
    
    return model

# Setup callbacks for training
def create_callbacks():
    """Create callbacks for model training"""
    return [
        # Stop training when validation loss doesn't improve
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        # Save best model
        ModelCheckpoint(
            MODEL_PATH,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        # Reduce learning rate when training plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        # TensorBoard logs for monitoring
        TensorBoard(log_dir=LOG_DIR)
    ]

# Training function
def train_model(model, train_ds, val_ds, epochs, callbacks, steps_per_epoch, validation_steps):
    """Train the model and return history"""
    print(f"🏋️ Starting model training for {epochs} epochs...")
    
    start_time = time.time()
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    return history

# Plot training history
def plot_training_history(history, title="Training History"):
    """Plot training and validation metrics"""
    metrics = ['accuracy', 'loss', 'auc', 'precision', 'recall', 'f1']
    plt.figure(figsize=(20, 10))
    
    for i, metric in enumerate(metrics):
        if metric in history.history:
            plt.subplot(2, 3, i+1)
            plt.plot(history.history[metric])
            plt.plot(history.history[f'val_{metric}'])
            plt.title(f'Model {metric.capitalize()}')
            plt.ylabel(metric.capitalize())
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='best')
            plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{title.replace(' ', '_').lower()}.png"))
    plt.close()

# Evaluate model performance
def evaluate_model(model, val_ds, val_gen, class_names):
    """Evaluate model and display metrics and visualizations"""
    print("📊 Evaluating model performance...")
    
    # Evaluate with metrics
    results = model.evaluate(val_ds, verbose=1)
    metrics = ['Loss', 'Accuracy', 'Precision', 'Recall', 'AUC', 'F1']
    
    print("\n===== Evaluation Results =====")
    for metric, value in zip(metrics, results):
        print(f"{metric}: {value:.4f}")
    
    # Get predictions for all validation samples
    all_labels = []
    all_preds = []
    
    for batch_x, batch_y in val_ds:
        batch_preds = model.predict(batch_x, verbose=0)
        all_labels.append(batch_y.numpy())
        all_preds.append(batch_preds)
    
    y_true = np.concatenate(all_labels)
    y_pred_prob = np.concatenate(all_preds).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=14)
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=12)
    plt.yticks(tick_marks, class_names, fontsize=12)
    
    # Add numbers to confusion matrix
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)
    
    plt.tight_layout()
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Save classification report to CSV
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(OUTPUT_DIR, "classification_report.csv"))
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver operating characteristic', fontsize=14)