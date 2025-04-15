import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.regularizers import l2
from datetime import datetime

# Configuration - Paths and hyperparameters
TRAIN_PATH = r"D:/Profile Checking/fake_profile_detection/fake_profile_detection/dataset/train"
VALID_PATH = r"D:/Profile Checking/fake_profile_detection/fake_profile_detection/dataset/validate"
OUTPUT_DIR = r"D:/Profile Checking/fake_profile_detection/profile_checker"
MODEL_PATH = os.path.join(OUTPUT_DIR, "efficient_profile_detector.h5")
TFLITE_PATH = os.path.join(OUTPUT_DIR, "profile_detector_model.tflite")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
IMG_SIZE = (224, 224)
BATCH_SIZE = 24
INITIAL_EPOCHS = 25
FINE_TUNE_EPOCHS = 15
INITIAL_LR = 0.0005
FINE_TUNE_LR = 0.0001

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Print dataset info
print(f"🔍 Analyzing dataset...")
train_classes = sorted(os.listdir(TRAIN_PATH))
val_classes = sorted(os.listdir(VALID_PATH))
print(f"Training classes: {train_classes}")
print(f"Validation classes: {val_classes}")

train_samples = sum([len(os.listdir(os.path.join(TRAIN_PATH, class_name))) for class_name in train_classes])
val_samples = sum([len(os.listdir(os.path.join(VALID_PATH, class_name))) for class_name in val_classes])
print(f"Training samples: {train_samples}")
print(f"Validation samples: {val_samples}")

for class_name in train_classes:
    count = len(os.listdir(os.path.join(TRAIN_PATH, class_name)))
    print(f"  - Training '{class_name}': {count} images ({count/train_samples*100:.1f}%)")

for class_name in val_classes:
    count = len(os.listdir(os.path.join(VALID_PATH, class_name)))
    print(f"  - Validation '{class_name}': {count} images ({count/val_samples*100:.1f}%)")

# Data preparation with augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.3,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    brightness_range=[0.8, 1.2],
    channel_shift_range=0.1
)

# Only rescale for validation - no augmentation
val_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
print(f"🔄 Creating data generators...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VALID_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False  # No need to shuffle validation data
)

# Compute class weights for imbalanced data
labels = np.array(train_generator.classes)
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
print(f"Class weights: {class_weight_dict}")

# Model building function
def build_model():
    """Create and compile the EfficientNetB0 model for binary classification"""
    # Base model with pretrained ImageNet weights
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    base_model.trainable = False  # Freeze base model initially
    
    # Build model architecture
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        
        # First dense block
        layers.Dense(512, kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),
        layers.Dropout(0.5),
        
        # Second dense block
        layers.Dense(256, kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),
        layers.Dropout(0.4),
        
        # Third dense block
        layers.Dense(128, kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LR),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model

# Create and display model summary
model = build_model()
model.summary()

# Setup callbacks for training
callbacks = [
    # Stop training when validation loss doesn't improve
    EarlyStopping(
        monitor='val_loss',
        patience=8,
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
        min_lr=1e-6,
        verbose=1
    ),
    # TensorBoard logs for monitoring
    TensorBoard(log_dir=LOG_DIR)
]

# Training function
def train_model(model, train_gen, val_gen, epochs, callbacks, class_weights):
    """Train the model and return history"""
    print(f"🏋️ Starting model training for {epochs} epochs...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    return history

# Initial training phase
initial_history = train_model(
    model, 
    train_generator, 
    val_generator, 
    INITIAL_EPOCHS, 
    callbacks, 
    class_weight_dict
)

# Fine-tuning phase
print("🔧 Fine-tuning the model by unfreezing EfficientNetB0 layers...")
base_model = model.layers[0]
base_model.trainable = True

# Freeze all layers except the last 20
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=FINE_TUNE_LR),
    loss='binary_crossentropy',
    metrics=['accuracy', 
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall'),
             tf.keras.metrics.AUC(name='auc')]
)

# Continue training with fine-tuning
fine_tune_history = train_model(
    model, 
    train_generator, 
    val_generator, 
    FINE_TUNE_EPOCHS, 
    callbacks, 
    class_weight_dict
)

# Visualization functions
def plot_training_history(history, title="Training History"):
    """Plot training and validation metrics"""
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot AUC
    plt.subplot(1, 3, 3)
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title('Model AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{title.replace(' ', '_').lower()}.png"))
    plt.close()

def evaluate_model(model, val_generator):
    """Evaluate model and display metrics and visualizations"""
    print("📊 Evaluating model performance...")
    
    # Reset generator
    val_generator.reset()
    
    # Get predictions
    y_pred_prob = model.predict(val_generator)
    y_pred = (y_pred_prob > 0.5).astype(int)
    y_true = val_generator.classes
    
    # Calculate metrics
    results = model.evaluate(val_generator, verbose=0)
    print(f"Loss: {results[0]:.4f}")
    print(f"Accuracy: {results[1]:.4f}")
    print(f"Precision: {results[2]:.4f}")
    print(f"Recall: {results[3]:.4f}")
    print(f"AUC: {results[4]:.4f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    class_names = list(val_generator.class_indices.keys())
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add numbers to confusion matrix
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"))
    plt.close()
    
    return results

# Plot training history
plot_training_history(initial_history, "Initial Training")
plot_training_history(fine_tune_history, "Fine Tuning")

# Evaluate model
evaluation_results = evaluate_model(model, val_generator)

# Convert to TFLite for mobile deployment
print("🚀 Converting model to TFLite format...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Enable optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)
print(f"✅ TFLite model saved at: {TFLITE_PATH}")

# Save final model in HDF5 format
model.save(MODEL_PATH)
print(f"✅ Final model saved at: {MODEL_PATH}")

# Optional: Save in newer Keras format
keras_path = os.path.join(OUTPUT_DIR, "efficient_profile_detector.keras")
model.save(keras_path)
print(f"✅ Keras model saved at: {keras_path}")

print("\n🎉 Training and evaluation complete!")