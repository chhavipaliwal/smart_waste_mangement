"""
Smart Waste Segregation System - Conference Ready
Classifies waste into Organic (O) and Recyclable (R) categories
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import kagglehub
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import random

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 80)
print("SMART WASTE SEGREGATION SYSTEM")
print("Organic vs Recyclable Classification")
print("=" * 80)

# ============================================================================
# 1. DATASET SETUP
# ============================================================================
print("\n[1/7] Loading Dataset...")

# Download dataset using kagglehub
path = kagglehub.dataset_download("techsash/waste-classification-data")
print(f"✅ Path to dataset files: {path}")

# Navigate to correct directory structure
dataset_path = Path(path)
train_dir = dataset_path / "DATASET" / "TRAIN"
test_dir = dataset_path / "DATASET" / "TEST"

# Verify paths exist
if not train_dir.exists():
    # Try alternative path
    train_dir = dataset_path / "train"
    test_dir = dataset_path / "TEST"

print(f"📁 Training directory: {train_dir}")
print(f"📁 Testing directory: {test_dir}")

# Count images in each category
def count_images(directory):
    counts = {}
    for category in ['O', 'R']:
        cat_path = directory / category
        if cat_path.exists():
            counts[category] = len(list(cat_path.glob('*.jpg'))) + len(list(cat_path.glob('*.png')))
    return counts

train_counts = count_images(train_dir)
test_counts = count_images(test_dir)

print(f"\n📊 Training Set:")
print(f"   Organic (O): {train_counts.get('O', 0)} images")
print(f"   Recyclable (R): {train_counts.get('R', 0)} images")
print(f"   Total: {sum(train_counts.values())} images")

print(f"\n📊 Test Set:")
print(f"   Organic (O): {test_counts.get('O', 0)} images")
print(f"   Recyclable (R): {test_counts.get('R', 0)} images")
print(f"   Total: {sum(test_counts.values())} images")

# ============================================================================
# 2. DATA VISUALIZATION - DATASET DISTRIBUTION
# ============================================================================
print("\n[2/7] Visualizing Dataset Distribution...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Training Set Distribution
categories = ['Organic', 'Recyclable']
train_values = [train_counts.get('O', 0), train_counts.get('R', 0)]
colors = ['#2ecc71', '#3498db']

axes[0].pie(train_values, labels=categories, autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
axes[0].set_title('Training Set Distribution', fontsize=14, weight='bold')

# Plot 2: Test Set Distribution
test_values = [test_counts.get('O', 0), test_counts.get('R', 0)]
axes[1].pie(test_values, labels=categories, autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
axes[1].set_title('Test Set Distribution', fontsize=14, weight='bold')
# Plot 3: Overall Distribution Bar Chart
x = np.arange(len(categories))
width = 0.35

bars1 = axes[2].bar(x - width/2, train_values, width, label='Train', color='#e74c3c', alpha=0.8)
bars2 = axes[2].bar(x + width/2, test_values, width, label='Test', color='#9b59b6', alpha=0.8)

axes[2].set_xlabel('Category', fontsize=12, weight='bold')
axes[2].set_ylabel('Number of Images', fontsize=12, weight='bold')
axes[2].set_title('Train vs Test Distribution', fontsize=14, weight='bold')
axes[2].set_xticks(x)
axes[2].set_xticklabels(categories)
axes[2].legend()
axes[2].grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('dataset_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Dataset visualization saved as 'dataset_distribution.png'")

# ============================================================================
# 3. SAMPLE IMAGES VISUALIZATION
# ============================================================================
print("\n[3/7] Displaying Sample Images...")

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('Sample Images from Dataset', fontsize=16, weight='bold', y=1.02)

for idx, category in enumerate(['O', 'R']):
    cat_path = train_dir / category
    images = list(cat_path.glob('*.jpg'))[:5]

    category_name = 'Organic' if category == 'O' else 'Recyclable'

    for i, img_path in enumerate(images):
        img = Image.open(img_path)
        axes[idx, i].imshow(img)
        axes[idx, i].axis('off')
        axes[idx, i].set_title(f'{category_name}\n{img_path.name}', fontsize=10)

plt.tight_layout()
plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Sample images visualization saved as 'sample_images.png'")

# ============================================================================
# 4. DATA PREPROCESSING & AUGMENTATION
# ============================================================================
print("\n[4/7] Setting up Data Preprocessing...")

IMG_SIZE = 224
BATCH_SIZE = 32

# Data augmentation for training (improves model generalization)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20% for validation
)

# Only rescaling for test data
test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"\n✅ Data generators created:")
print(f"   Training samples: {train_generator.samples}")
print(f"   Validation samples: {validation_generator.samples}")
print(f"   Test samples: {test_generator.samples}")
print(f"   Class indices: {train_generator.class_indices}")

# ============================================================================
# 5. MODEL ARCHITECTURE (Transfer Learning with MobileNetV2)
# ============================================================================
print("\n[5/7] Building Model Architecture...")

# Load pre-trained MobileNetV2 (efficient for real-time applications)
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model layers
base_model.trainable = False

# Build model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

print("\n📋 Model Summary:")
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-7)
]

# ============================================================================
# 6. MODEL TRAINING
# ============================================================================
print("\n[6/7] Training Model...")

EPOCHS = 10

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Model training completed!")

# Save model
model.save('waste_classification_model.h5')
print("✅ Model saved as 'waste_classification_model.h5'")

# ============================================================================
# 7. MODEL EVALUATION & PERFORMANCE METRICS
# ============================================================================
print("\n[7/7] Evaluating Model Performance...")

# Evaluate on test set
test_loss, test_acc, test_precision, test_recall = model.evaluate(test_generator, verbose=0)
test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)

print(f"\n📊 Test Set Performance:")
print(f"   Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"   Precision: {test_precision:.4f}")
print(f"   Recall:    {test_recall:.4f}")
print(f"   F1-Score:  {test_f1:.4f}")
print(f"   Loss:      {test_loss:.4f}")

# Get predictions
test_generator.reset()
y_pred_prob = model.predict(test_generator, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()
y_true = test_generator.classes

# Classification Report
print("\n📋 Detailed Classification Report:")
class_names = ['Organic', 'Recyclable']
print(classification_report(y_true, y_pred, target_names=class_names))

# ============================================================================
# VISUALIZATION: TRAINING HISTORY
# ============================================================================
print("\n📈 Generating Performance Visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2, marker='o')
axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2, marker='s')
axes[0, 0].set_title('Model Accuracy', fontsize=14, weight='bold')
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Accuracy', fontsize=12)
axes[0, 0].legend(loc='lower right')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Loss
axes[0, 1].plot(history.history['loss'], label='Train Loss', linewidth=2, marker='o')
axes[0, 1].plot(history.history['val_loss'], label='Val Loss', linewidth=2, marker='s')
axes[0, 1].set_title('Model Loss', fontsize=14, weight='bold')
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Loss', fontsize=12)
axes[0, 1].legend(loc='upper right')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
            yticklabels=class_names, ax=axes[1, 0], cbar_kws={'label': 'Count'})
axes[1, 0].set_title('Confusion Matrix', fontsize=14, weight='bold')
axes[1, 0].set_ylabel('True Label', fontsize=12)
axes[1, 0].set_xlabel('Predicted Label', fontsize=12)

# Plot 4: ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)

axes[1, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
axes[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
axes[1, 1].set_xlim([0.0, 1.0])
axes[1, 1].set_ylim([0.0, 1.05])
axes[1, 1].set_xlabel('False Positive Rate', fontsize=12)
axes[1, 1].set_ylabel('True Positive Rate', fontsize=12)
axes[1, 1].set_title('ROC Curve', fontsize=14, weight='bold')
axes[1, 1].legend(loc="lower right")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Performance visualizations saved as 'model_performance.png'")

# ============================================================================
# CLASSIFICATION ON 5 RANDOM TEST IMAGES
# ============================================================================
print("\n🎯 Classifying 5 Random Test Images...")

# Get random test images
all_test_images = []
for category in ['O', 'R']:
    cat_path = test_dir / category
    images = list(cat_path.glob('*.jpg'))
    for img_path in images:
        all_test_images.append((img_path, category))

random.shuffle(all_test_images)
sample_images = all_test_images[:5]

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle('Random Test Image Classifications', fontsize=16, weight='bold', y=1.05)

for idx, (img_path, true_label) in enumerate(sample_images):
    # Load and preprocess image
    img = Image.open(img_path).resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array, verbose=0)[0][0]
    predicted_label = 'R' if prediction > 0.5 else 'O'
    predicted_name = 'Recyclable' if prediction > 0.5 else 'Organic'
    true_name = 'Recyclable' if true_label == 'R' else 'Organic'
    confidence = prediction if prediction > 0.5 else (1 - prediction)

    # Display
    axes[idx].imshow(img)
    axes[idx].axis('off')

    color = 'green' if predicted_label == true_label else 'red'
    title = f'True: {true_name}\nPred: {predicted_name}\nConf: {confidence:.2%}'
    axes[idx].set_title(title, fontsize=10, color=color, weight='bold')

plt.tight_layout()
plt.savefig('random_predictions.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Random predictions saved as 'random_predictions.png'")

# ============================================================================
# REAL-TIME CAMERA CLASSIFICATION FUNCTION
# ============================================================================
print("\n📸 Real-Time Classification Function Ready!")

def classify_uploaded_image(image_path):
    """
    Classify a single uploaded image

    Args:
        image_path: Path to the image file (str or Path object)

    Returns:
        Dictionary with classification results
    """
    try:
        # Load and preprocess image
        img = Image.open(image_path).resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array, verbose=0)[0][0]

        # Determine class
        if prediction > 0.5:
            waste_type = "Recyclable"
            confidence = prediction
            recommendation = "♻️ Please dispose in the RECYCLABLE bin"
        else:
            waste_type = "Organic"
            confidence = 1 - prediction
            recommendation = "🌱 Please dispose in the ORGANIC bin"

        # Display result
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Show image
        axes[0].imshow(img)
        axes[0].axis('off')
        axes[0].set_title(f'Uploaded Image', fontsize=14, weight='bold')

        # Show classification
        axes[1].text(0.5, 0.6, f'Classification: {waste_type}',
                    ha='center', va='center', fontsize=20, weight='bold')
        axes[1].text(0.5, 0.4, f'Confidence: {confidence:.2%}',
                    ha='center', va='center', fontsize=16)
        axes[1].text(0.5, 0.2, recommendation,
                    ha='center', va='center', fontsize=14, style='italic')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig('uploaded_classification.png', dpi=300, bbox_inches='tight')
        plt.show()

        result = {
            'waste_type': waste_type,
            'confidence': confidence,
            'recommendation': recommendation
        }

        print(f"\n✅ Classification Result:")
        print(f"   Type: {waste_type}")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   {recommendation}")

        return result

    except Exception as e:
        print(f"❌ Error classifying image: {str(e)}")
        return None

print("\n" + "="*80)
print("HOW TO USE REAL-TIME CLASSIFICATION:")
print("="*80)
print("""
# To classify a new image, use:
result = classify_uploaded_image('path/to/your/image.jpg')

# Example:
result = classify_uploaded_image('my_waste_photo.jpg')
""")

# ============================================================================
# PERFORMANCE SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PERFORMANCE SUMMARY")
print("="*80)
print(f"""
📊 Model Performance Metrics:
   • Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)
   • Precision: {test_precision:.4f}
   • Recall:    {test_recall:.4f}
   • F1-Score:  {test_f1:.4f}
   • AUC:       {roc_auc:.4f}

📁 Dataset Statistics:
   • Total Training Images:   {train_generator.samples}
   • Total Validation Images: {validation_generator.samples}
   • Total Test Images:       {test_generator.samples}
   • Organic Images (Train):  {train_counts.get('O', 0)}
   • Recyclable Images (Train): {train_counts.get('R', 0)}

💾 Saved Files:
   ✓ waste_classification_model.h5  (Trained model)
   ✓ dataset_distribution.png        (Dataset visualization)
   ✓ sample_images.png               (Sample images)
   ✓ model_performance.png           (Training & evaluation metrics)
   ✓ random_predictions.png          (Test predictions)

🎯 Ready for Conference Presentation!
""")
print("="*80)
result = classify_uploaded_image('/content/used-plastic-bottles-background-recyclable-waste-concept-2A4T2DM.jpg')