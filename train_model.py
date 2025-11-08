import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import cv2
import os
import kagglehub
import random

# Download dataset
path = kagglehub.dataset_download("techsash/waste-classification-data")
print("✅ Path to dataset files:", path)

# Define dataset directories
train_dir = os.path.join(path, "DATASET/TRAIN")
test_dir = os.path.join(path, "DATASET/TEST")

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    subset='training',
    class_mode='categorical'
)

val_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    subset='validation',
    class_mode='categorical'
)

class_names = list(train_data.class_indices.keys())
print("♻️ Class Names:", class_names)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(train_data, epochs=5, validation_data=val_data)

# Plot accuracy and loss
sns.set_style("whitegrid")
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='green')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='red')
plt.plot(history.history['val_loss'], label='Val Loss', color='blue')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Confusion Matrix
val_preds = model.predict(val_data)
y_pred = np.argmax(val_preds, axis=1)
y_true = val_data.classes

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(7, 7))
disp.plot(ax=ax, cmap='Greens', colorbar=False)
plt.title("Confusion Matrix")
plt.show()

# Random test predictions
test_dir = os.path.join(path, "DATASET", "TEST")
class_names = ['R', 'O']

plt.figure(figsize=(20, 5))
for i in range(5):
    chosen_class = random.choice(os.listdir(test_dir))
    chosen_folder = os.path.join(test_dir, chosen_class)
    chosen_img = random.choice(os.listdir(chosen_folder))
    img_path = os.path.join(chosen_folder, chosen_img)

    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized / 255.0, axis=0)

    pred = model.predict(img_array)
    pred_class = np.argmax(pred)
    confidence = np.max(pred)

    plt.subplot(1, 5, i + 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Pred: {class_names[pred_class]}\n({confidence*100:.2f}%)",
              color='darkgreen', fontsize=12)

plt.suptitle("Random 5 Predictions from Test Dataset", fontsize=16, color='darkblue')
plt.tight_layout()
plt.show()

# Save trained model
model.save("waste_classifier.h5")
print("✅ Model saved as waste_classifier.h5")
