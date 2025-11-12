from flask import Flask, render_template, request, redirect, url_for
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os
import gdown

app = Flask(__name__)

# Create upload folder if it doesn’t exist
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model path and Google Drive link
MODEL_PATH = "waste_classification_model.h5"
# ✅ Updated Google Drive direct download link
MODEL_URL = "https://drive.google.com/uc?id=1gBbV1liz3_tevUu-QVaOMokrgtaFd2SR"

# Auto-download model if not present
if not os.path.exists(MODEL_PATH):
    print("📦 Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load trained model safely (handles older Keras formats)
try:
    model = keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Normal load failed: {e}")
    print("🔁 Trying legacy load...")
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Legacy model loaded!")

# Define class names
class_names = ['Organic', 'Recyclable']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Ensure a file was uploaded
    if 'file' not in request.files or request.files['file'].filename == '':
        return redirect(url_for('home'))

    file = request.files['file']

    # Save uploaded image
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Read and preprocess image
    img = cv2.imread(file_path)
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized / 255.0, axis=0)

    # Predict
    pred = model.predict(img_array)[0][0]

    # Determine class
    if pred > 0.5:
        pred_class = "Recyclable"
        confidence = pred
    else:
        pred_class = "Organic"
        confidence = 1 - pred

    result = {
        "class": pred_class,
        "confidence": round(float(confidence) * 100, 2),
        "image_path": "/" + file_path
    }

    return render_template('index.html', result=result)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
