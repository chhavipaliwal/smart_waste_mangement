from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os
import gdown  # 👈 added to auto-download model

app = Flask(__name__)

# Create upload folder inside static if not exist
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Path for model file
MODEL_PATH = "waste_classification_model.h5"

# Google Drive file ID
MODEL_URL = "https://drive.google.com/uc?id=1gBbV1liz3_tevUu-QVaOMokrgtaFd2SR"

# 🔽 Auto-download model if missing
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load trained model
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# Class names based on your training setup (binary)
class_names = ['Organic', 'Recyclable']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return redirect(url_for('home'))
    
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))

    # Save image
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Read and preprocess image
    img = cv2.imread(file_path)
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized / 255.0, axis=0)

    # Predict using sigmoid output
    pred = model.predict(img_array)[0][0]

    # Threshold at 0.5
    if pred > 0.5:
        pred_class = "Recyclable"
        confidence = pred
    else:
        pred_class = "Organic"
        confidence = 1 - pred

    # Prepare result
    result = {
        "class": pred_class,
        "confidence": float(confidence),
        "image_path": "/" + file_path
    }

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
