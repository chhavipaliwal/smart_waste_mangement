from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os

app = Flask(__name__)

# Create upload folder inside static if not exist
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = load_model("waste_classifier.h5")

# Define classes
class_names = ['Recyclable', 'Organic']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))

    # Save image
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Read and preprocess
    img = cv2.imread(file_path)
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized / 255.0, axis=0)

    # Predict
    pred = model.predict(img_array)
    pred_class = np.argmax(pred)
    confidence = np.max(pred)

    # Prepare result
    result = {
        "class": class_names[pred_class],
        "confidence": f"{confidence * 100:.2f}%",
        "image_path": "/" + file_path
    }

    # Render result back to index.html
    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
