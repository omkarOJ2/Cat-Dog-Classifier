from flask import Flask, request, send_from_directory, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import time

# ----------------- Config -----------------
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "cats_dogs_model2.keras"
IMAGE_SIZE = (150, 150)

# ----------------- Load Model -----------------
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None


# ----------------- Prediction Function -----------------
def predict_image(img_path):
    if model is None:
        return "Model not loaded"

    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    preds = model.predict(x)
    time.sleep(1.5)  # fake delay for demo spinner

    return "🐶 Dog" if preds[0] > 0.5 else "🐱 Cat"


# ----------------- Routes -----------------
@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        if "file" not in request.files or request.files["file"].filename == "":
            return render_template("index.html", result="⚠️ No file selected")

        file = request.files["file"]
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        result = predict_image(filepath)
        return render_template("index.html", result=result, filename=file.filename)

    return render_template("index.html")


# ----------------- Serve Uploaded Images -----------------
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# ----------------- Run App -----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
