from flask import Flask, render_template, request, jsonify
import numpy as np
import os, joblib, cv2
from tensorflow.keras.models import load_model

from utils.preprocess import preprocess_image, preprocess_numeric
from utils.fusion import combine

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load models
cnn = load_model("models/cnn_model.h5")
xgb = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")
pca = joblib.load("models/pca.pkl")
kmeans = joblib.load("models/kmeans.pkl")

image_classes = np.load("models/image_classes.npy", allow_pickle=True)
numeric_classes = np.load("models/numeric_classes.npy", allow_pickle=True)

# ---------- Predictions ----------
def predict_image(path):
    img = preprocess_image(path)
    if img is None:
        return None
    pred = cnn.predict(img)
    return image_classes[np.argmax(pred)]

def predict_numeric(data):
    final = preprocess_numeric(data, scaler, pca, kmeans)
    pred = xgb.predict(final)
    return numeric_classes[pred[0]]

def final_prediction(path, num):
    return combine(predict_image(path), predict_numeric(num))

# ---------- Routes ----------
@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/predict", methods=["GET","POST"])
def predict_page():
    result = None
    if request.method == "POST":
        file = request.files["image"]
        temp = float(request.form["temp"])
        hum = float(request.form["hum"])
        press = float(request.form["press"])
        wind = float(request.form["wind"])

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        result = final_prediction(filepath, [temp, hum, press, wind])

    return render_template("index.html", result=result)

# ---------- API ----------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    file = request.files["image"]
    temp = float(request.form["temp"])
    hum = float(request.form["hum"])
    press = float(request.form["press"])
    wind = float(request.form["wind"])

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    result = final_prediction(filepath, [temp, hum, press, wind])

    return jsonify({"prediction": result})

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)