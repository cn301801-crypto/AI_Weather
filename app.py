from flask import Flask, render_template, request, jsonify
import numpy as np
import os, joblib
from tensorflow.keras.models import load_model

from utils.preprocess import preprocess_image, preprocess_numeric
from utils.fusion import combine

app = Flask(__name__)

# -----------------------------
# Upload folder setup
# -----------------------------
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# -----------------------------
# Load Models
# -----------------------------
cnn = load_model("models/cnn_model.h5")
xgb = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")
pca = joblib.load("models/pca.pkl")
kmeans = joblib.load("models/kmeans.pkl")

image_classes = np.load("models/image_classes.npy", allow_pickle=True)
numeric_classes = np.load("models/numeric_classes.npy", allow_pickle=True)

# -----------------------------
# Prediction Functions
# -----------------------------
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
    img_pred = predict_image(path)
    num_pred = predict_numeric(num)

    if img_pred is None:
        return num_pred
    elif img_pred == num_pred:
        return img_pred
    else:
        return num_pred

# -----------------------------
# MAIN ROUTE (FIXED)
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    print("✅ Home page accessed")   # DEBUG

    result = None

    if request.method == "POST":
        print("✅ Form submitted")

        file = request.files.get("image")

        temp = float(request.form.get("temp"))
        hum = float(request.form.get("hum"))
        press = float(request.form.get("press"))
        wind = float(request.form.get("wind"))

        filepath = None

        if file and file.filename != "":
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

        result = final_prediction(filepath, [temp, hum, press, wind])

    return render_template("index.html", result=result)

# -----------------------------
# API (Optional)
# -----------------------------
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

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)