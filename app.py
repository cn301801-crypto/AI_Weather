from flask import Flask, render_template, request
import numpy as np
import cv2
import os
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# -----------------------------
# Upload Folder Setup
# -----------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -----------------------------
# Load Models (SAFE MODE)
# -----------------------------
cnn_model = None
xgb_model = None
scaler = None
image_classes = None
numeric_classes = None

try:
    cnn_model = load_model("models/cnn_model.h5")
    print("✅ CNN model loaded")
except Exception as e:
    print("❌ CNN model not found:", e)

try:
    xgb_model = joblib.load("models/xgb_model.pkl")
    print("✅ XGB model loaded")
except Exception as e:
    print("❌ XGB model not found:", e)

try:
    scaler = joblib.load("models/scaler.pkl")
    print("✅ Scaler loaded")
except Exception as e:
    print("❌ Scaler not found:", e)

try:
    image_classes = np.load("models/image_classes.npy", allow_pickle=True)
    numeric_classes = np.load("models/numeric_classes.npy", allow_pickle=True)
    print("✅ Classes loaded")
except Exception as e:
    print("❌ Class files not found:", e)

IMG_SIZE = 64

# -----------------------------
# Prediction Functions
# -----------------------------
def predict_image(image_path):
    if cnn_model is None or image_classes is None:
        return None

    if image_path is None:
        return None

    img = cv2.imread(image_path)

    if img is None:
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    pred = cnn_model.predict(img)
    return image_classes[np.argmax(pred)]


def predict_numeric(data):
    if xgb_model is None or scaler is None or numeric_classes is None:
        return "Model not available"

    data = scaler.transform([data])
    pred = xgb_model.predict(data)
    return numeric_classes[pred[0]]


def final_prediction(image_path, num_data):
    img_pred = predict_image(image_path)
    num_pred = predict_numeric(num_data)

    # Hybrid logic
    if img_pred is None:
        return num_pred
    elif img_pred == num_pred:
        return img_pred
    else:
        return num_pred


# -----------------------------
# Routes
# -----------------------------

# 🏠 Home Page
@app.route("/")
def home():
    return render_template("home.html")


# 📘 About Page
@app.route("/about")
def about():
    return render_template("about.html")


# ⚙️ How It Works Page
@app.route("/how")
def how():
    return render_template("how.html")


# 🔍 Prediction Page
@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    image_path = None

    if request.method == "POST":
        try:
            file = request.files.get("image")

            temp = float(request.form.get("temp"))
            hum = float(request.form.get("hum"))
            press = float(request.form.get("press"))
            wind = float(request.form.get("wind"))

            filepath = None

            # Save image if uploaded
            if file and file.filename != "":
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                image_path = filepath

            # Get prediction
            result = final_prediction(filepath, [temp, hum, press, wind])

        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template("predict.html", result=result, image_path=image_path)


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)