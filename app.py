from flask import Flask, render_template, request
import numpy as np
import cv2
import joblib
from tensorflow.keras.models import load_model
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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
    cnn_model = load_model("models/cnn_model.h5", compile=False)
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

IMG_SIZE = 32

# -----------------------------
# Prediction Functions
# -----------------------------
def predict_image(image_path):

    if image_path is None:
        return None

    img = cv2.imread(image_path)

    if img is None:
        return None

    img = cv2.resize(img, (32, 32))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = cnn_model.predict(img, verbose=0)

    return image_classes[int(np.argmax(pred))]

def predict_numeric(data):
    if xgb_model is None or scaler is None or numeric_classes is None:
        return "Model not available"

    data = scaler.transform([data])
    pred = xgb_model.predict(data)
    return numeric_classes[int(pred[0])]


def final_prediction(image_path, num_data):

    print("Starting image prediction")
    img_pred = predict_image(image_path)

    print("Image prediction done")

    print("Starting numeric prediction")
    num_pred = predict_numeric(num_data)

    print("Numeric prediction done")

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

    if request.method == "POST":

        try:
            print("FORM SUBMITTED")

            file = request.files.get("image")

            temp = float(request.form.get("temp"))
            hum = float(request.form.get("hum"))
            press = float(request.form.get("press"))
            wind = float(request.form.get("wind"))

            filepath = None

            # Save image
            if file and file.filename != "":
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)

                print("Image saved:", filepath)

            # Prediction
            result = final_prediction(filepath, [temp, hum, press, wind])

            print("Prediction:", result)

        except Exception as e:
            print("ERROR:", str(e))
            result = f"Error: {str(e)}"

    return render_template("predict.html", result=result)

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)