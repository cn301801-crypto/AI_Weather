import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib

# Load data
df = pd.read_csv("Numeric_weather.csv")

X = df[['Temp_C', 'Rel Hum_%', 'Press_kPa', 'Wind Speed_km/h']]
y = df['weather']

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# ✅ Fix label gaps
unique_labels = np.unique(y)
label_map = {old: new for new, old in enumerate(unique_labels)}
y = np.array([label_map[val] for val in y])

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Accuracy
print("Accuracy:", model.score(X_test, y_test))

# Save model
joblib.dump(model, "xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")
np.save("numeric_classes.npy", encoder.classes_)

print("✅ Numerical model trained and saved")