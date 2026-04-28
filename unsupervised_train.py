import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# -----------------------------
# LOAD CLEAN DATA
# -----------------------------
df = pd.read_csv("Numeric_weather.csv")

# Use same features as your model
X = df[['Temp_C', 'Rel Hum_%', 'Press_kPa', 'Wind Speed_km/h']]

# -----------------------------
# SCALING
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# PCA (UNSUPERVISED)
# -----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("PCA Done:", X_pca.shape)

# -----------------------------
# KMEANS (UNSUPERVISED)
# -----------------------------
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_pca)

print("KMeans Done")

# -----------------------------
# SAVE MODELS
# -----------------------------
joblib.dump(pca, "pca.pkl")
joblib.dump(kmeans, "kmeans.pkl")

print("✅ Unsupervised models saved")