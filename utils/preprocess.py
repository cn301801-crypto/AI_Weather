import cv2
import numpy as np

IMG_SIZE = 64

def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    return img.reshape(1, IMG_SIZE, IMG_SIZE, 3)

def preprocess_numeric(data, scaler, pca, kmeans):
    data = scaler.transform([data])
    data_pca = pca.transform(data)
    cluster = kmeans.predict(data_pca)
    return np.hstack((data, cluster.reshape(-1,1)))