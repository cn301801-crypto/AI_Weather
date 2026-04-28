import numpy as np
import cv2
import os

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

IMG_SIZE = 64
data = []
labels = []

for category in os.listdir("data"):
    path = os.path.join("data", category)

    if not os.path.isdir(path):
        continue

    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        img_array = cv2.imread(img_path)

        if img_array is None:
            continue

        img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        data.append(img_array)
        labels.append(category)

X = np.array(data) / 255.0

encoder = LabelEncoder()
y = encoder.fit_transform(labels)

model = Sequential([
    Input(shape=(64,64,3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=5)

# Save model
model.save("cnn_model.h5")

# Save encoder classes
np.save("image_classes.npy", encoder.classes_)

print("✅ Image model trained and saved")