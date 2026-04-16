import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

# Load model
model = load_model("model/plant_disease_model.h5")

# Load class labels
class_names = sorted(os.listdir("dataset"))

# Load image
img_path = "test.jpg"
img = cv2.imread(img_path)

# Preprocess
img = cv2.resize(img,(224,224))
img = img/255.0
img = np.reshape(img,(1,224,224,3))

# Predict
prediction = model.predict(img)
index = np.argmax(prediction)

print("Prediction:", class_names[index])
print("Confidence:", round(np.max(prediction)*100,2), "%")