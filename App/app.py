import os
import numpy as np
import cv2
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model
model = load_model("../model/plant_disease_model.h5")

# Load class names from dataset
class_names = sorted(os.listdir("../dataset"))

@app.route("/", methods=["GET","POST"])
def index():
    prediction = None
    confidence = None
    treatment = None
    filename = None

    if request.method == "POST":
        file = request.files["image"]
        filename = file.filename
        filepath = "static/" + filename
        file.save(filepath)

        # Image preprocessing
        img = cv2.imread(filepath)
        img = cv2.resize(img,(224,224))
        img = img/255.0
        img = np.reshape(img,(1,224,224,3))

        # Predict
        pred = model.predict(img)
        index = np.argmax(pred)

        prediction = class_names[index]
        confidence = round(np.max(pred)*100,2)

        # Treatment suggestions (based on your dataset)
        treatment_dict = {
            "Pepper__bell__Bacterial_spot": "Apply copper-based fungicide and remove infected leaves.",
            "Pepper__bell__healthy": "Plant is healthy. Maintain proper watering and sunlight.",
            "Potato__Early_blight": "Use fungicide spray and practice crop rotation.",
            "Potato__healthy": "Plant is healthy. Continue good farming practices.",
            "Potato__Late_blight": "Remove infected leaves and apply protective fungicide immediately.",
            "Tomato_Bacterial_spot": "Use copper fungicide and avoid overhead watering.",
            "Tomato_healthy": "Plant is healthy. Maintain proper nutrition and watering."
        }

        treatment = treatment_dict.get(prediction, "Consult agricultural expert.")

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        treatment=treatment,
        filename=filename
    )

if __name__ == "__main__":
    app.run(debug=True)