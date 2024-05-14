from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

app = FastAPI()

# Load the model
model_path = './banana_cnn_classifier.h5'
loaded_model = load_model(model_path)

class_labels = ['unripe', 'overripe', 'ripe']

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (64, 64))
    # img = img / 255.0  # Normalize pixel values
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    with open("temp.png", "wb") as f:
        f.write(contents)
    preprocessed_image = preprocess_image("temp.png")
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension
    predictions = loaded_model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class]
    predicted_probability = predictions[0][predicted_class]  # Probability of the predicted class
    return {"predicted_class": predicted_class_label, "probability": float(predicted_probability)}
