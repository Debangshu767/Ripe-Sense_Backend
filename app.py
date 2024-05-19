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

def preprocess_image(image_bytes):
    # Convert image bytes to a numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    # Decode image from numpy array
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Resize the image
    img = cv2.resize(img, (64, 64))
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    preprocessed_image = preprocess_image(contents)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension
    predictions = loaded_model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class]
    predicted_probability = predictions[0][predicted_class]  # Probability of the predicted class
    return {"predicted_class": predicted_class_label, "probability": float(predicted_probability)}