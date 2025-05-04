# This script is no longer needed as we are now using the frontend for predictions
# It was originally used for testing the emotion model on a single image input
import tensorflow as tf
import numpy as np
import cv2
import os

model = tf.keras.models.load_model("emotion_model_improved.h5")

img_path = "C:\\Users\\HP\\Desktop\\Claysys\\train\\disgust\\Training_8514439.jpg"  

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (48, 48))  

img = img / 255.0
img = np.expand_dims(img, axis=-1)  
img = np.expand_dims(img, axis=0)   

pred = model.predict(img)
class_index = np.argmax(pred)
confidence = np.max(pred)

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

print(f"Predicted emotion: {emotion_labels[class_index]} (confidence: {confidence:.2f})")
