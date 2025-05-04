# Facial-Expression-Recognition
This project detects 7 emotions — Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise — from face images. It’s trained on the FER2013 dataset using a deep learning model.  

## Features  
- Upload an image and detect the facial expression  
- Shows prediction with confidence score  
- Clean and responsive frontend  
- Deep learning backend using TensorFlow + Flask  

## Tech Stack  
- **Frontend**: HTML, CSS, JavaScript  
- **Backend**: Python, Flask  
- **Model**: MobileNetV2 (transfer learning), trained with grayscale 48x48 facial expression images

<pre> ``` Facial-Expression-Recognition/ ├── static/ │ ├── uploads/ # Stores uploaded images for prediction │ ├── style.css # CSS for styling the web page │ └── app.js # JavaScript for handling predictions ├── templates/ │ └── index.html # HTML frontend ├── emotion_model_improved.h5 # Trained facial expression model ├── app.py # Flask backend to handle requests ├── train.py # Script to train the emotion detection model ├── requirements.txt # Python dependencies ``` </pre>
