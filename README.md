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

## Dataset

The dataset used for training the model is from the **FER-2013** dataset, available on Kaggle. You can find it [here](https://www.kaggle.com/datasets/msambare/fer2013).

### About the Dataset

- The data consists of **48x48 pixel grayscale images** of faces. The faces have been automatically registered so that the face is more or less centered and occupies a similar amount of space in each image.
- The task is to categorize each face based on the **emotion** shown in the facial expression into one of seven categories:
  - **0 = Angry**
  - **1 = Disgust**
  - **2 = Fear**
  - **3 = Happy**
  - **4 = Sad**
  - **5 = Surprise**
  - **6 = Neutral**

### Data Statistics:
- **Training Set**: 28,709 examples
- **Public Test Set**: 3,589 examples

## Project Structure  
Facial-Expression-Recognition/  
├── static/  
│   ├── uploads/                       
│   ├── style.css                     
│   └── app.js                       
├── templates/  
│   └── index.html                      
├── emotion_model_improved.h5          
├── app.py                             
├── train.py                           
├── requirements.txt                  

## How to Run the Project

## How to Run the Project

1. **Clone the repository**:

```bash
git clone https://github.com/your-username/Facial-Expression-Recognition.git
cd Facial-Expression-Recognition
Install the required dependencies:

```bash
pip install -r requirements.txt
Run the Flask app:

bash
Copy
Edit
python app.py
Open the app in your browser:

Go to

text
Copy
Edit
http://127.0.0.1:5000/
Use the Web Interface:

Upload a face image and click the "Detect Expression" button to view the prediction.

