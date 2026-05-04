# Arabic Sign Language Recognition System

A Deep Learning-Based Classification System for Arabic Sign Alphabet using MobileNet.

---

## About the Project

This project is developed to recognize Arabic Sign Language hand gestures using Deep Learning.

People with hearing and speech disabilities use sign language for communication. This system helps identify Arabic hand signs and converts them into understandable output.

The model is trained using the ArASL dataset and MobileNet CNN architecture for accurate image classification.

The project focuses on improving communication and making sign language recognition easier using Artificial Intelligence.

---

## Problem Statement

Many people do not understand sign language, which creates communication problems for hearing-impaired individuals.

This project solves that problem by automatically recognizing Arabic hand gestures and predicting the correct sign alphabet.

---

## Dataset Used

### ArASL Dataset (Arabic Alphabet Sign Language)

The dataset contains thousands of labeled hand gesture images representing Arabic sign alphabets.

Each image is used to train the model for better gesture recognition.

---

## Technologies Used

* Python
* TensorFlow
* Keras
* OpenCV
* NumPy
* Streamlit
* MobileNet CNN
* Google Colab

---

## Features

* Upload hand gesture image
* Automatic gesture recognition
* Deep Learning-based prediction
* MobileNet lightweight model
* Fast and accurate classification
* Streamlit web application interface

---

## Project Workflow

1. Load ArASL dataset
2. Preprocess gesture images
3. Resize and normalize images
4. Train MobileNet CNN model
5. Extract image features
6. Classify Arabic hand gestures
7. Display prediction output

---

## Project Files

* `arabic_sign_app.py` → Main Streamlit application
* `arabic_sign_deploy_streamlit_colab.ipynb` → Google Colab training notebook
* `arabic_sign.pkl` → Label encoder support file
* `arabic_sign.h5` → Trained deep learning model
* `arabic_sign.keras` → Saved Keras model
* Project Documentation PDF

---

## How to Run the Project

### Step 1: Install Required Libraries

```bash id="8n2jcp"
pip install tensorflow keras opencv-python numpy streamlit pillow scikit-learn
```

---

### Step 2: Run the Streamlit App

```bash id="v88f7w"
streamlit run arabic_sign_app.py
```

---

### Step 3: Open Browser

If it does not open automatically, open this link:

```bash id="j6x19a"
http://localhost:8501
```

---

## Output

The system predicts the Arabic sign alphabet from the uploaded hand gesture image and displays the recognized result.

---

## Future Improvements

* Real-time webcam sign recognition
* Better model accuracy
* Live prediction support
* Multi-language sign language recognition
* Cloud deployment using Streamlit Cloud

---

## Note

Large model files like `.h5` and `.keras` may be stored separately because GitHub has file size upload limits.

---
