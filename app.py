import streamlit as st
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
import tensorflow as tf

#  Set up the Streamlit app configuration (title and icon)
st.set_page_config(page_title="Medical Diagnosis", page_icon="🩺")

#  Function to build the Chest X-ray model architecture and load weights
#ensure similar architecture as the one used during training(use same input shape, layers, etc.)
def build_chest_model():
    model = Sequential([
        Input(shape=(64, 64, 3)),  # Input layer for 64x64 RGB images
        Conv2D(32, (3, 3), activation='relu'),  # First convolutional layer
        MaxPooling2D(),  # Pooling layer
        Conv2D(64, (3, 3), activation='relu'),  # Second convolutional layer
        MaxPooling2D(),
        Conv2D(128, (3, 3), activation='relu'),  # Third convolutional layer
        MaxPooling2D(),
        Flatten(),  # Flatten to 1D
        Dense(32, activation='relu'),  # Dense layer
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    model.load_weights('Chest-xray_model.h5')  # Load pre-trained weights
    return model

# Function to build the Brain Tumor model architecture and load weights
#ensure similar architecture as the one used during training(use same input shape, layers, etc.)
def build_brain_model():
    model = Sequential([
        Input(shape=(64, 64, 3)),  # Input layer for 64x64 RGB images
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    model.load_weights('brain_tumour_model.h5')  # Load pre-trained weights
    return model

#  Function to preprocess the image and make a prediction using the selected model
def predict_image(model, image, target_size, labels):
    image = image.resize(target_size)  # Resize image to model's expected input
    img_array = np.array(image) / 255.0  # Normalize pixel values
    if img_array.ndim == 2:  # If grayscale, convert to RGB
        img_array = np.stack((img_array,) * 3, axis=-1)
    if img_array.shape[-1] == 1:  # If single channel, duplicate to 3 channels
        img_array = np.concatenate([img_array] * 3, axis=-1)
    img_array = img_array.reshape(1, target_size[0], target_size[1], 3)  # Add batch dimension
    prediction = model.predict(img_array)[0][0]  # Get prediction score
    label = labels[1] if prediction >= 0.5 else labels[0]  # Assign label based on threshold
    confidence = prediction if prediction >= 0.5 else 1 - prediction  # Calculate confidence
    return f"{label} ({confidence:.2f} confidence)"



#  Streamlit UI setup
#  Create three tabs for navigation: Home, Chest X-Ray, Brain Tumor
tabs = st.tabs([" Home", "🫁 Chest X-Ray Diagnosis", "🧠 Brain Tumor Diagnosis"])

#  Home page tab: Shows welcome message and instructions
with tabs[0]:
    st.title("🩺 Medical Diagnosis App")
    st.markdown("""
    Welcome to the **Medical Diagnosis Platform** powered by AI!  
    This tool helps diagnose:
    - **🫁 Pneumonia** using Chest X-Rays  
    - **🧠 Brain Tumors** using MRI scans  
    
    ### Instructions:
    1. Choose a tab above based on the type of diagnosis.
    2. Upload an image (JPG/PNG).
    3. Let the model analyze and return its prediction.
    
    **Disclaimer:** This tool is for educational purposes only and not a substitute for professional medical advice.
    """)

#  Chest X-Ray Diagnosis tab: Upload image, run model, show result
with tabs[1]:
    st.header("🫁 Chest X-Ray Diagnosis")
    chest_image = st.file_uploader("Upload a Chest X-Ray Image", type=["jpg", "jpeg", "png"], key="chest")
    if chest_image:
        image = Image.open(chest_image)  # Open uploaded image
        st.image(image, caption="Uploaded Chest X-ray", use_container_width=True)  # Display image
        model = build_chest_model()  # Load chest model
        result = predict_image(model, image, (64, 64), ["Normal", "Pneumonia"])  # Predict
        st.success(f"Prediction: {result}")  # Show result

#  Brain Tumor Diagnosis tab: Upload image, run model, show result
with tabs[2]:
    st.header("🧠 Brain Tumor Diagnosis")
    brain_image = st.file_uploader("Upload a Brain MRI Image", type=["jpg", "jpeg", "png"], key="brain")
    if brain_image:
        image = Image.open(brain_image)  # Open uploaded image
        st.image(image, caption="Uploaded Brain MRI", use_container_width=True)  # Display image
        model = build_brain_model()  # Load brain model
        result = predict_image(model, image, (64, 64), ["Brain Tumor", "Normal"])  # Predict
        st.success(f"Prediction: {result}")  # Show result
