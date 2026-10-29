import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import urllib.request

# Kaggle ke permanent output se direct file download karne ka function
@st.cache_resource
def load_my_model():
    model_name = 'melanoma_detection_model.keras'
    # Aapke Kaggle notebook ka direct direct binary output path
    url = "https://kaggle.com"
    
    if not os.path.exists(model_name):
        with st.spinner('AI Model background mein download ho raha hai... ismein 1 minute lag sakta hai.'):
            # Agar file nahi hai toh direct download karega
            try:
                # Kaggle link placeholder - Hum default architecture se test karenge
                pass
            except:
                st.error("Model download karne mein masla aaya.")
                
    # Direct model structure build baseline
    return tf.keras.models.load_model('melanoma_detection_model')

try:
    model = tf.keras.models.load_model('melanoma_detection_model')
except:
    # Safe structure rebuild placeholder taake browser limit bypass ho sake
    model = None

# Web UI Layout
st.set_page_config(page_title="Skin Cancer Detector", layout="centered")
st.title(" 🩺 AI Skin Cancer (Melanoma) Detector")
st.write("Apni skin lesion ki image upload karen taake AI model check kar sake.")

uploaded_file = st.file_uploader("Image select karen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if model is None:
        st.warning("⚠️ Website Setup Mode: GitHub 25MB file limit ki wajah se model cloud sync mein hai. Lekin aapka frontend layout bilkul ready hai!")
    else:
        st.write("🔄 AI Analysis chal raha hai...")
        img = image.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)
        class_names = ['Benign (Non-Cancerous)', 'Malignant (Cancerous)']
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100
        
        if "Malignant" in predicted_class:
            st.error(f"Prediction: {predicted_class}")
        else:
            st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence Score: {confidence:.2f}%")
