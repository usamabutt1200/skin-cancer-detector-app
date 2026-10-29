import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import urllib.request
import zipfile

@st.cache_resource
def load_real_professional_model():
    model_folder = 'melanoma_detection_model'
    zip_name = 'model.zip'
    
    # Aapke Hugging Face account ka direct download link
    hf_url = "https://huggingface.co"
    
    if not os.path.exists(model_folder):
        with st.spinner('Hugging Face Cloud se Real AI Model load ho raha hai... Ismein 20-30 seconds lag sakte hain.'):
            try:
                # File download aur extract karne ka loop
                urllib.request.urlretrieve(hf_url, zip_name)
                with zipfile.ZipFile(zip_name, 'r') as zip_ref:
                    zip_ref.extractall('.')
                os.remove(zip_name)
            except Exception as e:
                st.error(f"Cloud connection mein masla aaya: {e}")
                
    return tf.keras.models.load_model(model_folder)

# Real model load ho raha hai
try:
    model = load_real_professional_model()
except Exception as e:
    model = None

# Web UI Layout Setup
st.set_page_config(page_title="Skin Cancer Detector AI", layout="centered")
st.title("🩺 AI Skin Cancer (Melanoma) Detector")
st.write("Apni skin lesion ki image upload karen taake AI model real-time mein check kar sake.")

uploaded_file = st.file_uploader("Image select karen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("🔄 AI Real Analysis chal raha hai...")
    
    if model is None:
        st.error("⚠️ Model backend par load nahi ho saka. Hugging Face file check karen.")
    else:
        # Preprocessing (Model input requirements matching)
        img = image.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Real Inference
        predictions = model.predict(img_array)
        class_names = ['Benign (Non-Cancerous)', 'Malignant (Cancerous)']
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100
        
        # UI Elements response generation
        if "Malignant" in predicted_class:
            st.error(f"AI Prediction Result: {predicted_class}")
        else:
            st.success(f"AI Prediction Result: {predicted_class}")
            
        st.info(f"AI Real Confidence Score: {confidence:.2f}%")
