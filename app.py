import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Folder base par model ko load karte hain
@st.cache_resource
def load_my_model():
    # Folder ka naam pass kar rahe hain jismein model components hain
    return tf.keras.models.load_model('melanoma_detection_model')

model = load_my_model()

# Web UI Layout
st.set_page_config(page_title="Skin Cancer Detector", layout="centered")
st.title("🩺 AI Skin Cancer (Melanoma) Detector")
st.write("Apni skin lesion ki image upload karen taake AI model check kar sake.")

# Image upload area
uploaded_file = st.file_uploader("Image select karen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("🔄 AI Analysis chal raha hai...")
    
    # Preprocessing
    img = image.resize((128, 128))
    img_array = np.array(img)
    img_array = img_array / 255.0  # Pixels normalize karna
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction execution
    predictions = model.predict(img_array)
    class_names = ['Benign (Non-Cancerous)', 'Malignant (Cancerous)']
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    
    # Display Result safely
    if "Malignant" in predicted_class:
        st.error(f"Prediction: {predicted_class}")
    else:
        st.success(f"Prediction: {predicted_class}")
        
    st.info(f"Confidence Score: {confidence:.2f}%")
