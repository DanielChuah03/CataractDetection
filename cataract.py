import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model('inceptionv3_cataract_model.keras')

# Function to preprocess and predict
def predict(image):
    # Preprocess image
    img = image.resize((299, 299))  # Resize image
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Predict
    prediction = model.predict(img_array)
    return prediction

st.title('Cataract Detection')
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    prediction = predict(image)
    st.write(f'Prediction: {"Normal" if prediction[0] > 0.5 else "Cataract"}')