import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

my_image = r'C:\dev\diagnose-covid-using-cnn-model\covidapi\models\trained_model.h5'

# Load pre-trained model
model = tf.keras.models.load_model(my_image)

# Define class labels for covid and non-covid ct-scans
class_names = ['COVID', 'NonCOVID']


# Set the function to preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((256, 256))
    img = img.convert('L')  # Convert to grayscale
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 256, 256, 1))
    return img_array

# Set Streamlit App
st.title('COVID and Non-COVID CT-Scan Images Classifier')

uploaded_image = st.file_uploader("Upload a CT-scan image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((100, 100))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess uploaded image
            img_array = preprocess_image(uploaded_image)

            # Predict using the pre-trained model
            result = model.predict(img_array)
            # st.write(str(result))
            predicted_class = np.argmax(result)
            prediction = class_names[predicted_class]

            st.success(f'Prediction: {prediction}')
