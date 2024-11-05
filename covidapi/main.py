import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import preprocess_input

from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Input

# Define class labels
class_names = ['COVID', 'NonCOVID']

# Rebuild the model structure without using `batch_shape`
def build_model():


# Set up VGG19 with locally downloaded weights
    base_model = VGG19(weights='models/vgg19_weights.h5', include_top=False, input_shape=(224, 224, 3))

#base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)  # Assuming binary classification
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

model = build_model()

# Load weights only, bypassing potential issues with `batch_shape`
model.load_weights(r'C:\dev\diagnose-covid-using-cnn-model\covidapi\models\vgg19_trained_model.h5')

# Preprocess uploaded image
def preprocess_image(image):
    img = Image.open(image).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 224, 224, 3))
    img_array = preprocess_input(img_array)
    return img_array


# Set up the Streamlit app
st.title('COVID-19 Prediction Web Application')

# Upload the image
uploaded_image = st.file_uploader("Upload a CT-scan image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((100, 100))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            img_array = preprocess_image(uploaded_image)
            result = model.predict(img_array)
            predicted_class = np.argmax(result)
            prediction = class_names[predicted_class]
            st.success(f'Prediction: {prediction}')