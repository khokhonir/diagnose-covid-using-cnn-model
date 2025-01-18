import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Define class labels
class_names = ['COVID', 'NonCOVID']

# Load the complete model
model = load_model(r'C:\dev\diagnose-covid-using-cnn-model\covidapi\models\pretrained_model.h5')

# Preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match model input size
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array /= 255.0  # Normalize pixel values
    return image_array

# Streamlit App
def main():
    st.title("COVID-19 Prediction from CT Scan")
    st.write("Upload a CT scan image to predict whether it indicates COVID-19 or not.")

    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Load and preprocess the image
        image = load_img(uploaded_file)
        preprocessed_image = preprocess_image(image)

        # Predict using the model
        st.write("Processing image...")
        prediction = model.predict(preprocessed_image)
        class_label = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Display prediction result
        st.write(f"Prediction: **{class_label}**")
        st.write(f"Confidence: **{confidence:.2f}**")

# Run the app
if __name__ == "__main__":
    main()