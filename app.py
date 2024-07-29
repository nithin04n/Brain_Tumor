import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2

# Load your trained model
model = tf.keras.models.load_model('brain_tumor.h5')

# Function to process and predict the image
def import_and_predict(image_data, model):
    size = (64, 64)  # Resize to the same size as training images
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img = np.asarray(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis, ...]
    img_reshape = img_reshape / 255.0  # Normalize the image
    prediction = model.predict(img_reshape)
    return prediction

# Streamlit UI
st.write("""
# Brain Tumor Classification
Upload an image of an MRI scan to classify if it contains a brain tumor or not.
""")

file = st.file_uploader("Please upload an MRI scan file", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names = ['No Tumor', 'Yes Tumor']
    st.write(predictions)  # Debug: Print the raw predictions
    predicted_class = np.argmax(predictions, axis=1)
    string = "This image is: " + class_names[predicted_class[0]]
    st.success(string)
