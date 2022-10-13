import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

# --- SET PAGES ---
st.set_page_config(page_title="Image Classification and Recognition", page_icon=':tanabata_tree:', layout="centered")

# Set Title
st.write('----------------------------------------------------------------')
st.header(" :postal_horn: Image Classification and Recognition Vegetable using Transfer Learning")
st.write('----------------------------------------------------------------')

st.write("")
st.write("This web application can classify 15 types of vegetables, what types of vegetables?")
st.write("""
Bean, Bitter Gourd, Bottle Gourd, Brinjal, Broccoli, Cabbage, Capsicum, Carrot, Cauliflower, Cucumber,  Papaya, Potato,  Pumpkin,  Radish, and Tomato.
""")

# Load models
model = tf.keras.models.load_model('model_base_InceptionV3.h5')

# Create Classification
class_map= {0: 'Bean', 
            1: 'Bitter_Gourd', 
            2: 'Bottle_Gourd', 
            3: 'Brinjal', 
            4: 'Broccoli', 
            5: 'Cabbage', 
            6: 'Capsicum', 
            7: 'Carrot', 
            8: 'Cauliflower', 
            9: 'Cucumber', 
            10: 'Papaya', 
            11: 'Potato', 
            12: 'Pumpkin', 
            13: 'Radish', 
            14: 'Tomato'}

# load file
uploaded_file = st.file_uploader("Choose a image file", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns(2)

with col1:
    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(opencv_image,(180,180))

        # Now do something with the image! For example, let's display it:
        st.image(opencv_image, channels="RGB")
        resized = mobilenet_v2_preprocess_input(resized)
        img_reshape = resized[np.newaxis,...]

with col2:
    Genrate_pred = st.button("Classify Image")
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.write("Predicted Label for the image is {}".format(class_map [prediction]))
