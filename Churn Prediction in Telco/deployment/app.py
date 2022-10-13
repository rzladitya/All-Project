# model deployment using all in one method

import streamlit as st
import pickle
import pandas as pd
import numpy as np 
import tensorflow as tf
from tensorflow import keras


# --- SET PAGES ---
st.set_page_config(page_title="Web Application", page_icon=None, layout="centered")

# Load preprocessor
with open("preprocessor.pkl", "rb") as proses_file:
    preprocessor = pickle.load(proses_file)

# Load models
model = tf.keras.models.load_model('model_functional.h5')

columns = ['tenure', 'MonthlyCharges', 'TotalCharges', 'OnlineSecurity', 'TechSupport', 'Contract', 'PaperlessBilling']
label = ['No', 'Yes']

# Set Title
st.write('----------------------------------------------------------------')
st.title("Web Aplikasi Pengecekan Churn")
st.write('----------------------------------------------------------------')

left_column, right_column = st.columns(2)
with left_column:
    tenure =st.slider("Tenure" '(1-72 months)', min_value=1, max_value=72)
    TechSupport = st.selectbox("Tech Support", ['Yes', 'No', 'No Service'])
    OnlineSecurity = st.selectbox("Online Security", ['Yes', 'No', 'No Service'])
with right_column:
    MonthlyCharges =st.slider('Monthly Charges', min_value=18, max_value=120)
    Contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    PaperlessBilling = st.selectbox("Paperless Billing", ['Yes', 'No'])
TotalCharges = st.slider('Total Charges', min_value=18, max_value=9000)

# sex = st.text_input("Gender")

# inference
new_data = [tenure, MonthlyCharges, TotalCharges, OnlineSecurity, TechSupport, Contract, PaperlessBilling]
new_data = pd.DataFrame([new_data], columns=columns)

# Add data inference to Pipeline
data_inf_pipe = preprocessor.fit_transform(new_data)

# Predict Model Inference
data_process = model.predict(data_inf_pipe)
predict_grid = []
for element in data_process:
    if element > 0.5:
        predict_grid.append(1)
    else:
        predict_grid.append(0)

st.text("Apakah pelanggan akan berhenti berlangganan?")
if st.button("Predict"):
    st.success(label[predict_grid[0]])