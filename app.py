import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# load the encoders and scaler
# load he encoder and scaler
with open('one_hot_encoder_geo.pkl', 'rb') as file:
    one_hot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:

    label_encoder_gender = pickle.load(file)

## streamlit app
st.title("Customer Churn Prediction")

 # User input
geography = st.selectbox("Geography", one_hot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.number_input("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.number_input("Tenure")
num_of_products = st.number_input("Number of Products", 1,4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Prepare the input data
# First, encode the gender using the label encoder
gender_encoded = label_encoder_gender.transform([gender])[0]

input_data = pd.DataFrame({
    # 'Geography': [geography],
    'Gender': [gender_encoded],
    'Age': [age],
    'Balance': [balance],
    'CreditScore': [credit_score],
    'EstimatedSalary': [estimated_salary],
    'Tenure': [tenure],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member]
})

# one-hot encode ''Geography
geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

# combine the one hot encoded columns with the original data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data - convert to numpy array to avoid feature name mismatch
# The scaler expects a numpy array, not a DataFrame with feature names
input_data_scaled = scaler.transform(input_data.values)

# prediction churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

if prediction_proba > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")

