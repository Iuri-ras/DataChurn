import streamlit as st
import pandas as pd
import joblib  # if you have a saved model

st.title("Customer Churn Prediction")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df.head())
    
    # Here you can add your model prediction logic and display output
    # e.g., load your trained model and predict churn
