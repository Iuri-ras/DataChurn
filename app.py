import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

st.title("Customer Churn Data Cleaning & Preprocessing")

uploaded_file = st.file_uploader("Upload your churn dataset CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the uploaded CSV file into DataFrame
    df = pd.read_csv(uploaded_file)
    
    st.write("### Raw Data Preview")
    st.dataframe(df.head())
    
    # 1. Check for missing values
    st.write("### Missing values per column:")
    st.write(df.isnull().sum())
    
    # 2. Convert 'TotalCharges' to numeric, coerce errors (invalid entries become NaN)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    st.write("Missing values in 'TotalCharges':", df['TotalCharges'].isnull().sum())
    
    # Show rows with missing TotalCharges
    missing_rows = df[df['TotalCharges'].isnull()]
    st.write("Rows with missing TotalCharges (usually tenure=0):")
    st.write(missing_rows[['customerID', 'tenure', 'TotalCharges']])
    
    # Drop rows with missing TotalCharges
    df = df.dropna(subset=['TotalCharges'])
    
    # 3. Check for duplicates and drop customerID
    st.write("Number of duplicate customerIDs:", df['customerID'].duplicated().sum())
    df = df.drop(columns=['customerID'])
    
    # 4. Normalize binary columns to lowercase
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        df[col] = df[col].str.lower()
    
    # 5. Simplify 'No internet service' and 'No phone service'
    cols_to_simplify = [
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    for col in cols_to_simplify:
        df[col] = df[col].replace({'No internet service': 'No', 'No phone service': 'No'})
    
    st.write("### Missing values after cleaning:")
    st.write(df.isnull().sum())
    
    st.write("### Data types after cleaning:")
    st.write(df.dtypes)
    
    st.write("Cleaned dataset shape:", df.shape)
    
    # 6. Encode binary columns
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0, 'female': 0, 'male': 1})
    
    # 7. One-hot encode categorical columns
    categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                        'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                        'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=categorical_cols)
    
    # 8. Feature engineering: tenure group
    def tenure_group(tenure):
        if tenure <= 12:
            return '0-1 year'
        elif tenure <= 24:
            return '1-2 years'
        elif tenure <= 48:
            return '2-4 years'
        elif tenure <= 60:
            return '4-5 years'
        else:
            return '5+ years'
    
    df['tenure_group'] = df['tenure'].apply(tenure_group)
    df = pd.get_dummies(df, columns=['tenure_group'])
    
    # 9. Normalize numerical features
    scaler = MinMaxScaler()
    df[['MonthlyCharges', 'TotalCharges', 'tenure']] = scaler.fit_transform(
        df[['MonthlyCharges', 'TotalCharges', 'tenure']]
    )
    
    st.write("### Transformed dataset shape:", df.shape)
    st.write("### Sample transformed data:")
    st.dataframe(df.head())
    
else:
    st.info("Please upload a CSV file to get started.")
