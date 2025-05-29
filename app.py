import streamlit as st
import pandas as pd
import joblib  # if you have a saved model
from sklearn.preprocessing import MinMaxScaler

st.title("Customer Churn Prediction")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df.head())
    
# 1. Check for missing values
print("Missing values per column:\n", df.isnull().sum())

# Check 'TotalCharges' column which is object type
print("Data types before cleaning:\n", df.dtypes)

# 2. Convert 'TotalCharges' to numeric, coerce errors (invalid entries become NaN)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Count how many NaNs were introduced
print("Missing values in 'TotalCharges':", df['TotalCharges'].isnull().sum())

# View rows with missing 'TotalCharges'
missing_rows = df[df['TotalCharges'].isnull()]
print(missing_rows[['customerID', 'tenure', 'TotalCharges']])

# Drop rows with missing 'TotalCharges' (mostly tenure == 0)
df = df.dropna(subset=['TotalCharges'])

# 3. Check for duplicates using 'customerID'
print("Number of duplicate customerIDs:", df['customerID'].duplicated().sum())

# 4. Drop customerID (if not needed in modeling)
df = df.drop(columns=['customerID'])

# 5. Convert binary categorical columns to consistent lower-case
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].str.lower()

# 6. Optional: Simplify "No internet service" and "No phone service" to "No"
cols_to_simplify = [
    'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
]
for col in cols_to_simplify:
    df[col] = df[col].replace({'No internet service': 'No', 'No phone service': 'No'})

# 7. Final check for missing values
print("Final missing values:\n", df.isnull().sum())

# 8. Data types after cleaning
print("Data types after cleaning:\n", df.dtypes)

# 9. Shape of cleaned data
print("Cleaned dataset shape:", df.shape)

print("\n")

# Assume df is the cleaned DataFrame from previous step

# 1. Encode binary columns
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 
               'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0, 'female': 0, 'male': 1})

# 2. One-hot encode categorical columns
categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                    'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

df = pd.get_dummies(df, columns=categorical_cols)

# 3. Feature Engineering: Tenure Group
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

# 4. Normalize numerical features
scaler = MinMaxScaler()
df[['MonthlyCharges', 'TotalCharges', 'tenure']] = scaler.fit_transform(
    df[['MonthlyCharges', 'TotalCharges', 'tenure']]
)

# 5. Confirm final shape
print("Transformed dataset shape:", df.shape)
print("Sample data:\n", df.head())
