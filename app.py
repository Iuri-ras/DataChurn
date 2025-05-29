import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Customer Churn Data Cleaning, Preprocessing & EDA")

uploaded_file = st.file_uploader("Upload your churn dataset CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Data Preview")
    st.dataframe(df.head())

    # Data cleaning and preprocessing (same as before)
    st.write("### Data Cleaning and Preprocessing")

    st.write("Missing values per column:")
    st.write(df.isnull().sum())

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    st.write("Missing values in 'TotalCharges':", df['TotalCharges'].isnull().sum())

    missing_rows = df[df['TotalCharges'].isnull()]
    st.write("Rows with missing TotalCharges:")
    st.write(missing_rows[['customerID', 'tenure', 'TotalCharges']])

    df = df.dropna(subset=['TotalCharges'])

    st.write("Number of duplicate customerIDs:", df['customerID'].duplicated().sum())
    df = df.drop(columns=['customerID'])

    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        df[col] = df[col].str.lower()

    cols_to_simplify = [
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    for col in cols_to_simplify:
        df[col] = df[col].replace({'No internet service': 'No', 'No phone service': 'No'})

    st.write("Missing values after cleaning:")
    st.write(df.isnull().sum())

    st.write("Data types after cleaning:")
    st.write(df.dtypes)

    st.write("Cleaned dataset shape:", df.shape)

    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0, 'female': 0, 'male': 1})

    categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                        'OnlineBackup', 'DeviceProtection', 'TechSupport',
                        'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

    df = pd.get_dummies(df, columns=categorical_cols)

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

    scaler = MinMaxScaler()
    df[['MonthlyCharges', 'TotalCharges', 'tenure']] = scaler.fit_transform(
        df[['MonthlyCharges', 'TotalCharges', 'tenure']]
    )

    st.write("Transformed dataset shape:", df.shape)
    st.write("Sample transformed data:")
    st.dataframe(df.head())

    # === EDA VISUALIZATIONS ===
    st.write("---")
    st.header("Exploratory Data Analysis (EDA)")

    # Use original df with churn encoded as 1/0 for EDA plots
    # Reload and clean churn for easier EDA plotting (using initial df with Churn encoded)
    eda_df = pd.read_csv(uploaded_file)
    eda_df['TotalCharges'] = pd.to_numeric(eda_df['TotalCharges'], errors='coerce')
    eda_df = eda_df.dropna(subset=['TotalCharges'])
    eda_df['Churn'] = eda_df['Churn'].str.lower().map({'yes': 1, 'no': 0})

    # 1. Histogram of tenure by churn
    st.subheader("1. Customer Tenure Distribution by Churn Status")
    fig1, ax1 = plt.subplots(figsize=(8,5))
    sns.histplot(data=eda_df, x='tenure', hue='Churn', multiple='stack', bins=30, ax=ax1)
    ax1.set_xlabel('Tenure (months)')
    ax1.set_ylabel('Number of Customers')
    st.pyplot(fig1)

    # 2. Bar plot of churn rate by contract type
    st.subheader("2. Churn Rate by Contract Type")
    contract_churn = eda_df.groupby('Contract')['Churn'].mean().reset_index()
    fig2, ax2 = plt.subplots(figsize=(6,4))
    sns.barplot(x='Contract', y='Churn', data=contract_churn, ax=ax2)
    ax2.set_ylabel('Churn Rate')
    st.pyplot(fig2)

    # 3. Correlation heatmap
    st.subheader("3. Correlation Heatmap of Numeric Features")
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
    corr = eda_df[numeric_cols].corr()
    fig3, ax3 = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax3)
    st.pyplot(fig3)

    # === INSIGHTS ===
    st.write("---")
    st.header("Key Insights")

    st.markdown("""
    - **Tenure Distribution:** Customers who churn generally have lower tenure, indicating many leave within the first 1-2 years.
    - **Contract Type:** Month-to-month customers have a significantly higher churn rate compared to customers on longer contracts, showing contract length helps retention.
    - **Correlations:** Churn is negatively correlated with tenure and positively correlated with MonthlyCharges, meaning customers who stay longer churn less, and higher monthly charges might increase churn risk.
    """)

else:
    st.info("Please upload a CSV file to get started.")
