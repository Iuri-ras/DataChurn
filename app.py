import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

st.title("Customer Churn Prediction: Data Prep, EDA, Modeling & Deployment")

# Load & preprocess dataset once (same as before)
df = pd.read_csv('DataChurn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])
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

X = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn', 'Churn'])

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Sidebar navigation
page = st.sidebar.selectbox("Navigate", ["Overview & EDA", "Model Results", "Predict Churn"])

if page == "Overview & EDA":
    st.header("1. Data Overview & EDA")

    st.write("### Cleaned & Preprocessed Data Sample")
    st.dataframe(df.head())

    raw_df = pd.read_csv('DataChurn.csv')
    raw_df['TotalCharges'] = pd.to_numeric(raw_df['TotalCharges'], errors='coerce')
    raw_df = raw_df.dropna(subset=['TotalCharges'])
    raw_df['Churn'] = raw_df['Churn'].str.lower().map({'yes': 1, 'no': 0})

    st.subheader("Customer Tenure Distribution by Churn Status")
    fig1, ax1 = plt.subplots(figsize=(8,5))
    sns.histplot(data=raw_df, x='tenure', hue='Churn', multiple='stack', bins=30, ax=ax1)
    ax1.set_xlabel('Tenure (months)')
    ax1.set_ylabel('Number of Customers')
    st.pyplot(fig1)

    st.subheader("Churn Rate by Contract Type")
    contract_churn = raw_df.groupby('Contract')['Churn'].mean().reset_index()
    fig2, ax2 = plt.subplots(figsize=(6,4))
    sns.barplot(x='Contract', y='Churn', data=contract_churn, ax=ax2)
    ax2.set_ylabel('Churn Rate')
    st.pyplot(fig2)

    st.subheader("Correlation Heatmap")
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
    corr = raw_df[numeric_cols].corr()
    fig3, ax3 = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax3)
    st.pyplot(fig3)

elif page == "Model Results":
    st.header("2. Model Training & Evaluation")

    st.write(f"5-fold Cross-Validation F1 Scores: {cv_scores}")
    st.write(f"Mean Cross-Validation F1 Score: {cv_scores.mean():.4f}")

    st.subheader("Test Set Performance Metrics")
    st.write(f"- Accuracy: {acc:.4f}")
    st.write(f"- Precision: {prec:.4f}")
    st.write(f"- Recall: {rec:.4f}")
    st.write(f"- F1 Score: {f1:.4f}")

    fig_cm, ax_cm = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax_cm)
    st.pyplot(fig_cm)
    st.write("Confusion matrix shows true vs. predicted labels.")

    st.header("Feature Importance")
    fig_feat, ax_feat = plt.subplots(figsize=(10,6))
    sns.barplot(x=importances[indices], y=X.columns[indices], ax=ax_feat)
    ax_feat.set_title("Feature Importance")
    ax_feat.set_xlabel("Importance")
    ax_feat.set_ylabel("Feature")
    st.pyplot(fig_feat)

elif page == "Predict Churn":
    st.header("3. Predict Customer Churn")

    st.markdown("Fill out the form below with customer data and click 'Predict'.")

    with st.form("prediction_form"):
        gender = st.selectbox("Gender", ["Female", "Male"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        PaymentMethod = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
        TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

        submit = st.form_submit_button("Predict")

    if submit:
        input_dict = {
            'gender': 1 if gender.lower() == 'male' else 0,
            'SeniorCitizen': SeniorCitizen,
            'Partner': 1 if Partner.lower() == 'yes' else 0,
            'Dependents': 1 if Dependents.lower() == 'yes' else 0,
            'tenure': tenure,
            'PhoneService': 1 if PhoneService.lower() == 'yes' else 0,
            'MultipleLines': MultipleLines,
            'InternetService': InternetService,
            'OnlineSecurity': OnlineSecurity,
            'OnlineBackup': OnlineBackup,
            'DeviceProtection': DeviceProtection,
            'TechSupport': TechSupport,
            'StreamingTV': StreamingTV,
            'StreamingMovies': StreamingMovies,
            'Contract': Contract,
            'PaperlessBilling': 1 if PaperlessBilling.lower() == 'yes' else 0,
            'PaymentMethod': PaymentMethod,
            'MonthlyCharges': MonthlyCharges,
            'TotalCharges': TotalCharges,
        }

        input_df = pd.DataFrame([input_dict])

        # Simplify no internet/phone service values
        cols_to_simplify = [
            'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        for col in cols_to_simplify:
            input_df[col] = input_df[col].replace({'No internet service': 'No', 'No phone service': 'No'})

        # One-hot encode categorical variables
        input_df = pd.get_dummies(input_df, columns=categorical_cols)

        # Add missing columns
        missing_cols = set(X.columns) - set(input_df.columns)
        for c in missing_cols:
            input_df[c] = 0

        # Reorder columns to match training data
        input_df = input_df[X.columns]

        # Normalize numeric columns
        for col in ['MonthlyCharges', 'TotalCharges', 'tenure']:
            idx = list(scaler.feature_names_in_).index(col)
            min_val = scaler.data_min_[idx]
            max_val = scaler.data_max_[idx]
            input_df[col] = (input_df[col] - min_val) / (max_val - min_val)

        pred_prob = model.predict_proba(input_df)[0][1]
        pred_class = model.predict(input_df)[0]

        st.subheader("Prediction Results")
        st.write(f"Churn Probability: {pred_prob:.2%}")
        st.write(f"Predicted Class: {'Churn' if pred_class == 1 else 'No Churn'}")
