import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

st.title("Customer Churn Prediction: Data Prep, EDA, and Modeling")

# Load dataset (make sure DataChurn.csv is in the same folder)
df = pd.read_csv('DataChurn.csv')

# Show raw data preview
st.write("### Raw Data Preview")
st.dataframe(df.head())

# Data cleaning and preprocessing
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

# Use raw data for clear EDA plots with original scales
raw_df = pd.read_csv('DataChurn.csv')
raw_df['TotalCharges'] = pd.to_numeric(raw_df['TotalCharges'], errors='coerce')
raw_df = raw_df.dropna(subset=['TotalCharges'])
raw_df['Churn'] = raw_df['Churn'].str.lower().map({'yes': 1, 'no': 0})

# 1. Histogram of tenure by churn
st.subheader("1. Customer Tenure Distribution by Churn Status")
fig1, ax1 = plt.subplots(figsize=(8,5))
sns.histplot(data=raw_df, x='tenure', hue='Churn', multiple='stack', bins=30, ax=ax1)
ax1.set_xlabel('Tenure (months)')
ax1.set_ylabel('Number of Customers')
st.pyplot(fig1)

# 2. Bar plot of churn rate by contract type
st.subheader("2. Churn Rate by Contract Type")
contract_churn = raw_df.groupby('Contract')['Churn'].mean().reset_index()
fig2, ax2 = plt.subplots(figsize=(6,4))
sns.barplot(x='Contract', y='Churn', data=contract_churn, ax=ax2)
ax2.set_ylabel('Churn Rate')
st.pyplot(fig2)

# 3. Correlation heatmap
st.subheader("3. Correlation Heatmap of Numeric Features")
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
corr = raw_df[numeric_cols].corr()
fig3, ax3 = plt.subplots(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax3)
st.pyplot(fig3)

st.write("---")
st.header("Key Insights")

st.markdown("""
- **Tenure Distribution:** Customers who churn generally have lower tenure, indicating many leave within the first 1-2 years.
- **Contract Type:** Month-to-month customers have a significantly higher churn rate compared to customers on longer contracts, showing contract length helps retention.
- **Correlations:** Churn is negatively correlated with tenure and positively correlated with MonthlyCharges, meaning customers who stay longer churn less, and higher monthly charges might increase churn risk.
""")

# === MODEL BUILDING AND EVALUATION ===
st.write("---")
st.header("Model Building and Evaluation")

# Prepare features and target
X = df.drop(columns=['Churn'])
y = df['Churn']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize Random Forest model
model = RandomForestClassifier(random_state=42)

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
st.write(f"5-fold Cross-Validation F1 Scores: {cv_scores}")
st.write(f"Mean CV F1 Score: {cv_scores.mean():.4f}")

# Train on full training set
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Performance metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

st.write("### Model Performance on Test Set")
st.write(f"Accuracy: {acc:.4f}")
st.write(f"Precision: {prec:.4f}")
st.write(f"Recall: {rec:.4f}")
st.write(f"F1 Score: {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn', 'Churn'])

fig_cm, ax_cm = plt.subplots(figsize=(6,6))
disp.plot(ax=ax_cm)
st.pyplot(fig_cm)
