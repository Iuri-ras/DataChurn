import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

st.title("Customer Churn Prediction: Data Preparation, EDA & Modeling")

st.markdown("""
This app performs data cleaning, exploratory data analysis (EDA), and builds a Random Forest model to predict customer churn.
Upload your dataset as 'DataChurn.csv' in the same directory and run this app.
""")

# Load dataset
df = pd.read_csv('DataChurn.csv')

# --- Raw Data Preview ---
st.header("1. Raw Data Preview")
st.write("Here's a sample of the original dataset before any cleaning or preprocessing:")
st.dataframe(df.head())

# --- Data Cleaning and Preprocessing ---
st.header("2. Data Cleaning & Preprocessing")

st.write("**Checking missing values in each column:**")
st.write(df.isnull().sum())

st.write("Converting 'TotalCharges' to numeric. Invalid parsing will be coerced to NaN.")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
st.write(f"Missing values in 'TotalCharges' after conversion: {df['TotalCharges'].isnull().sum()}")

st.write("Rows with missing 'TotalCharges' (typically customers with tenure = 0):")
missing_rows = df[df['TotalCharges'].isnull()]
st.dataframe(missing_rows[['customerID', 'tenure', 'TotalCharges']])

st.write("Dropping rows with missing 'TotalCharges' to clean data.")
df = df.dropna(subset=['TotalCharges'])

st.write(f"Duplicate customer IDs found: {df['customerID'].duplicated().sum()}. Dropping 'customerID' as it's not needed for modeling.")
df = df.drop(columns=['customerID'])

st.write("Converting binary categorical columns to lowercase for consistency.")
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].str.lower()

st.write("Simplifying categories 'No internet service' and 'No phone service' to 'No' in multiple columns.")
cols_to_simplify = [
    'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
]
for col in cols_to_simplify:
    df[col] = df[col].replace({'No internet service': 'No', 'No phone service': 'No'})

st.write("Final check for missing values after cleaning:")
st.write(df.isnull().sum())

st.write("Data types after cleaning:")
st.write(df.dtypes)

st.write(f"Cleaned dataset shape: {df.shape}")

st.write("Encoding binary columns to numeric (yes=1, no=0, female=0, male=1).")
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0, 'female': 0, 'male': 1})

st.write("Applying one-hot encoding to categorical columns.")
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

st.write("Creating a new feature 'tenure_group' by grouping tenure into categories.")
df['tenure_group'] = df['tenure'].apply(tenure_group)
df = pd.get_dummies(df, columns=['tenure_group'])

st.write("Normalizing numeric features 'MonthlyCharges', 'TotalCharges', and 'tenure' with Min-Max scaling.")
scaler = MinMaxScaler()
df[['MonthlyCharges', 'TotalCharges', 'tenure']] = scaler.fit_transform(
    df[['MonthlyCharges', 'TotalCharges', 'tenure']]
)

st.write(f"Transformed dataset shape: {df.shape}")
st.write("Sample of the transformed data:")
st.dataframe(df.head())

# --- Exploratory Data Analysis ---
st.header("3. Exploratory Data Analysis (EDA)")

st.markdown("""
The following visualizations help us understand the distribution of key variables and their relationships with churn.
""")

# Prepare raw data for EDA (original scales for easier interpretation)
raw_df = pd.read_csv('DataChurn.csv')
raw_df['TotalCharges'] = pd.to_numeric(raw_df['TotalCharges'], errors='coerce')
raw_df = raw_df.dropna(subset=['TotalCharges'])
raw_df['Churn'] = raw_df['Churn'].str.lower().map({'yes': 1, 'no': 0})

st.subheader("3.1 Customer Tenure Distribution by Churn Status")
fig1, ax1 = plt.subplots(figsize=(8,5))
sns.histplot(data=raw_df, x='tenure', hue='Churn', multiple='stack', bins=30, ax=ax1)
ax1.set_xlabel('Tenure (months)')
ax1.set_ylabel('Number of Customers')
st.pyplot(fig1)
st.write("**Insight:** Customers who churn tend to have shorter tenures, many leaving within the first 1-2 years.")

st.subheader("3.2 Churn Rate by Contract Type")
contract_churn = raw_df.groupby('Contract')['Churn'].mean().reset_index()
fig2, ax2 = plt.subplots(figsize=(6,4))
sns.barplot(x='Contract', y='Churn', data=contract_churn, ax=ax2)
ax2.set_ylabel('Churn Rate')
st.pyplot(fig2)
st.write("**Insight:** Month-to-month contracts show higher churn rates than longer-term contracts, indicating contract length improves retention.")

st.subheader("3.3 Correlation Heatmap")
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
corr = raw_df[numeric_cols].corr()
fig3, ax3 = plt.subplots(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax3)
st.pyplot(fig3)
st.write("**Insight:** Churn correlates negatively with tenure and positively with MonthlyCharges, meaning longer-tenured customers churn less, while higher bills may increase churn risk.")

# --- Model Building and Evaluation ---
st.header("4. Model Building and Evaluation")

st.markdown("""
We train a **Random Forest Classifier** to predict churn.  
The model is evaluated using 5-fold cross-validation and tested on a held-out test set.  
Key performance metrics and a confusion matrix are presented below.
""")

# Prepare features and target
X = df.drop(columns=['Churn'])
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize model
model = RandomForestClassifier(random_state=42)

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
st.write(f"5-fold Cross-Validation F1 Scores: {cv_scores}")
st.write(f"Mean Cross-Validation F1 Score: {cv_scores.mean():.4f}")

# Train on full train set
model.fit(X_train, y_train)

# Predict test set
y_pred = model.predict(X_test)

# Calculate metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

st.subheader("Test Set Performance")
st.write(f"- Accuracy: {acc:.4f}")
st.write(f"- Precision: {prec:.4f}")
st.write(f"- Recall: {rec:.4f}")
st.write(f"- F1 Score: {f1:.4f}")

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn', 'Churn'])
fig_cm, ax_cm = plt.subplots(figsize=(6,6))
disp.plot(ax=ax_cm)
st.pyplot(fig_cm)
st.write("**Confusion Matrix:** Shows true vs. predicted labels to understand errors.")

