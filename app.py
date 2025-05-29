import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. Load the dataset
df = pd.read_csv('DataChurn.csv')  # Adjust filename/path if needed

# 2. Initial data checks
print("Missing values per column:\n", df.isnull().sum())
print("Data types before cleaning:\n", df.dtypes)

# 3. Convert 'TotalCharges' to numeric, coerce errors (invalid entries become NaN)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

print("Missing values in 'TotalCharges':", df['TotalCharges'].isnull().sum())

missing_rows = df[df['TotalCharges'].isnull()]
print("Rows with missing TotalCharges:\n", missing_rows[['customerID', 'tenure', 'TotalCharges']])

# Drop rows with missing 'TotalCharges'
df = df.dropna(subset=['TotalCharges'])

# 4. Check for duplicates and drop customerID column
print("Number of duplicate customerIDs:", df['customerID'].duplicated().sum())
df = df.drop(columns=['customerID'])

# 5. Normalize binary columns to lowercase
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].str.lower()

# 6. Simplify 'No internet service' and 'No phone service' entries
cols_to_simplify = [
    'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
]
for col in cols_to_simplify:
    df[col] = df[col].replace({'No internet service': 'No', 'No phone service': 'No'})

print("Final missing values:\n", df.isnull().sum())
print("Data types after cleaning:\n", df.dtypes)
print("Cleaned dataset shape:", df.shape)

# 7. Encode binary columns
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0, 'female': 0, 'male': 1})

# 8. One-hot encode categorical columns
categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                    'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

df = pd.get_dummies(df, columns=categorical_cols)

# 9. Feature engineering: tenure group
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

# 10. Normalize numerical features
scaler = MinMaxScaler()
df[['MonthlyCharges', 'TotalCharges', 'tenure']] = scaler.fit_transform(
    df[['MonthlyCharges', 'TotalCharges', 'tenure']]
)

print("Transformed dataset shape:", df.shape)
print("Sample data:\n", df.head())

# You can save the cleaned data or proceed to model training here
# e.g., df.to_csv('cleaned_churn.csv', index=False)
