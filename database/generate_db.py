# database/generate_db.py

import pandas as pd
import sqlite3
import os
from sklearn.preprocessing import StandardScaler

# --- Cleaned data creation (from EDA notebook) ---
def create_cleaned_credit_data(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM credit_data;", conn)
    # One-hot encode categorical variables
    categorical_cols = [
        "Status_of_existing_checking_account", "Credit_history", "Purpose",
        "Savings_account_bonds", "Present_employment_since",
        "Personal_status_and_sex", "Other_debtors_guarantors", "Property",
        "Other_installment_plans", "Housing", "Job", "Telephone", "foreign_worker"
    ]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    # Scale continuous numeric features
    scaler = StandardScaler()
    numerical_cols = [
        "Duration_in_month", "Credit_amount", "Installment_rate_in_percentage_of_disposable_income",
        "Present_residence_since", "Age_in_years", "Number_of_existing_credits_at_this_bank",
        "Number_of_people_being_liable_to_provide_maintenance_for"
    ]
    df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
    # Optional: reorder columns so target is at the end
    target = df_encoded.pop("Credit_risk")
    df_encoded["Credit_risk"] = target
    # Save cleaned version into SQLite
    df_encoded.to_sql("cleaned_credit_data", conn, if_exists="replace", index=False)
    print("✅ Cleaned data saved to 'cleaned_credit_data' table.")
    print("Shape:", df_encoded.shape)
    conn.close()

# --- Robust path construction ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "german.data")
DB_PATH = os.path.join(PROJECT_ROOT, "database", "german_credit.db")
# --- End robust path ---

# Column names based on UCI documentation
columns = [
    "Status_of_existing_checking_account",
    "Duration_in_month",
    "Credit_history",
    "Purpose",
    "Credit_amount",
    "Savings_account_bonds",
    "Present_employment_since",
    "Installment_rate_in_percentage_of_disposable_income",
    "Personal_status_and_sex",
    "Other_debtors_guarantors",
    "Present_residence_since",
    "Property",
    "Age_in_years",
    "Other_installment_plans",
    "Housing",
    "Number_of_existing_credits_at_this_bank",
    "Job",
    "Number_of_people_being_liable_to_provide_maintenance_for",
    "Telephone",
    "foreign_worker",
    "Credit_risk"  # Target: 1 = Good, 2 = Bad
]

# Load data
try:
    df = pd.read_csv(DATA_PATH, sep=' ', header=None, names=columns)
except Exception as e:
    print(f"Failed to load data file: {e}")
    exit(1)

# Convert target to binary (Good=1, Bad=0)
df["Credit_risk"] = df["Credit_risk"].apply(lambda x: 1 if x == 1 else 0)

# Create SQLite DB
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
conn = sqlite3.connect(DB_PATH)
df.to_sql("credit_data", conn, if_exists="replace", index=False)

# Preview a few rows
sample = pd.read_sql("SELECT * FROM credit_data LIMIT 5;", conn)
print(sample)

# After creating credit_data table:
create_cleaned_credit_data(DB_PATH)

conn.close()
print(f"\u2705 Database created successfully at: {DB_PATH}")
