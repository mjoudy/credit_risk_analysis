# database/generate_db.py

import pandas as pd
import sqlite3
import os

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
txt_path = os.path.join("..", "data", "german.data")
df = pd.read_csv(txt_path, sep=' ', header=None, names=columns)

# Convert target to binary (Good=1, Bad=0)
df["Credit_risk"] = df["Credit_risk"].apply(lambda x: 1 if x == 1 else 0)

# Create SQLite DB
os.makedirs("database", exist_ok=True)
conn = sqlite3.connect("database/german_credit.db")
df.to_sql("credit_data", conn, if_exists="replace", index=False)

# Preview a few rows
sample = pd.read_sql("SELECT * FROM credit_data LIMIT 5;", conn)
print(sample)

conn.close()
print("✅ Database created successfully at: database/german_credit.db")
