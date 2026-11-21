
import os
import pandas as pd
import numpy as np
from google.colab import files

# Project data folder
os.makedirs("data", exist_ok=True)

# Public LendingClub dataset
url = "https://digintu.tech/tmp/cs109a/loan_accepted_10.csv"

print("Downloading loan dataset...")

df = pd.read_csv(url, low_memory=False)

print("Dataset downloaded successfully!")
print("Records:", len(df))
print("Columns:", len(df.columns))

# Save original downloaded dataset
df.to_csv(
    "data/lendingclub_raw.csv",
    index=False
)

# Select fields required for the project
selected_columns = [
    "id",
    "loan_amnt",
    "funded_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "issue_d",
    "loan_status",
    "purpose",
    "addr_state",
    "annual_inc",
    "dti",
    "home_ownership",
    "emp_length",
    "out_prncp",
    "total_pymnt",
    "total_rec_prncp",
    "recoveries",
    "last_pymnt_d",
    "delinq_2yrs",
    "verification_status"
]

available_columns = [
    col for col in selected_columns
    if col in df.columns
]

loan_data = df[available_columns].copy()

# Save project dataset
loan_data.to_csv(
    "data/lendingclub_project_data.csv",
    index=False
)

# Create a quality report
quality_report = pd.DataFrame({
    "Column": loan_data.columns,
    "Data Type": loan_data.dtypes.astype(str).values,
    "Missing Values": loan_data.isnull().sum().values,
    "Missing Percentage": (
        loan_data.isnull().mean().values * 100
    ).round(2)
})

quality_report.to_csv(
    "data/data_quality_report.csv",
    index=False
)

# Create dataset documentation
dictionary = {
    "id": "Unique loan identification number",
    "loan_amnt": "Loan amount requested by borrower",
    "funded_amnt": "Amount funded for the loan",
    "term": "Loan repayment term",
    "int_rate": "Annual loan interest rate",
    "installment": "Scheduled monthly installment",
    "grade": "LendingClub credit grade",
    "sub_grade": "Detailed credit sub-grade",
    "issue_d": "Loan origination month",
    "loan_status": "Loan repayment status",
    "purpose": "Purpose of the loan",
    "addr_state": "Borrower's state",
    "annual_inc": "Borrower's annual income",
    "dti": "Debt-to-income ratio",
    "home_ownership": "Borrower's home ownership status",
    "emp_length": "Borrower's employment length",
    "out_prncp": "Outstanding principal balance",
    "total_pymnt": "Total payments received",
    "total_rec_prncp": "Principal repaid",
    "recoveries": "Amount recovered after charge-off",
    "last_pymnt_d": "Date of last payment",
    "delinq_2yrs": "Reported delinquencies in previous two years",
    "verification_status": "Income verification status"
}

dictionary_df = pd.DataFrame(
    dictionary.items(),
    columns=["Column", "Description"]
)

dictionary_df.to_csv(
    "data/data_dictionary.csv",
    index=False
)

print("\nDATA ASSETS CREATED")

for file in os.listdir("data"):
    print(file)

# Download project-ready dataset
files.download("data/lendingclub_project_data.csv")
files.download("data/data_quality_report.csv")
files.download("data/data_dictionary.csv")
