
!pip -q install nbformat

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def markdown(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

markdown("""
# Project 12: Bank Loan Portfolio & Credit Risk Monitoring

**Domain:** Banking and Financial Services  
**Language:** Python  
**Libraries:** Pandas, NumPy, Matplotlib, Seaborn

## Business Objective

Analyze a historical loan portfolio to monitor:
- Portfolio composition
- Loan repayment performance
- Delinquency patterns
- Vintage performance
- Credit concentration
- Potential credit losses under stress scenarios

**Dataset:** Historical LendingClub loan data.
""")

markdown("## 1. Import Libraries")

code("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

print("Libraries imported successfully")
""")

markdown("## 2. Load Dataset")

code("""
from google.colab import files

try:
    df = pd.read_csv(
        "data/lendingclub_project_data.csv",
        low_memory=False
    )
except FileNotFoundError:
    print("Upload your LendingClub CSV file")
    uploaded = files.upload()
    filename = next(iter(uploaded))
    df = pd.read_csv(filename, low_memory=False)

print("Dataset shape:", df.shape)
display(df.head())
""")

markdown("## 3. Data Quality Audit")

code("""
audit_report = pd.DataFrame({
    "Metric": [
        "Total records",
        "Missing loan IDs",
        "Duplicate loan IDs",
        "Missing loan status",
        "Missing origination dates"
    ],
    "Count": [
        len(df),
        df["id"].isna().sum(),
        df["id"].duplicated().sum(),
        df["loan_status"].isna().sum(),
        df["issue_d"].isna().sum()
    ]
})

audit_report["Percentage"] = (
    audit_report["Count"] / len(df) * 100
)

display(audit_report)

df = df.dropna(subset=["id"]).copy()
df = df.drop_duplicates(subset=["id"]).copy()
""")

markdown("## 4. Data Cleaning")

code("""
numeric_columns = [
    "loan_amnt", "funded_amnt", "int_rate",
    "annual_inc", "dti", "out_prncp",
    "total_pymnt", "total_rec_prncp",
    "recoveries"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace("%", "", regex=False),
            errors="coerce"
        )

df["issue_d"] = pd.to_datetime(
    df["issue_d"],
    format="%b-%Y",
    errors="coerce"
)

df = df.dropna(subset=["issue_d"]).copy()

print("Cleaned records:", len(df))
""")

markdown("## 5. Balance Reconciliation")

code("""
balance_report = pd.DataFrame({
    "Metric": [
        "Original funded amount",
        "Principal repaid",
        "Outstanding principal"
    ],
    "Amount": [
        df["funded_amnt"].sum(),
        df["total_rec_prncp"].sum(),
        df["out_prncp"].sum()
    ]
})

display(balance_report)

print("Negative balances:", (df["out_prncp"] < 0).sum())
print("Missing balances:", df["out_prncp"].isna().sum())
""")

markdown("## 6. Feature Engineering")

code("""
status_mapping = {
    "Fully Paid": "Closed",
    "Current": "Current",
    "In Grace Period": "Grace Period",
    "Late (16-30 days)": "16-30 DPD",
    "Late (31-120 days)": "31-120 DPD",
    "Charged Off": "Charged Off",
    "Default": "Default"
}

df["risk_status"] = (
    df["loan_status"].map(status_mapping).fillna("Other")
)

df["delinquency_flag"] = df["risk_status"].isin([
    "Grace Period", "16-30 DPD", "31-120 DPD"
])

df["default_flag"] = df["risk_status"].isin([
    "Charged Off", "Default"
])

df["exposure"] = df["out_prncp"]
df["vintage_year"] = df["issue_d"].dt.year

df["term_months"] = pd.to_numeric(
    df["term"].astype(str).str.extract(r"(\\d+)")[0],
    errors="coerce"
)

display(df.head())
""")

markdown("## 7. Portfolio Overview")

code("""
total_loans = df["id"].nunique()
total_funded = df["funded_amnt"].sum()

valid_exposure = (
    df["exposure"].notna() &
    (df["exposure"] >= 0)
)

total_exposure = df.loc[valid_exposure, "exposure"].sum()

delinquent_exposure = df.loc[
    df["delinquency_flag"] & valid_exposure,
    "exposure"
].sum()

portfolio_summary = pd.DataFrame({
    "Metric": [
        "Total loans",
        "Total funded amount",
        "Outstanding exposure",
        "Delinquent exposure",
        "Exposure weighted delinquency rate",
        "Defaulted loan percentage"
    ],
    "Value": [
        total_loans,
        total_funded,
        total_exposure,
        delinquent_exposure,
        delinquent_exposure / total_exposure * 100
            if total_exposure > 0 else np.nan,
        df["default_flag"].mean() * 100
    ]
})

display(portfolio_summary.round(2))
""")

markdown("## 8. NumPy Weighted Interest Rate and Quantiles")

code("""
balance = df["exposure"].to_numpy(dtype=float)
rate = df["int_rate"].to_numpy(dtype=float)

valid = (
    np.isfinite(balance) &
    np.isfinite(rate) &
    (balance > 0) &
    (rate >= 0)
)

weighted_rate = (
    np.average(rate[valid], weights=balance[valid])
    if valid.any() else np.nan
)

valid_balance = balance[
    np.isfinite(balance) & (balance >= 0)
]

quantiles = np.quantile(
    valid_balance,
    [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
)

print("Weighted interest rate:", weighted_rate)

display(pd.DataFrame({
    "Percentile": ["25%", "50%", "75%", "90%", "95%", "99%"],
    "Exposure": quantiles
}))
""")

markdown("## 9. Delinquency Analysis")

code("""
delinquency_table = (
    df.groupby("risk_status")
    .agg(
        loan_count=("id", "nunique"),
        funded_amount=("funded_amnt", "sum"),
        outstanding=("exposure", "sum")
    )
    .reset_index()
)

display(delinquency_table)

sns.barplot(
    data=delinquency_table,
    x="risk_status",
    y="outstanding"
)

plt.title("Outstanding Exposure by Repayment Status")
plt.xticks(rotation=40)
plt.tight_layout()
plt.show()
""")

markdown("## 10. Vintage Analysis")

code("""
vintage_table = (
    df.groupby("vintage_year")
    .agg(
        loan_count=("id", "nunique"),
        funded_amount=("funded_amnt", "sum"),
        outstanding=("exposure", "sum"),
        delinquency_rate=("delinquency_flag", "mean"),
        default_rate=("default_flag", "mean")
    )
    .reset_index()
)

vintage_table["delinquency_rate"] *= 100
vintage_table["default_rate"] *= 100

display(vintage_table)

sns.barplot(
    data=vintage_table,
    x="vintage_year",
    y="default_rate"
)

plt.title("Observed Default Rate by Vintage")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=45)
plt.show()
""")

markdown("## 11. Segment Analysis")

code("""
term_analysis = df.groupby("term_months").agg(
    loan_count=("id", "nunique"),
    exposure=("exposure", "sum"),
    default_rate=("default_flag", "mean")
).reset_index()

grade_analysis = df.groupby("grade").agg(
    loan_count=("id", "nunique"),
    exposure=("exposure", "sum"),
    default_rate=("default_flag", "mean")
).reset_index()

purpose_analysis = df.groupby("purpose").agg(
    loan_count=("id", "nunique"),
    exposure=("exposure", "sum"),
    default_rate=("default_flag", "mean")
).reset_index()

for table in [term_analysis, grade_analysis, purpose_analysis]:
    table["default_rate"] *= 100

display(term_analysis)
display(grade_analysis)
display(purpose_analysis)
""")

markdown("## 12. Concentration Analysis")

code("""
def concentration_analysis(data, segment):

    grouped = data.groupby(
        segment, dropna=False
    )["exposure"].sum().dropna()

    total = grouped.sum()

    if total <= 0:
        return pd.DataFrame(), np.nan

    shares = grouped / total * 100

    result = shares.rename(
        "exposure_share"
    ).reset_index()

    hhi = (shares ** 2).sum()

    return result.sort_values(
        "exposure_share", ascending=False
    ), hhi


grade_concentration, grade_hhi = concentration_analysis(df, "grade")
purpose_concentration, purpose_hhi = concentration_analysis(df, "purpose")
state_concentration, state_hhi = concentration_analysis(df, "addr_state")

print("Grade HHI:", grade_hhi)
print("Purpose HHI:", purpose_hhi)
print("Geographic HHI:", state_hhi)

display(grade_concentration)
display(purpose_concentration)
display(state_concentration)
""")

markdown("## 13. Credit Loss Stress Testing")

code("""
portfolio_ead = total_exposure

scenarios = pd.DataFrame({
    "Scenario": ["Base", "Moderate Stress", "Severe Stress"],
    "PD": [0.03, 0.06, 0.10],
    "LGD": [0.40, 0.50, 0.65]
})

scenarios["EAD"] = portfolio_ead

scenarios["Estimated_Loss"] = (
    scenarios["PD"] *
    scenarios["LGD"] *
    scenarios["EAD"]
)

display(scenarios.round(2))
""")

markdown("## 14. Sensitivity Analysis")

code("""
pd_values = np.array([0.02, 0.04, 0.06, 0.08, 0.10])
lgd_values = np.array([0.30, 0.40, 0.50, 0.60, 0.70])

loss_matrix = (
    np.outer(pd_values, lgd_values) * portfolio_ead
)

sensitivity = pd.DataFrame(
    loss_matrix,
    index=[f"{x:.0%}" for x in pd_values],
    columns=[f"{x:.0%}" for x in lgd_values]
)

display(sensitivity.round(2))

sns.heatmap(sensitivity, annot=True, fmt=".0f", cmap="Reds")
plt.title("Credit Loss Sensitivity")
plt.xlabel("LGD")
plt.ylabel("PD")
plt.show()
""")

markdown("## 15. Export Results")

code("""
with pd.ExcelWriter(
    "Bank_Loan_Credit_Risk_Report.xlsx",
    engine="openpyxl"
) as writer:

    audit_report.to_excel(writer, sheet_name="Quality Audit", index=False)
    balance_report.to_excel(writer, sheet_name="Balance Audit", index=False)
    portfolio_summary.to_excel(writer, sheet_name="Portfolio Overview", index=False)
    delinquency_table.to_excel(writer, sheet_name="Delinquency", index=False)
    vintage_table.to_excel(writer, sheet_name="Vintage", index=False)
    term_analysis.to_excel(writer, sheet_name="Term Analysis", index=False)
    grade_analysis.to_excel(writer, sheet_name="Grade Analysis", index=False)
    purpose_analysis.to_excel(writer, sheet_name="Purpose Analysis", index=False)
    scenarios.to_excel(writer, sheet_name="Stress Testing", index=False)
    sensitivity.to_excel(writer, sheet_name="Sensitivity")

print("Excel report generated.")
""")

markdown("""
## 16. Limitations

- LendingClub is not representative of all bank portfolios.
- Historical observations may have selection bias.
- Different vintages have different observation periods.
- A single snapshot cannot produce genuine roll rates.
- Stress assumptions are illustrative, not calibrated.
- Results are not suitable for direct regulatory reporting
  or individual lending decisions.

## Conclusion

Use the notebook outputs to identify portfolio weaknesses,
concentration risks and sensitivity to adverse credit assumptions.

**Author:** Your Name  
**Project:** Bank Loan Portfolio & Credit Risk Monitoring
""")

nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "colab": {
        "name": "Bank_Loan_Portfolio_Credit_Risk.ipynb",
        "provenance": []
    }
}

filename = "Bank_Loan_Portfolio_Credit_Risk.ipynb"

with open(filename, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook created successfully:", filename)

files.download(filename)
