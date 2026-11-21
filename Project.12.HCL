
# PROJECT 12: BANK LOAN PORTFOLIO & CREDIT RISK
# File generator: Notebook + Dataset + PDF codebook

!pip -q install nbformat reportlab

import os
import urllib.request
import nbformat as nbf
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Preformatted, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib import colors
from google.colab import files

# --------------------------------------------------
# 1. DOWNLOAD THE PUBLIC DATASET
# --------------------------------------------------

data_url = (
    "https://digintu.tech/tmp/cs109a/"
    "loan_accepted_10.csv"
)

data_file = "LendingClub_Loan_Data.csv"

print("Downloading LendingClub dataset...")

urllib.request.urlretrieve(data_url, data_file)

print("Dataset downloaded:", data_file)

# --------------------------------------------------
# 2. CREATE THE NOTEBOOK
# --------------------------------------------------

notebook = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell("""
# Project 12: Bank Loan Portfolio & Credit Risk Monitoring

**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn

**Dataset:** Historical LendingClub accepted loans.

## Business objective

Monitor portfolio composition, repayment performance,
delinquency patterns, exposure concentration and
credit loss sensitivity.

The project includes:
1. Portfolio quality audit
2. Balance reconciliation
3. Vintage and delinquency analysis
4. NumPy weighted calculations
5. Segment performance
6. Concentration analysis
7. Stress testing
8. Risk limitations

This is an educational portfolio analysis, not a
regulatory credit risk model.
"""))

cells.append(nbf.v4.new_code_cell("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

plt.rcParams["figure.figsize"] = (10, 5)

print("Libraries imported successfully.")
"""))

cells.append(nbf.v4.new_code_cell("""
# Load the downloaded loan dataset

df = pd.read_csv(
    "LendingClub_Loan_Data.csv",
    low_memory=False
)

print("Dataset shape:", df.shape)
display(df.head())

print("\\\\nAvailable columns:")
print(df.columns.tolist())
"""))

cells.append(nbf.v4.new_code_cell("""
# Portfolio quality audit

audit = {
    "Total Records": len(df),
    "Missing IDs": df["id"].isna().sum(),
    "Duplicate IDs": df["id"].duplicated().sum(),
    "Missing Loan Status": df["loan_status"].isna().sum(),
    "Missing Origination Date": df["issue_d"].isna().sum()
}

audit_report = pd.DataFrame(
    audit.items(),
    columns=["Quality Check", "Issue Count"]
)

audit_report["Issue Percentage"] = (
    audit_report["Issue Count"] / len(df) * 100
)

display(audit_report)

# Remove unidentified and duplicate loans
df = df.dropna(subset=["id"]).copy()
df = df.drop_duplicates(subset=["id"]).copy()
"""))

cells.append(nbf.v4.new_code_cell("""
# Clean numerical fields and dates

numeric_columns = [
    "loan_amnt", "funded_amnt", "int_rate",
    "installment", "annual_inc", "dti",
    "out_prncp", "total_pymnt",
    "total_rec_prncp", "recoveries"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(
                "%", "", regex=False
            ),
            errors="coerce"
        )

df["issue_d"] = pd.to_datetime(
    df["issue_d"],
    format="%b-%Y",
    errors="coerce"
)

df = df.dropna(subset=["issue_d"]).copy()

print("Cleaned dataset shape:", df.shape)
"""))

cells.append(nbf.v4.new_code_cell("""
# Create loan risk categories

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
    df["loan_status"]
    .map(status_mapping)
    .fillna("Other")
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
    df["term"].astype(str).str.extract(r"(\\\\d+)")[0],
    errors="coerce"
)

display(df[[
    "id", "issue_d", "risk_status",
    "exposure", "vintage_year"
]].head())
"""))

cells.append(nbf.v4.new_code_cell("""
# Portfolio overview

total_loans = df["id"].nunique()
total_funded = df["funded_amnt"].sum()
total_exposure = df["exposure"].sum()

delinquent_exposure = df.loc[
    df["delinquency_flag"], "exposure"
].sum()

delinquency_rate = (
    delinquent_exposure / total_exposure * 100
    if total_exposure > 0 else np.nan
)

portfolio_summary = pd.DataFrame({
    "Metric": [
        "Total Loans",
        "Total Funded Amount",
        "Outstanding Exposure",
        "Delinquent Exposure",
        "Delinquency Rate"
    ],
    "Value": [
        total_loans,
        total_funded,
        total_exposure,
        delinquent_exposure,
        delinquency_rate
    ]
})

display(portfolio_summary)
"""))

cells.append(nbf.v4.new_code_cell("""
# NumPy weighted interest rate and exposure quantiles

balance = df["exposure"].to_numpy(dtype=float)
rate = df["int_rate"].to_numpy(dtype=float)

valid = (
    np.isfinite(balance) &
    np.isfinite(rate) &
    (balance > 0) &
    (balance >= 0)
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
"""))

cells.append(nbf.v4.new_code_cell("""
# Delinquency analysis

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
"""))

cells.append(nbf.v4.new_code_cell("""
# Vintage analysis

vintage_table = (
    df.groupby("vintage_year")
    .agg(
        loan_count=("id", "nunique"),
        funded_amount=("funded_amnt", "sum"),
        outstanding=("exposure", "sum"),
        default_rate=("default_flag", "mean"),
        delinquency_rate=("delinquency_flag", "mean")
    )
    .reset_index()
)

vintage_table["default_rate"] *= 100
vintage_table["delinquency_rate"] *= 100

display(vintage_table)

sns.barplot(
    data=vintage_table,
    x="vintage_year",
    y="default_rate",
    color="steelblue"
)

plt.title("Observed Default Rate by Vintage")
plt.xticks(rotation=45)
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""
# Performance by term, purpose and credit grade

term_table = df.groupby("term_months").agg(
    loans=("id", "nunique"),
    delinquency_rate=("delinquency_flag", "mean"),
    default_rate=("default_flag", "mean")
).reset_index()

grade_table = df.groupby("grade").agg(
    loans=("id", "nunique"),
    exposure=("exposure", "sum"),
    default_rate=("default_flag", "mean"),
    delinquency_rate=("delinquency_flag", "mean")
).reset_index()

purpose_table = df.groupby("purpose").agg(
    loans=("id", "nunique"),
    exposure=("exposure", "sum"),
    default_rate=("default_flag", "mean")
).reset_index()

for table in [term_table, grade_table, purpose_table]:
    for col in ["default_rate", "delinquency_rate"]:
        if col in table.columns:
            table[col] *= 100

display(term_table)
display(grade_table)
display(purpose_table)
"""))

cells.append(nbf.v4.new_code_cell("""
# Concentration analysis using HHI

def calculate_hhi(data, segment):

    grouped = data.groupby(
        segment, dropna=False
    )["exposure"].sum().dropna()

    total = grouped.sum()

    if total <= 0:
        return pd.DataFrame(), np.nan

    shares = grouped / total * 100

    result = shares.reset_index()
    result.columns = [segment, "exposure_share"]

    hhi = (shares ** 2).sum()

    return result.sort_values(
        "exposure_share", ascending=False
    ), hhi


grade_concentration, grade_hhi = calculate_hhi(
    df, "grade"
)

purpose_concentration, purpose_hhi = calculate_hhi(
    df, "purpose"
)

state_concentration, state_hhi = calculate_hhi(
    df, "addr_state"
)

print("Grade HHI:", grade_hhi)
print("Purpose HHI:", purpose_hhi)
print("Geographic HHI:", state_hhi)

display(grade_concentration)
display(purpose_concentration)
display(state_concentration)
"""))

cells.append(nbf.v4.new_code_cell("""
# Stress testing using PD x LGD x EAD

portfolio_ead = df["exposure"].sum()

scenarios = pd.DataFrame({
    "Scenario": ["Base", "Moderate", "Severe"],
    "PD": [0.03, 0.06, 0.10],
    "LGD": [0.40, 0.50, 0.65]
})

scenarios["EAD"] = portfolio_ead

scenarios["Estimated_Loss"] = (
    scenarios["PD"] *
    scenarios["LGD"] *
    scenarios["EAD"]
)

display(scenarios)
"""))

cells.append(nbf.v4.new_code_cell("""
# NumPy sensitivity analysis

pd_values = np.array([0.02, 0.04, 0.06, 0.08, 0.10])
lgd_values = np.array([0.30, 0.40, 0.50, 0.60, 0.70])

loss_matrix = (
    np.outer(pd_values, lgd_values) *
    portfolio_ead
)

sensitivity_table = pd.DataFrame(
    loss_matrix,
    index=[f"{x:.0%}" for x in pd_values],
    columns=[f"{x:.0%}" for x in lgd_values]
)

display(sensitivity_table)

sns.heatmap(
    sensitivity_table,
    annot=True,
    fmt=".0f",
    cmap="Reds"
)

plt.title("Credit Loss Sensitivity")
plt.xlabel("LGD")
plt.ylabel("PD")
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""
# Export the analysis tables into Excel

with pd.ExcelWriter(
    "Bank_Loan_Risk_Report.xlsx",
    engine="openpyxl"
) as writer:

    audit_report.to_excel(
        writer, sheet_name="Quality Audit", index=False
    )

    portfolio_summary.to_excel(
        writer, sheet_name="Portfolio Overview", index=False
    )

    delinquency_table.to_excel(
        writer, sheet_name="Delinquency", index=False
    )

    vintage_table.to_excel(
        writer, sheet_name="Vintage", index=False
    )

    term_table.to_excel(
        writer, sheet_name="Term", index=False
    )

    grade_table.to_excel(
        writer, sheet_name="Credit Grade", index=False
    )

    purpose_table.to_excel(
        writer, sheet_name="Purpose", index=False
    )

    scenarios.to_excel(
        writer, sheet_name="Stress Testing", index=False
    )

    sensitivity_table.to_excel(
        writer, sheet_name="Sensitivity"
    )

print("Excel report created.")
"""))

cells.append(nbf.v4.new_markdown_cell("""
## Important limitations

- LendingClub is not a representative regulated bank portfolio.
- A single snapshot cannot establish historical roll rates.
- Observed default percentages are not time-adjusted PD estimates.
- The current outstanding balance is not necessarily historical EAD.
- Stress assumptions are illustrative, not calibrated.
- The project must not be used directly for regulatory reporting,
  capital adequacy or individual lending decisions.

## Final conclusion

Use the actual analysis outputs to identify the portfolio's
weaker-performing segments, concentration risks and sensitivity
to adverse credit assumptions.
"""))

notebook.cells = cells
notebook.metadata = {
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

ipynb_file = "Bank_Loan_Portfolio_Credit_Risk.ipynb"

with open(ipynb_file, "w", encoding="utf-8") as file:
    nbf.write(notebook, file)

print("Notebook created:", ipynb_file)

# --------------------------------------------------
# 3. CREATE A PDF VERSION OF THE PYTHON CODE
# --------------------------------------------------

pdf_file = "Bank_Loan_Credit_Risk_Python_Code.pdf"

styles = getSampleStyleSheet()

story = []

story.append(Paragraph(
    "BANK LOAN PORTFOLIO &amp; CREDIT RISK MONITORING",
    styles["Title"]
))

story.append(Spacer(1, 8))

story.append(Paragraph(
    "Google Colab Python Notebook - Complete Code Listing",
    styles["Heading2"]
))

story.append(Spacer(1, 15))

story.append(Paragraph(
    "Dataset: Historical LendingClub accepted loan records. "
    "The notebook includes portfolio quality checks, "
    "delinquency analysis, vintage analysis, concentration "
    "and stress testing.",
    styles["BodyText"]
))

story.append(Spacer(1, 15))

for number, cell in enumerate(cells, start=1):

    if cell.cell_type == "markdown":
        for line in cell.source.splitlines():
            if line.startswith("# "):
                story.append(Paragraph(
                    line[2:], styles["Heading1"]
                ))
            elif line.startswith("## "):
                story.append(Paragraph(
                    line[3:], styles["Heading2"]
                ))
            elif line.strip():
                story.append(Paragraph(
                    line.replace("&", "&amp;").replace("<", "&lt;"),
                    styles["BodyText"]
                ))

    elif cell.cell_type == "code":

        story.append(Paragraph(
            f"Code Cell {number}",
            styles["Heading3"]
        ))

        story.append(
            Preformatted(
                cell.source,
                styles["Code"],
                maxLineLength=95
            )
        )

        story.append(Spacer(1, 8))

def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(
        18 * mm, 10 * mm,
        "Bank Loan Portfolio & Credit Risk Monitoring"
    )
    canvas.drawRightString(
        192 * mm, 10 * mm,
        f"Page {doc.page}"
    )
    canvas.restoreState()

pdf = SimpleDocTemplate(
    pdf_file,
    pagesize=A4,
    rightMargin=16 * mm,
    leftMargin=16 * mm,
    topMargin=17 * mm,
    bottomMargin=18 * mm
)

pdf.build(
    story,
    onFirstPage=add_page_number,
    onLaterPages=add_page_number
)

print("PDF created:", pdf_file)

# --------------------------------------------------
# 4. DOWNLOAD ALL THREE FILES
# --------------------------------------------------

print("\\nAll project assets are ready!")

for file_name in [
    ipynb_file,
    data_file,
    pdf_file
]:
    print(file_name, "-", round(
        os.path.getsize(file_name) / (1024 * 1024), 2
    ), "MB")

files.download(ipynb_file)
files.download(data_file)
files.download(pdf_file)
