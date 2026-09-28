"""Load the raw LendingClub file, reading only the columns this project needs.

The raw CSV (~1.6 GB uncompressed) is parsed once and cached as parquet in
data/processed/. All later filtering and feature prep happens in DuckDB (sql/).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv.gz"
PROCESSED = ROOT / "data" / "processed"
RAW_PARQUET = PROCESSED / "loans_raw.parquet"

# Known at application time -> allowed as model inputs.
APPLICATION_COLS = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "purpose", "addr_state", "zip_code", "dti", "delinq_2yrs",
    "earliest_cr_line", "fico_range_low", "fico_range_high", "inq_last_6mths",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc",
    "pub_rec_bankruptcies",
]

# Outcomes: used only to measure results, never as model inputs.
OUTCOME_COLS = [
    "loan_status", "total_pymnt", "recoveries", "collection_recovery_fee",
    "funded_amnt",
    # Payment components, loaded only to verify whether total_pymnt includes recoveries.
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    # Date of last payment: how long capital was tied up (for the cost-of-funds adjustment).
    "last_pymnt_d",
]

KEY_COLS = ["id", "issue_d"]

USECOLS = KEY_COLS + APPLICATION_COLS + OUTCOME_COLS


def load_raw(force: bool = False) -> pd.DataFrame:
    """Return all accepted loans (all statuses) with only USECOLS, cached as parquet."""
    if RAW_PARQUET.exists() and not force:
        return pd.read_parquet(RAW_PARQUET)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(
        RAW_PATH,
        usecols=USECOLS,
        dtype={"id": "string", "zip_code": "string", "emp_length": "string"},
        low_memory=False,
    )
    # The file ends with a few summary rows that have no issue date; drop them.
    df = df[df["issue_d"].notna()].copy()
    df.to_parquet(RAW_PARQUET, index=False)
    return df


if __name__ == "__main__":
    d = load_raw(force=True)
    print(d.shape)
    print(d.dtypes)
