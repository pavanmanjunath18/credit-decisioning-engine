"""SQL-driven feature prep (DuckDB) and the single source of truth for model input columns."""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.data import PROCESSED, RAW_PARQUET, load_raw

ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = ROOT / "sql"
MODEL_PARQUET = PROCESSED / "loans_model.parquet"

# LendingClub's own risk outputs. Models are trained with and without these.
LC_RISK_COLS = ["grade", "sub_grade", "sub_grade_num", "int_rate"]

# Derived from LC's price: for a fixed 36-month term, installment / loan_amnt pins down int_rate
# (the annuity formula recovers int_rate within 0.1pp for 99.8% of loans). So these only enter the
# "with LC" models; otherwise the "without LC" model would quietly see LC's pricing.
LC_PRICE_DERIVED = ["installment", "payment_to_income"]

NUMERIC_FEATURES = [
    "loan_amnt", "emp_length_yrs", "annual_inc", "dti",
    "delinq_2yrs", "credit_history_months", "fico_mid", "inq_last_6mths",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc",
    "pub_rec_bankruptcies", "loan_to_income",
]
# addr_state is deliberately NOT a model input: decline reasons must reflect what drives the decision,
# and "your state of residence" is not a defensible reason. Dropping it cost 0.0006 test AUC and
# $51K of 2015 profit at the 2014-chosen cutoff (src/state_check.py). It is kept for fairness checks.
CATEGORICAL_FEATURES = ["home_ownership", "verification_status", "purpose"]

# Outcome columns: never allowed as model inputs (enforced in tests/).
OUTCOME_COLS = [
    "loan_status", "is_default", "funded_amnt", "total_pymnt", "recoveries",
    "collection_recovery_fee", "net_profit", "months_outstanding", "dollar_years_outstanding",
]


# Time-based split by issue year. Random splits would let the model see loans from the same
# months it is tested on, hiding the vintage-to-vintage drift a real lender faces.
SPLIT_YEARS = {
    "train": (2007, 2013),
    "valid": (2014, 2014),
    "test": (2015, 2015),
}


def time_split(df: pd.DataFrame, years: dict = SPLIT_YEARS) -> dict[str, pd.DataFrame]:
    return {
        name: df[df["issue_year"].between(lo, hi)].reset_index(drop=True)
        for name, (lo, hi) in years.items()
    }


class Clipper(BaseEstimator, TransformerMixin):
    """Cap each column at its training 1st/99th percentile.

    A logistic regression is linear in its inputs, so a single $9M income would otherwise
    dominate the fitted coefficient. Missing values pass through untouched (imputed next).
    """

    def __init__(self, lower=1, upper=99):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanpercentile(X, self.lower, axis=0)
        self.hi_ = np.nanpercentile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


def feature_columns(include_lc_risk: bool) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) model inputs."""
    num = list(NUMERIC_FEATURES)
    cat = list(CATEGORICAL_FEATURES)
    if include_lc_risk:
        num += LC_PRICE_DERIVED + ["sub_grade_num", "int_rate"]
    return num, cat


def read_sql(name: str) -> str:
    return (SQL_DIR / name).read_text()


def connect() -> duckdb.DuckDBPyConnection:
    """DuckDB connection with raw_loans (all statuses) and, if built, loans (modeling table)."""
    if not RAW_PARQUET.exists():
        load_raw()
    con = duckdb.connect()
    con.execute(f"CREATE VIEW raw_loans AS SELECT * FROM '{RAW_PARQUET}'")
    if MODEL_PARQUET.exists():
        con.execute(f"CREATE VIEW loans AS SELECT * FROM '{MODEL_PARQUET}'")
    return con


def build_model_table() -> pd.DataFrame:
    con = connect()
    df = con.sql(read_sql("03_base_loans.sql")).df()
    df.to_parquet(MODEL_PARQUET, index=False)
    return df


def load_model_table() -> pd.DataFrame:
    df = build_model_table() if not MODEL_PARQUET.exists() else pd.read_parquet(MODEL_PARQUET)
    # DuckDB integers with NULLs arrive as pandas nullable ints (pd.NA); models need plain floats/NaN.
    num = NUMERIC_FEATURES + LC_PRICE_DERIVED + ["sub_grade_num", "int_rate"]
    df[num] = df[num].astype("float64")
    return df


if __name__ == "__main__":
    d = build_model_table()
    print(d.shape)
    print(d["issue_year"].value_counts().sort_index())
