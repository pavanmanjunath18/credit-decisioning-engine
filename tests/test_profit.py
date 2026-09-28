"""The profit formula, checked by hand on made-up loans run through the real SQL (sql/03_base_loans.sql).

net_profit = total_pymnt - collection_recovery_fee - funded_amnt
(total_pymnt already includes post-charge-off recoveries, so recoveries are NOT added again.)
"""
import duckdb
import pandas as pd
import pytest

from src.features import read_sql

BASE = {  # application fields the SQL needs; values don't matter for profit
    "loan_amnt": 10000.0, "int_rate": 12.0, "installment": 332.14, "grade": "B", "sub_grade": "B3",
    "emp_length": "5 years", "home_ownership": "RENT", "annual_inc": 60000.0,
    "verification_status": "Verified", "purpose": "credit_card", "addr_state": "AZ", "zip_code": "852xx",
    "dti": 15.0, "delinq_2yrs": 0.0, "earliest_cr_line": "Jan-2005", "fico_range_low": 700.0,
    "fico_range_high": 704.0, "inq_last_6mths": 0.0, "open_acc": 8.0, "pub_rec": 0.0, "revol_bal": 5000.0,
    "revol_util": 40.0, "total_acc": 20.0, "mort_acc": 0.0, "pub_rec_bankruptcies": 0.0,
    "total_rec_int": 0.0, "total_rec_late_fee": 0.0,
}


def loan(**kw):
    return {**BASE, **kw}


RAW = pd.DataFrame([
    # A: paid in full, on schedule. Received $11,743.20 on $10,000 funded -> +$1,743.20.
    #    Capital: 36 months at an average balance of 10,000 - 10,000/2 = 5,000 -> 36 * 5,000 / 12 = 15,000 dollar-years.
    loan(id="A", issue_d="Jan-2014", term=" 36 months", loan_status="Fully Paid", funded_amnt=10000.0,
         total_pymnt=11743.20, recoveries=0.0, collection_recovery_fee=0.0, total_rec_prncp=10000.0,
         last_pymnt_d="Jan-2017"),
    # B: charged off. Received $5,200 in total, which ALREADY includes an $800 recovery; the collection
    #    agency took $144. Profit = 5,200 - 144 - 12,000 = -$6,944 (adding the recovery again would give -$6,144).
    #    Capital: 12 paying months at 12,000 - 3,000/2 = 10,500, then 5 months at 12,000 - 3,000 = 9,000
    #    -> (12 * 10,500 + 5 * 9,000) / 12 = 14,250 dollar-years.
    loan(id="B", issue_d="Mar-2014", term=" 36 months", loan_status="Charged Off", funded_amnt=12000.0,
         total_pymnt=5200.0, recoveries=800.0, collection_recovery_fee=144.0, total_rec_prncp=3000.0,
         last_pymnt_d="Mar-2015"),
    # Excluded by scope: a 60-month loan, a 2016 loan (not matured), and a loan still running.
    loan(id="C", issue_d="Jan-2013", term=" 60 months", loan_status="Fully Paid", funded_amnt=10000.0,
         total_pymnt=12000.0, recoveries=0.0, collection_recovery_fee=0.0, total_rec_prncp=10000.0,
         last_pymnt_d="Jan-2018"),
    loan(id="D", issue_d="Jun-2016", term=" 36 months", loan_status="Fully Paid", funded_amnt=10000.0,
         total_pymnt=10200.0, recoveries=0.0, collection_recovery_fee=0.0, total_rec_prncp=10000.0,
         last_pymnt_d="Jan-2017"),
    loan(id="E", issue_d="Jun-2015", term=" 36 months", loan_status="Current", funded_amnt=10000.0,
         total_pymnt=9000.0, recoveries=0.0, collection_recovery_fee=0.0, total_rec_prncp=8000.0,
         last_pymnt_d="Mar-2019"),
])


@pytest.fixture(scope="module")
def result() -> pd.DataFrame:
    con = duckdb.connect()
    con.register("raw_loans", RAW)
    return con.sql(read_sql("03_base_loans.sql")).df().set_index("id")


def test_only_matured_resolved_36_month_loans_are_kept(result):
    assert sorted(result.index) == ["A", "B"]


def test_profit_on_a_paid_loan(result):
    assert result.loc["A", "net_profit"] == pytest.approx(1743.20)
    assert result.loc["A", "is_default"] == 0


def test_profit_on_a_charged_off_loan_does_not_double_count_recoveries(result):
    assert result.loc["B", "net_profit"] == pytest.approx(-6944.0)
    assert result.loc["B", "is_default"] == 1


def test_capital_outstanding_for_cost_of_funds(result):
    assert result.loc["A", "dollar_years_outstanding"] == pytest.approx(15000.0)
    assert result.loc["B", "months_outstanding"] == 17
    assert result.loc["B", "dollar_years_outstanding"] == pytest.approx(14250.0)
