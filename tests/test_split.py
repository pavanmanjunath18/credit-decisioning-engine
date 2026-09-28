"""The time split must never let a loan (or a later month) leak across train / valid / test."""
import pandas as pd

from src.features import SPLIT_YEARS, time_split


def _toy_loans():
    dates = pd.date_range("2007-06-01", "2015-12-01", freq="MS")
    return pd.DataFrame({
        "id": [str(i) for i in range(len(dates))],
        "issue_date": dates,
        "issue_year": dates.year,
    })


def test_split_has_no_overlap_and_is_ordered_in_time():
    splits = time_split(_toy_loans())
    ids = {k: set(v["id"]) for k, v in splits.items()}
    assert not ids["train"] & ids["valid"]
    assert not ids["train"] & ids["test"]
    assert not ids["valid"] & ids["test"]
    assert splits["train"].issue_date.max() < splits["valid"].issue_date.min()
    assert splits["valid"].issue_date.max() < splits["test"].issue_date.min()


def test_split_covers_every_loan_exactly_once():
    df = _toy_loans()
    splits = time_split(df)
    assert sum(len(v) for v in splits.values()) == len(df)


def test_split_years_do_not_overlap():
    ranges = sorted(SPLIT_YEARS.values())
    for (_, prev_hi), (next_lo, _) in zip(ranges, ranges[1:]):
        assert prev_hi < next_lo
