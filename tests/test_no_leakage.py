"""Model inputs must be application-time fields only: no outcomes, no zip code, no time keys."""
from pathlib import Path

import joblib
import pytest

from src.features import OUTCOME_COLS, feature_columns

MODELS = Path(__file__).resolve().parents[1] / "models"

# Beyond the declared outcome list, any column that describes what happened after funding.
# (total_acc is an application field - number of credit lines - so match "total_pymnt"/"total_rec", not "total_".)
POST_ORIGINATION_PREFIXES = ("total_pymnt", "total_rec", "recover", "collection", "last_pymnt",
                             "out_prncp", "settlement")
# Excluded by decision: zip3 and state are geography (fair-lending proxy risk, not a defensible decline
# reason); ids/dates would let the model memorize vintage.
EXCLUDED = {"zip3", "zip_code", "addr_state", "id", "issue_date", "issue_year", "issue_d"}


@pytest.mark.parametrize("include_lc", [False, True])
def test_feature_lists_have_no_outcomes(include_lc):
    num, cat = feature_columns(include_lc)
    inputs = set(num + cat)
    assert not inputs & set(OUTCOME_COLS)
    assert not inputs & EXCLUDED
    assert not [c for c in inputs if c.startswith(POST_ORIGINATION_PREFIXES)]


def test_lc_risk_columns_only_in_with_lc_variant():
    num_no, cat_no = feature_columns(False)
    num_with, _ = feature_columns(True)
    lc_pricing = {"int_rate", "sub_grade_num", "grade", "sub_grade", "installment", "payment_to_income"}
    assert not lc_pricing & set(num_no + cat_no)
    assert {"int_rate", "sub_grade_num", "installment", "payment_to_income"} <= set(num_with)


@pytest.mark.parametrize("path", sorted(MODELS.glob("*.joblib")), ids=lambda p: p.stem)
def test_saved_models_use_only_allowed_inputs(path):
    bundle = joblib.load(path)
    inputs = set(bundle["num"] + bundle["cat"])
    assert not inputs & set(OUTCOME_COLS)
    assert not inputs & EXCLUDED
