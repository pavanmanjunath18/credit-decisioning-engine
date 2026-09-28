"""The static site must show the same numbers as the analysis.

The browser computes profit = profit_before_cof - rate * dollar_years and picks cutoffs with argmax over
the 1% grid. This test does exactly that in Python from site/data/strategy.json and compares every policy
to src/strategy.py's output (reports/tables/strategy_comparison.csv and strategy_curves.csv).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE = json.loads((ROOT / "site" / "data" / "strategy.json").read_text())
COMPARISON = pd.read_csv(ROOT / "reports" / "tables" / "strategy_comparison.csv")
CURVES = pd.read_csv(ROOT / "reports" / "tables" / "strategy_curves.csv")
RANKINGS = ["xgboost_with_lc", "xgboost_no_lc", "lc_subgrade_rank"]


def profit(block: dict, cof: float) -> np.ndarray:
    """What site/app.js computes: exact because cost of funds is linear in the rate."""
    return np.array(block["profit_before_cof"]) - cof * np.array(block["dollar_years"])


def at(block: dict, i: int, cof: float) -> dict:
    return {"approval_rate": block["approval_rate"][i], "default_rate": block["default_rate"][i],
            "dollar_loss": block["dollar_loss"][i], "total_profit": profit(block, cof)[i]}


def single(block: dict, cof: float) -> dict:
    return {"approval_rate": block["approval_rate"], "default_rate": block["default_rate"],
            "dollar_loss": block["dollar_loss"],
            "total_profit": block["profit_before_cof"] - cof * block["dollar_years"]}


def site_policies(model: str, cof: float) -> dict:
    rk = SITE["rankings"][model]
    i_2014 = int(profit(rk["valid"], cof).argmax())
    i_2015 = int(profit(rk["test"], cof).argmax())
    last = len(SITE["grid"]) - 1
    return {
        "Approve everyone": at(rk["test"], last, cof),
        "Decline grades E-G (LC grade rule)": single(SITE["grade_rule"], cof),
        "Model, same approval rate as E-G rule": single(rk["same_volume_as_grade_rule"], cof),
        "PD cutoff chosen on 2014": at(rk["pd_cutoff_on_test"], i_2014, cof),
        "Approval-rate cutoff chosen on 2014": at(rk["test"], i_2014, cof),
        "Best possible 2015 cutoff (hindsight)": at(rk["test"], i_2015, cof),
    }


@pytest.mark.parametrize("model", RANKINGS)
@pytest.mark.parametrize("cof", [0.0, 0.04])
def test_every_policy_matches_strategy_py(model, cof):
    ref = COMPARISON[(COMPARISON.model == model) & np.isclose(COMPARISON.cof_rate, cof)].set_index("policy")
    for policy, got in site_policies(model, cof).items():
        exp = ref.loc[policy]
        assert got["approval_rate"] == pytest.approx(exp.approval_rate, abs=1e-6), policy
        assert got["default_rate"] == pytest.approx(exp.default_rate, abs=1e-6), policy
        assert got["dollar_loss"] == pytest.approx(exp.dollar_loss, abs=1.0), policy  # within $1
        assert got["total_profit"] == pytest.approx(exp.total_profit, abs=1.0), policy


@pytest.mark.parametrize("model", RANKINGS)
@pytest.mark.parametrize("cof", [0.0, 0.04])
def test_curves_match_strategy_py(model, cof):
    ref = CURVES[(CURVES.split == "test") & (CURVES.model == model) & np.isclose(CURVES.cof_rate, cof)]
    ref = ref.sort_values("target_approval")
    block = SITE["rankings"][model]["test"]
    np.testing.assert_allclose(SITE["grid"], ref.target_approval, atol=1e-9)
    np.testing.assert_allclose(block["default_rate"], ref.default_rate, atol=1e-6)
    np.testing.assert_allclose(profit(block, cof), ref.total_profit, atol=1.0)


def test_site_stays_small():
    total = sum(f.stat().st_size for f in (ROOT / "site").rglob("*") if f.is_file())
    assert total < 3_000_000
