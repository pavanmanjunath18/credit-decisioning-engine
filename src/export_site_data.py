"""Write the small JSON files the static site reads (site/data/).

The site never trains or sees raw data. Every number it shows comes from these files, which are built
from the pipeline's own outputs, and tests/test_site_data.py checks them against src/strategy.py.

Profit at any cost-of-funds rate is exact in the browser because it is linear in the rate:
    profit(rate) = profit_before_cof - rate * dollar_years
so each approval step stores those two totals instead of a precomputed profit.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.explain import REASONS_PATH
from src.features import connect, load_model_table, read_sql, time_split
from src.strategy import APPROVAL_GRID, DECISION_MODEL, approve_top, load_scored
from src.train import TAB

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "data"
RANKINGS = {
    "xgboost_with_lc": "Model (with LendingClub grade)",
    "xgboost_no_lc": "Model (borrower data only)",
    "lc_subgrade_rank": "LendingClub sub-grade",
}
CHOSEN_COF = 0.04  # the cost-of-funds assumption the "chosen policy" is defined at
N_EVEN, N_TAIL = 250, 50  # applicant sample: evenly across the risk range + extra above the cutoff


def r(x, nd=6):
    """Round for compact JSON; nd=6 keeps dollar totals exact to well under $1."""
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def totals(d: pd.DataFrame, approved: np.ndarray) -> dict:
    a = d[approved]
    return {
        "approval_rate": r(approved.mean()),
        "n_approved": int(approved.sum()),
        "default_rate": r(a.is_default.mean()),
        "dollar_loss": r(-a.loc[a.is_default == 1, "net_profit"].sum(), 2),
        "profit_before_cof": r(a.net_profit.sum(), 2),
        "dollar_years": r(a.dollar_years_outstanding.sum(), 2),
    }


def ranking_curves(d: pd.DataFrame, model: str) -> dict:
    """Totals for approving the safest X% (X on the 1% grid), stored as parallel arrays."""
    rows, cutoffs = [], []
    for rate in APPROVAL_GRID:
        approved = approve_top(d[model], rate)
        rows.append(totals(d, approved))
        cutoffs.append(r(d.loc[approved, model].max(), 8))
    out = {k: [row[k] for row in rows] for k in rows[0]}
    out["pd_cutoff"] = cutoffs
    return out


def pd_cutoff_policies(valid: pd.DataFrame, test: pd.DataFrame, model: str) -> dict:
    """For each 2014 grid point, what its PD cutoff approves on 2015 (volume floats, so it is off-grid)."""
    rows = []
    for rate in APPROVAL_GRID:
        cutoff = valid.loc[approve_top(valid[model], rate), model].max()
        rows.append(totals(test, (test[model] <= cutoff).to_numpy()))
    return {k: [row[k] for row in rows] for k in rows[0]}


def export_strategy(scored: dict) -> dict:
    valid, test = scored["valid"], scored["test"]
    grade_rule = ~test.grade.isin(["E", "F", "G"]).to_numpy()
    data = {"grid": [r(x, 2) for x in APPROVAL_GRID], "n_test": len(test), "rankings": {}}
    for model, label in RANKINGS.items():
        data["rankings"][model] = {
            "label": label,
            "test": ranking_curves(test, model),
            "valid": ranking_curves(valid, model),
            "pd_cutoff_on_test": pd_cutoff_policies(valid, test, model),
            "same_volume_as_grade_rule": totals(test, approve_top(test[model], grade_rule.mean())),
        }
    data["grade_rule"] = totals(test, grade_rule)
    return data


def chosen_policy(strategy: dict) -> dict:
    """The policy a lender would actually run: 2014-optimal approval rate at the 4% assumption."""
    v = strategy["rankings"][DECISION_MODEL]["valid"]
    profit = np.array(v["profit_before_cof"]) - CHOSEN_COF * np.array(v["dollar_years"])
    i = int(profit.argmax())
    return {"cof": CHOSEN_COF, "approval_rate": strategy["grid"][i], "pd_cutoff": v["pd_cutoff"][i]}


def export_applicants(test: pd.DataFrame, policy: dict) -> list:
    reasons = pd.read_parquet(REASONS_PATH)
    d = test.merge(reasons, on="id")
    d["percentile"] = d.pd.rank(pct=True)
    d = d.sort_values("pd").reset_index(drop=True)
    # Evenly spaced across the whole risk range, plus extra from above the cutoff so declines are
    # represented (the chosen policy declines only ~1% of applicants).
    even = np.linspace(0, len(d) - 1, N_EVEN).round().astype(int)
    above = d.index[d.pd > policy["pd_cutoff"]]
    tail = np.random.default_rng(42).choice(above.difference(even), N_TAIL, replace=False)
    pick = d.loc[np.sort(np.concatenate([even, tail]))]
    out = []
    for _, a in pick.iterrows():
        out.append({
            "id": str(a.id),
            "pd": r(a.pd, 4), "percentile": r(a.percentile, 4),
            "approved": bool(a.pd <= policy["pd_cutoff"]),
            "reasons": [x for x in (a.reason_1, a.reason_2, a.reason_3) if isinstance(x, str)],
            "outcome": "Charged off" if a.is_default == 1 else "Paid in full",
            "net_profit": r(a.net_profit, 0), "funded": r(a.funded_amnt, 0),
            "loan_amnt": r(a.loan_amnt, 0), "purpose": a.purpose.replace("_", " "),
            "sub_grade": a.sub_grade, "int_rate": r(a.int_rate, 2), "annual_inc": r(a.annual_inc, 0),
            "fico": r(a.fico_mid, 0), "dti": r(a.dti, 1), "home_ownership": a.home_ownership.lower(),
            "emp_length": None if pd.isna(a.emp_length_yrs) else int(a.emp_length_yrs),
        })
    return out


def export_fairness() -> dict:
    f = pd.read_csv(TAB / "fairness_by_group.csv")
    f = f[f.n_applicants >= 100]  # drops home_ownership OTHER (1 applicant)
    out = {}
    for op, g in f.groupby("operating_point"):
        key = "stress_80" if op.startswith("stress") else "chosen"
        out[key] = {"label": op, "groups": [
            {"dimension": row.dimension, "group": row.grp.split(": ")[-1], "n": int(row.n_applicants),
             "approval_rate": r(row.approval_rate, 4), "approval_ratio": r(row.approval_ratio, 4),
             "default_rate_approved": r(row.default_rate_approved, 4),
             "mean_pd_approved": r(row.mean_pd_approved, 4), "small": bool(row.small_group)}
            for row in g.itertuples()]}
    return out


def key_numbers(df: pd.DataFrame, strategy: dict, policy: dict) -> dict:
    mat = pd.read_csv(TAB / "maturity_by_year_term.csv")
    m36 = mat[mat.term_months == 36].set_index("issue_year")
    t = pd.read_csv(TAB / "model_metrics.csv")
    comp = pd.read_csv(TAB / "strategy_comparison.csv")
    c = lambda cof: comp[(comp.model == DECISION_MODEL) & (comp.cof_rate == cof)].set_index("policy")
    c0, c4 = c(0.0), c(0.04)
    curves = pd.read_csv(TAB / "strategy_curves.csv")
    at84 = curves[(curves.split == "test") & (curves.cof_rate == 0.04) & (curves.target_approval == 0.84)]
    at84 = at84.set_index("model").total_profit
    psi = pd.read_csv(TAB / "psi.csv").set_index("feature")
    fair = pd.read_csv(TAB / "fairness_by_group.csv")
    stress = fair[fair.operating_point.str.startswith("stress")].set_index(["dimension", "grp"])
    low, high = stress.loc[("income_band", "1: <$40K")], stress.loc[("income_band", "4: $80-100K")]
    rate_b = connect().sql("SELECT issue_year, avg(int_rate) r FROM loans WHERE grade = 'B' AND issue_year "
                           "IN (2014, 2015) GROUP BY 1").df().set_index("issue_year").r
    ablation = pd.read_csv(TAB / "state_ablation.csv").set_index("model")
    hind = "Best possible 2015 cutoff (hindsight)"
    return {
        "n_loans": int(len(df)),
        "n_test": strategy["n_test"],
        "share_unfinished_2018": r(m36.loc[2018, "share_unfinished"], 4),
        "metrics": [{k: (r(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
                    for row in t.to_dict("records")],
        "profit_all_0": r(c0.loc["Approve everyone", "total_profit"], 0),
        "profit_all_4": r(c4.loc["Approve everyone", "total_profit"], 0),
        "hindsight_gain_0": r(c0.loc[hind, "profit_vs_all"] - 1, 4),
        "hindsight_approval_4": r(c4.loc[hind, "approval_rate"], 4),
        "hindsight_gain_4": r(c4.loc[hind, "profit_vs_all"] - 1, 4),
        "hindsight_profit_4": r(c4.loc[hind, "total_profit"], 0),
        "chosen_gain_4": r(c4.loc["Approval-rate cutoff chosen on 2014", "profit_vs_all"] - 1, 4),
        "chosen_profit_4": r(c4.loc["Approval-rate cutoff chosen on 2014", "total_profit"], 0),
        "chosen_policy": policy,
        "model_vs_subgrade_84": r(at84["xgboost_with_lc"] - at84["lc_subgrade_rank"], 0),
        "psi_int_rate": r(psi.loc["int_rate", "psi_train_vs_test"], 4),
        "psi_score": r(psi.loc["score", "psi_train_vs_test"], 4),
        "grade_b_rate_2014": r(rate_b.loc[2014], 2),
        "grade_b_rate_2015": r(rate_b.loc[2015], 2),
        "low_income_ratio_80": r(low.approval_ratio, 4),
        "income_gap_predicted": r(low.mean_pd_approved - high.mean_pd_approved, 4),
        "income_gap_actual": r(low.default_rate_approved - high.default_rate_approved, 4),
        "state_drop_auc_change": r(ablation.loc["change (no_state - with_state)", "test_roc_auc"], 4),
        "state_drop_profit_change": r(ablation.loc["change (no_state - with_state)", "profit_pd_cutoff"], 0),
    }


def write(name: str, obj):
    (OUT / name).write_text(json.dumps(obj, separators=(",", ":")))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_model_table()
    test = time_split(df)["test"]
    scored = load_scored()

    strategy = export_strategy(scored)
    policy = chosen_policy(strategy)
    write("strategy.json", strategy)
    write("applicants.json", export_applicants(test, policy))
    write("fairness.json", export_fairness())

    psi = pd.read_csv(TAB / "psi.csv")
    write("psi.json", [{"feature": row.feature, "gain_share": r(row.gain_share, 4),
                        "psi_2014": r(row.psi_train_vs_valid, 4), "psi_2015": r(row.psi_train_vs_test, 4)}
                       for row in psi.itertuples()])

    vint = connect().sql(read_sql("08_vintage_curves.sql")).df()
    write("vintages.json", {str(y): {"n": int(g.n_loans.iloc[0]),
                                     "rate": [r(x, 5) for x in g.sort_values("months_since_issue").cumulative_default_rate]}
                            for y, g in vint.groupby("issue_year")})
    write("key_numbers.json", key_numbers(df, strategy, policy))

    total = 0
    for f in sorted(OUT.iterdir()):
        total += f.stat().st_size
        print(f"{f.name:22s} {f.stat().st_size / 1e3:8.1f} KB")
    print(f"{'TOTAL':22s} {total / 1e3:8.1f} KB")


if __name__ == "__main__":
    main()
