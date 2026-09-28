"""Phase 4: turn scores into approve/decline decisions and measure them in dollars.

Every cutoff is CHOSEN on 2014 (validation) and REPORTED on 2015 (test). The best-possible 2015 cutoff
is shown only as a yardstick for how much was left on the table.

Two ways to carry a 2014 cutoff into 2015:
  * PD cutoff:            approve if predicted PD <= the 2014 cutoff value. Approval volume floats.
  * Approval-rate cutoff: approve the safest X% of applicants. Volume is fixed; the PD bar floats.
They react differently to calibration drift, which is the point of comparing them.

Profit per loan uses actual outcomes: net_profit = total_pymnt - collection_recovery_fee - funded_amnt.
Optional cost of funds: annual_rate * dollar_years_outstanding (see sql/03_base_loans.sql).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.eda import BLUE, FIG, GRID, INK2, MUTED, ORANGE, _pct, _save, plt
from src.features import load_model_table, time_split
from src.train import PRED_PATH, TAB

DECISION_MODEL = "xgboost_with_lc"
COMPARE_MODELS = ["xgboost_with_lc", "xgboost_no_lc", "lc_subgrade_rank"]
MODEL_NAMES = {
    "xgboost_with_lc": "XGBoost with LC grade/rate (decision model)",
    "xgboost_no_lc": "XGBoost without LC grade/rate",
    "lc_subgrade_rank": "Rank by LC sub-grade only",
}
# An input assumption, not a result: a plausible annual cost of capital for a consumer lender.
DEFAULT_COF_RATE = 0.04
APPROVAL_GRID = np.round(np.arange(0.50, 1.0001, 0.01), 2)
GREEN = "#1baf7a"


def load_scored() -> dict[str, pd.DataFrame]:
    """Per split: one row per loan with outcomes and a score column per model (higher = riskier)."""
    df = load_model_table()
    splits = time_split(df)
    preds = pd.read_parquet(PRED_PATH)
    wide = preds.pivot_table(index="id", columns="model", values="pd")
    subgrade_rate = splits["train"].groupby("sub_grade").is_default.mean()
    out = {}
    for name in ["valid", "test"]:
        d = splits[name][["id", "issue_year", "grade", "sub_grade", "is_default", "funded_amnt",
                          "net_profit", "dollar_years_outstanding"]].merge(wide, on="id")
        d["lc_subgrade_rank"] = d.sub_grade.map(subgrade_rate)
        out[name] = d
    return out


def profit(d: pd.DataFrame, cof_rate: float) -> pd.Series:
    return d.net_profit - cof_rate * d.dollar_years_outstanding


def summarize(d: pd.DataFrame, approved: np.ndarray, cof_rate: float) -> dict:
    """Portfolio outcome of approving the rows where `approved` is True."""
    a = d[approved]
    p = profit(a, cof_rate)
    return {
        "approval_rate": approved.mean(),
        "n_approved": int(approved.sum()),
        "default_rate": a.is_default.mean(),
        # Dollar loss = net money lost on the approved loans that charged off.
        "dollar_loss": -a.loc[a.is_default == 1, "net_profit"].sum(),
        "total_profit": p.sum(),
        "profit_per_loan": p.mean(),
    }


def approve_top(scores: pd.Series, rate: float) -> np.ndarray:
    """Approve the safest `rate` share of applicants (lowest score). Ties broken by row order."""
    n = int(round(rate * len(scores)))
    order = np.argsort(scores.to_numpy(), kind="stable")
    approved = np.zeros(len(scores), dtype=bool)
    approved[order[:n]] = True
    return approved


def curve(d: pd.DataFrame, model: str, cof_rate: float) -> pd.DataFrame:
    rows = []
    for rate in APPROVAL_GRID:
        approved = approve_top(d[model], rate)
        pd_cutoff = d.loc[approved, model].max()
        rows.append({"target_approval": rate, "pd_cutoff": pd_cutoff, **summarize(d, approved, cof_rate)})
    return pd.DataFrame(rows)


def compare_strategies(valid, test, model, cof_rate) -> pd.DataFrame:
    """All policies for one scoring model and one profit definition, reported on 2015."""
    v_curve, t_curve = curve(valid, model, cof_rate), curve(test, model, cof_rate)
    best_v = v_curve.loc[v_curve.total_profit.idxmax()]
    best_t = t_curve.loc[t_curve.total_profit.idxmax()]

    grade_rule = ~test.grade.isin(["E", "F", "G"]).to_numpy()
    policies = {
        "Approve everyone": np.ones(len(test), dtype=bool),
        "Decline grades E-G (LC grade rule)": grade_rule,
        # Same volume as the grade rule: a like-for-like test of which ranking picks better loans.
        "Model, same approval rate as E-G rule": approve_top(test[model], grade_rule.mean()),
        "PD cutoff chosen on 2014": (test[model] <= best_v.pd_cutoff).to_numpy(),
        "Approval-rate cutoff chosen on 2014": approve_top(test[model], best_v.target_approval),
        "Best possible 2015 cutoff (hindsight)": approve_top(test[model], best_t.target_approval),
    }
    rows = []
    for name, approved in policies.items():
        rows.append({"model": model, "cof_rate": cof_rate, "policy": name, **summarize(test, approved, cof_rate)})
    out = pd.DataFrame(rows)
    out["chosen_on_2014_approval"] = best_v.target_approval
    out["chosen_on_2014_pd_cutoff"] = best_v.pd_cutoff
    out["valid_approval_at_pd_cutoff"] = best_v.approval_rate
    base = out.iloc[0]
    out["loss_cut_vs_all"] = 1 - out.dollar_loss / base.dollar_loss
    out["profit_vs_all"] = out.total_profit / base.total_profit
    return out


def business_sentence(row) -> str:
    kept = row.profit_vs_all
    kept_txt = f"keeps {kept:.0%} of profit" if kept <= 1 else f"raises profit {kept - 1:.1%}"
    return (f"{row.policy}: approving {row.approval_rate:.1%} of applicants cuts dollar losses "
            f"{row.loss_cut_vs_all:.1%} and {kept_txt} vs approving everyone "
            f"(default rate {row.default_rate:.1%}, profit ${row.total_profit / 1e6:,.1f}M).")


def profit_by_decile(d: pd.DataFrame, model: str, cof_rate: float) -> pd.DataFrame:
    """Average outcome per loan within each tenth of applicants, from safest (1) to riskiest (10)."""
    t = d.assign(decile=pd.qcut(d[model].rank(method="first"), 10, labels=range(1, 11)),
                 profit=profit(d, cof_rate))
    return (t.groupby("decile", observed=True)
             .agg(n=("id", "size"), mean_pd=(model, "mean"), default_rate=("is_default", "mean"),
                  avg_profit=("profit", "mean"), total_profit=("profit", "sum"))
             .reset_index().assign(cof_rate=cof_rate))


def plot_profit_curves(curves: pd.DataFrame, comparison: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    colors = {"xgboost_with_lc": BLUE, "xgboost_no_lc": ORANGE, "lc_subgrade_rank": GREEN}
    for ax, cof in zip(axes, [0.0, DEFAULT_COF_RATE]):
        c = curves[(curves.split == "test") & (curves.cof_rate == cof)]
        for model in COMPARE_MODELS:
            m = c[c.model == model]
            ax.plot(m.approval_rate, m.total_profit / 1e6, color=colors[model], lw=2, label=MODEL_NAMES[model])
        comp = comparison[(comparison.model == DECISION_MODEL) & (comparison.cof_rate == cof)].set_index("policy")
        eg = comp.loc["Decline grades E-G (LC grade rule)"]
        ax.plot(eg.approval_rate, eg.total_profit / 1e6, "s", color=INK2, ms=8, label="Decline grades E–G")
        pdc = comp.loc["PD cutoff chosen on 2014"]
        ax.plot(pdc.approval_rate, pdc.total_profit / 1e6, "o", color=BLUE, ms=9, mfc="white", mew=2,
                label="PD cutoff chosen on 2014")
        arc = comp.loc["Approval-rate cutoff chosen on 2014"]
        ax.plot(arc.approval_rate, arc.total_profit / 1e6, "D", color=BLUE, ms=8, mfc="white", mew=2,
                label="Approval-rate cutoff chosen on 2014")
        _pct(ax, "x")
        ax.grid(axis="x")
        ax.set_xlabel("Approval rate (approve the safest X% by score)")
        ax.set_ylabel("Total net profit on 2015 loans ($M)")
        ax.set_title("No cost of funds" if cof == 0 else f"After {cof:.0%} annual cost of funds")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("2015 test set: profit by approval rate", x=0.01, ha="left", fontsize=13, fontweight="bold")
    _save(fig, "12_profit_curve.png")


def plot_default_curve(curves: pd.DataFrame, comparison: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = {"xgboost_with_lc": BLUE, "xgboost_no_lc": ORANGE, "lc_subgrade_rank": GREEN}
    c = curves[(curves.split == "test") & (curves.cof_rate == 0.0)]
    for model in COMPARE_MODELS:
        m = c[c.model == model]
        ax.plot(m.approval_rate, m.default_rate, color=colors[model], lw=2, label=MODEL_NAMES[model])
    eg = comparison[(comparison.model == DECISION_MODEL) & (comparison.cof_rate == 0.0)
                    & (comparison.policy == "Decline grades E-G (LC grade rule)")].iloc[0]
    ax.plot(eg.approval_rate, eg.default_rate, "s", color=INK2, ms=8, label="Decline grades E–G")
    _pct(ax); _pct(ax, "x"); ax.grid(axis="x")
    ax.set_xlabel("Approval rate"); ax.set_ylabel("Default rate among approved loans")
    ax.set_title("2015 test set: default rate of the approved book")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _save(fig, "13_default_rate_curve.png")


def main():
    scored = load_scored()
    valid, test = scored["valid"], scored["test"]

    curves = []
    for split, d in [("valid", valid), ("test", test)]:
        for model in COMPARE_MODELS:
            for cof in [0.0, DEFAULT_COF_RATE]:
                curves.append(curve(d, model, cof).assign(split=split, model=model, cof_rate=cof))
    curves = pd.concat(curves, ignore_index=True)
    curves.to_csv(TAB / "strategy_curves.csv", index=False)

    comparison = pd.concat([compare_strategies(valid, test, m, cof)
                            for m in COMPARE_MODELS for cof in [0.0, DEFAULT_COF_RATE]], ignore_index=True)
    comparison.to_csv(TAB / "strategy_comparison.csv", index=False)

    # How sensitive is the chosen cutoff to the cost-of-funds assumption?
    sens = []
    for cof in [0.0, 0.02, 0.04, 0.06, 0.08]:
        r = compare_strategies(valid, test, DECISION_MODEL, cof).set_index("policy")
        sens.append({"cof_rate": cof,
                     "approval_chosen_on_2014": r.iloc[0].chosen_on_2014_approval,
                     "profit_approve_all": r.loc["Approve everyone"].total_profit,
                     "profit_pd_cutoff": r.loc["PD cutoff chosen on 2014"].total_profit,
                     "approval_pd_cutoff": r.loc["PD cutoff chosen on 2014"].approval_rate,
                     "profit_approval_rate_cutoff": r.loc["Approval-rate cutoff chosen on 2014"].total_profit,
                     "profit_best_possible": r.loc["Best possible 2015 cutoff (hindsight)"].total_profit,
                     "approval_best_possible": r.loc["Best possible 2015 cutoff (hindsight)"].approval_rate})
    sens = pd.DataFrame(sens)
    sens.to_csv(TAB / "strategy_cof_sensitivity.csv", index=False)

    deciles = pd.concat([profit_by_decile(d, DECISION_MODEL, cof).assign(split=split)
                         for split, d in [("valid", valid), ("test", test)] for cof in [0.0, DEFAULT_COF_RATE]])
    deciles.to_csv(TAB / "strategy_profit_by_decile.csv", index=False)

    plot_profit_curves(curves, comparison)
    plot_default_curve(curves, comparison)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    cols = ["model", "cof_rate", "policy", "approval_rate", "n_approved", "default_rate", "dollar_loss",
            "total_profit", "loss_cut_vs_all", "profit_vs_all", "chosen_on_2014_approval",
            "chosen_on_2014_pd_cutoff", "valid_approval_at_pd_cutoff"]
    print(comparison[cols].round(4).to_string(index=False), "\n")
    print(sens.round(4).to_string(index=False), "\n")
    print(deciles.round(4).to_string(index=False), "\n")
    for cof in [0.0, DEFAULT_COF_RATE]:
        print(f"--- {DECISION_MODEL}, cost of funds {cof:.0%}")
        for _, row in comparison[(comparison.model == DECISION_MODEL) & (comparison.cof_rate == cof)].iloc[1:].iterrows():
            print(business_sentence(row))


if __name__ == "__main__":
    main()
