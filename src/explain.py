"""Phase 5a: top-3 decline reasons per applicant, in plain English.

Contributions come from SHAP values (XGBoost computes them exactly for tree models via
pred_contribs=True): for each loan, how much each feature pushed its predicted log-odds of default
above or below the average applicant. A decline reason = a feature that pushed risk UP.

Rules for what may be stated as a reason:
  * LendingClub's own pricing (sub-grade, interest rate, installment, payment-to-income) is never a
    reason: "your interest rate was high" is circular - the rate was set from the same risk judgement.
  * Only features that increased risk for this applicant (positive contribution).
  * The wording depends on whether the applicant's value is above or below the typical (training
    median) value, so the reason says which way the feature hurt.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.eda import BLUE, INK2, _save, plt
from src.features import LC_PRICE_DERIVED, load_model_table, time_split
from src.train import MODELS, TAB, xgb_frame

ROOT = Path(__file__).resolve().parents[1]
REASONS_PATH = ROOT / "data" / "processed" / "test_reasons.parquet"
DECISION_MODEL = "xgboost_with_lc"
NOT_A_REASON = set(LC_PRICE_DERIVED) | {"sub_grade_num", "int_rate"}

# feature -> (text if applicant is above the typical value, text if below)
NUMERIC_REASONS = {
    "fico_mid": ("Credit score", "Credit score too low"),
    "dti": ("Debt-to-income ratio too high", "Low debt-to-income ratio"),
    "annual_inc": ("Income level", "Income too low"),
    "loan_to_income": ("Loan amount too high relative to income", "Loan amount relative to income"),
    "loan_amnt": ("Requested loan amount too large", "Requested loan amount"),
    "revol_util": ("High use of available revolving credit", "Revolving credit utilization"),
    "revol_bal": ("High revolving credit balances", "Limited use of revolving credit"),
    "inq_last_6mths": ("Too many credit inquiries in the last 6 months", "Recent credit inquiries"),
    "credit_history_months": ("Length of credit history", "Limited length of credit history"),
    "emp_length_yrs": ("Length of employment", "Short time with current employer"),
    "open_acc": ("Too many open credit accounts", "Too few open credit accounts"),
    "total_acc": ("Number of credit accounts", "Limited number of credit accounts"),
    "mort_acc": ("Number of mortgage accounts", "No or few mortgage accounts"),
    "delinq_2yrs": ("Delinquencies in the last 2 years", "Delinquency history"),
    "pub_rec": ("Public records on credit file", "Public record history"),
    "pub_rec_bankruptcies": ("Bankruptcy on credit file", "Bankruptcy history"),
}
MISSING_REASONS = {
    "emp_length_yrs": "Employment length not provided",
    "mort_acc": "Mortgage account history not available",
    "revol_util": "Revolving credit utilization not available",
    "dti": "Debt-to-income ratio not available",
}


def categorical_reason(feature: str, value) -> str:
    if feature == "home_ownership":
        return {"RENT": "Rents rather than owns home", "OWN": "Housing status (owns home outright)",
                "MORTGAGE": "Housing status (mortgage)"}.get(value, "Housing status")
    if feature == "purpose":
        return f"Loan purpose ({str(value).replace('_', ' ')})"
    if feature == "verification_status":
        return {"Not Verified": "Income not verified"}.get(value, f"Income verification status ({value})")
    return feature


def shap_contributions(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    X, _ = xgb_frame(df, bundle["num"], bundle["cat"], bundle["categories"])
    contrib = bundle["model"].get_booster().predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)
    return pd.DataFrame(contrib[:, :-1], columns=X.columns, index=df.index)  # last column = bias


def reason_text(feature: str, value, typical) -> str:
    if feature in NUMERIC_REASONS:
        if pd.isna(value):
            return MISSING_REASONS.get(feature, f"{feature} not available")
        high, low = NUMERIC_REASONS[feature]
        return high if value > typical else low
    return categorical_reason(feature, value)


def top_reasons(df: pd.DataFrame, contrib: pd.DataFrame, medians: pd.Series, k: int = 3) -> pd.DataFrame:
    allowed = [c for c in contrib.columns if c not in NOT_A_REASON]
    c = contrib[allowed].to_numpy()
    order = np.argsort(-c, axis=1)[:, :k]
    rows = []
    for i, idx in enumerate(order):
        row = {}
        for rank, j in enumerate(idx, start=1):
            feat = allowed[j]
            if c[i, j] <= 0:  # only features that pushed risk up count as reasons
                row[f"reason_{rank}"], row[f"reason_{rank}_feature"] = None, None
                continue
            row[f"reason_{rank}"] = reason_text(feat, df.iloc[i][feat], medians.get(feat))
            row[f"reason_{rank}_feature"] = feat
        rows.append(row)
    return pd.DataFrame(rows, index=df.index)


def reason_frequency(reasons: pd.DataFrame, declined: np.ndarray) -> pd.DataFrame:
    """How often each reason appears in a declined applicant's top 3, and as the #1 reason."""
    d = reasons[declined]
    any3 = pd.concat([d[f"reason_{r}"] for r in (1, 2, 3)]).value_counts()
    first = d["reason_1"].value_counts()
    return (pd.DataFrame({"share_in_top3": any3 / len(d), "share_as_top1": first / len(d)})
              .fillna(0).sort_values("share_in_top3", ascending=False).rename_axis("reason").reset_index())


def plot_reasons(freq: pd.DataFrame, label: str, n_declined: int):
    f = freq.head(12)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(f.reason, f.share_in_top3, color=BLUE, height=0.6)
    ax.invert_yaxis(); ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    for b, v in zip(bars, f.share_in_top3):
        ax.text(b.get_width() + 0.005, b.get_y() + b.get_height() / 2, f"{v:.0%}", va="center", fontsize=9, color=INK2)
    ax.set_xlim(0, f.share_in_top3.max() * 1.15)
    ax.set_xlabel("Share of declined applicants with this reason in their top 3")
    ax.set_title(f"Most common decline reasons, 2015\n{label}: {n_declined:,} applicants declined", fontsize=11)
    _save(fig, "14_decline_reasons.png")


def main():
    splits = time_split(load_model_table())
    train, test = splits["train"], splits["test"].reset_index(drop=True)
    bundle = joblib.load(MODELS / f"{DECISION_MODEL}.joblib")
    medians = train[bundle["num"]].median()

    contrib = shap_contributions(bundle, test)
    reasons = top_reasons(test, contrib, medians)
    pd_score = bundle["model"].predict_proba(xgb_frame(test, bundle["num"], bundle["cat"], bundle["categories"])[0])[:, 1]
    out = pd.concat([test[["id"]], pd.Series(pd_score, name="pd"), reasons], axis=1)
    out.to_parquet(REASONS_PATH, index=False)

    # Sanity check: SHAP contributions + bias reproduce the model's log-odds exactly.
    X = xgb_frame(test.head(1000), bundle["num"], bundle["cat"], bundle["categories"])[0]
    full = bundle["model"].get_booster().predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)
    margin = bundle["model"].get_booster().predict(xgb.DMatrix(X, enable_categorical=True), output_margin=True)
    assert np.allclose(full.sum(axis=1), margin, atol=1e-4)

    # Which applicants are declined: the policy chosen on 2014 (4% cost of funds, approval-rate cutoff),
    # plus an 80% approval stress point so the reason mix is visible at a meaningful decline volume.
    comp = pd.read_csv(TAB / "strategy_comparison.csv")
    chosen = comp[(comp.model == DECISION_MODEL) & (comp.cof_rate == 0.04)].iloc[0].chosen_on_2014_approval
    rank = pd.Series(pd_score).rank(method="first") / len(pd_score)
    freqs = []
    for label, rate in [(f"policy chosen on 2014: approve {chosen:.0%}", chosen), ("stress test: approve 80%", 0.80)]:
        declined = (rank > rate).to_numpy()
        f = reason_frequency(reasons, declined).assign(operating_point=label, n_declined=int(declined.sum()))
        freqs.append(f)
    freqs = pd.concat(freqs, ignore_index=True)
    freqs.to_csv(TAB / "decline_reason_frequency.csv", index=False)
    stress = freqs[freqs.operating_point.str.startswith("stress")]
    plot_reasons(stress, "approve safest 80%", int(stress.n_declined.iloc[0]))

    # A few worked examples: the riskiest declined applicants.
    examples = out.sort_values("pd", ascending=False).head(5)[["id", "pd", "reason_1", "reason_2", "reason_3"]]
    examples.to_csv(TAB / "decline_reason_examples.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    print("chosen approval rate:", chosen)
    print(freqs.groupby("operating_point").head(12).round(3).to_string(index=False), "\n")
    print(examples.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
