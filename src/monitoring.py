"""Phase 5c: population stability index (PSI) - has the applicant population shifted since training?

PSI = sum over bins of (actual% - expected%) * ln(actual% / expected%).
Bins are the training deciles for numbers (so each holds 10% of training loans) and the levels for
categories. Conventional reading: < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 major shift.

PSI needs no outcomes, so in production it can run the day applications arrive - long before
anyone knows who defaults. That is why it is the first line of monitoring.
"""
import joblib
import numpy as np
import pandas as pd

from src.eda import BLUE, INK2, MUTED, ORANGE, _save, plt
from src.features import load_model_table, time_split
from src.train import MODELS, TAB, xgb_frame

DECISION_MODEL = "xgboost_with_lc"
EPS = 1e-4  # keeps ln() finite when a bin is empty in one period


def psi_from_shares(expected: pd.Series, actual: pd.Series) -> float:
    e, a = expected.clip(lower=EPS), actual.clip(lower=EPS)
    return float(((a - e) * np.log(a / e)).sum())


def psi_numeric(train: pd.Series, other: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.nanquantile(train, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    def shares(x):
        # Missing values get their own bin: a jump in missingness is itself a drift signal.
        b = pd.cut(x, edges, include_lowest=True).astype(str).where(x.notna(), "missing")
        return b.value_counts(normalize=True)
    e, a = shares(train), shares(other)
    idx = e.index.union(a.index)
    return psi_from_shares(e.reindex(idx, fill_value=0), a.reindex(idx, fill_value=0))


def psi_categorical(train: pd.Series, other: pd.Series) -> float:
    e = train.fillna("missing").value_counts(normalize=True)
    a = other.fillna("missing").value_counts(normalize=True)
    idx = e.index.union(a.index)
    return psi_from_shares(e.reindex(idx, fill_value=0), a.reindex(idx, fill_value=0))


def flag(v: float) -> str:
    return "major shift" if v > 0.25 else "moderate shift" if v > 0.10 else "stable"


def main():
    splits = time_split(load_model_table())
    bundle = joblib.load(MODELS / f"{DECISION_MODEL}.joblib")
    num, cat = bundle["num"], bundle["cat"]
    for name, d in splits.items():
        d["score"] = bundle["model"].predict_proba(xgb_frame(d, num, cat, bundle["categories"])[0])[:, 1]

    gain = pd.read_csv(TAB / "xgboost_feature_importance.csv")
    gain = gain[gain.model == DECISION_MODEL].set_index("feature").gain_share

    rows = []
    for feat in ["score"] + num + cat:
        fn = psi_categorical if feat in cat else psi_numeric
        row = {"feature": feat, "gain_share": gain.get(feat, np.nan)}
        for other in ["valid", "test"]:
            row[f"psi_train_vs_{other}"] = fn(splits["train"][feat], splits[other][feat])
        row["flag_test"] = flag(row["psi_train_vs_test"])
        rows.append(row)
    out = pd.DataFrame(rows)
    out = pd.concat([out.iloc[:1], out.iloc[1:].sort_values("gain_share", ascending=False)])
    out.to_csv(TAB / "psi.csv", index=False)

    # Chart: model score + the 10 most important features
    top = out.head(11).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(top))
    ax.barh(y + 0.2, top.psi_train_vs_valid, height=0.38, color=MUTED, label="Train → 2014 valid")
    ax.barh(y - 0.2, top.psi_train_vs_test, height=0.38, color=BLUE, label="Train → 2015 test")
    ax.set_yticks(y, [f"{f}  (model score)" if f == "score" else f for f in top.feature])
    for lim, txt in [(0.10, "0.10 moderate"), (0.25, "0.25 major")]:
        ax.axvline(lim, color=ORANGE, lw=1, ls="--")
        ax.text(lim, len(top) - 0.3, txt, color=INK2, fontsize=8, ha="left")
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Population stability index vs 2007–2013 training loans")
    ax.set_title("Drift: model score and top-10 features (by XGBoost gain)")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, "16_psi.png")

    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
