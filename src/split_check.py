"""Phase 2: justify the time split with data.

1. Summarize each split (size, default rate, date range) and confirm no overlap.
2. Test the training window: does including 2007-2011 help or hurt on the 2014 validation set?
   Those years predate LendingClub collecting mort_acc and span the financial crisis.
   Only validation data is used here; the 2015 test set is not touched.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.eda import BLUE, FIG, INK2, MUTED, _pct, _save, plt
from src.features import SPLIT_YEARS, feature_columns, load_model_table, time_split

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "reports" / "tables"


def split_summary(splits: dict) -> pd.DataFrame:
    rows = []
    for name, d in splits.items():
        rows.append({
            "split": name,
            "first_issue": d.issue_date.min().date(),
            "last_issue": d.issue_date.max().date(),
            "n_loans": len(d),
            "share_of_total": np.nan,
            "n_defaults": int(d.is_default.sum()),
            "default_rate": d.is_default.mean(),
            "avg_net_profit": d.net_profit.mean(),
        })
    out = pd.DataFrame(rows)
    out["share_of_total"] = out.n_loans / out.n_loans.sum()
    return out


def quick_models(num, cat):
    # Deliberately untuned: this only compares training windows, not models.
    lr = make_pipeline(
        ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), cat),
        ]),
        LogisticRegression(max_iter=2000),
    )
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, enable_categorical=True, tree_method="hist",
                        random_state=42, n_jobs=-1)
    return lr, xgb


def window_experiment(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df.issue_year == 2014]
    rows = []
    for include_lc in [False, True]:
        num, cat = feature_columns(include_lc)
        for start in [2007, 2010, 2012]:
            train = df[df.issue_year.between(start, 2013)]
            lr, xgb = quick_models(num, cat)
            lr.fit(train[num + cat], train.is_default)
            p_lr = lr.predict_proba(valid[num + cat])[:, 1]
            Xtr = train[num + cat].astype({c: "category" for c in cat})
            # Categories never seen in training (e.g. state ND) become missing, explicitly.
            Xva = valid[num + cat].copy()
            for c in cat:
                known = Xtr[c].cat.categories
                Xva[c] = pd.Categorical(Xva[c].where(Xva[c].isin(known)), categories=known)
            xgb.fit(Xtr, train.is_default)
            p_xgb = xgb.predict_proba(Xva)[:, 1]
            for model, p in [("logistic", p_lr), ("xgboost", p_xgb)]:
                rows.append({
                    "features": "with LC grade/rate" if include_lc else "without LC grade/rate",
                    "train_years": f"{start}-2013", "n_train": len(train), "model": model,
                    "valid_roc_auc": roc_auc_score(valid.is_default, p),
                    "valid_pr_auc": average_precision_score(valid.is_default, p),
                    "valid_mean_pd": p.mean(), "valid_actual_dr": valid.is_default.mean(),
                })
    return pd.DataFrame(rows)


def plot_split(df: pd.DataFrame):
    """Quarterly default rate with the three split periods shaded."""
    q = (df.assign(q=df.issue_date.dt.to_period("Q").dt.start_time)
           .groupby("q").agg(n=("id", "size"), dr=("is_default", "mean")).reset_index())
    q = q[q.n >= 500]  # early quarters are too small to plot a stable rate
    fig, ax = plt.subplots(figsize=(10, 4.5))
    shades = {"train": "#eef3fb", "valid": "#fdf0ea", "test": "#eaf6f0"}
    for name, (lo, hi) in SPLIT_YEARS.items():
        ax.axvspan(pd.Timestamp(lo, 1, 1), pd.Timestamp(hi, 12, 31), color=shades[name], zorder=0)
        n = int(df.issue_year.between(lo, hi).sum())
        ax.text(pd.Timestamp(max(lo, 2010), 3, 1), 0.165, f"{name}\n{lo if lo == hi else f'{lo}–{hi}'}\nn={n:,}",
                fontsize=9, color=INK2, va="top")
    ax.plot(q.q, q.dr, color=BLUE, lw=2, marker="o", ms=4)
    _pct(ax); ax.set_ylim(0.08, 0.17)
    ax.set_xlim(pd.Timestamp(2009, 7, 1), pd.Timestamp(2016, 1, 31))
    ax.set_title("Quarterly default rate across the time split (quarters with ≥500 loans)")
    ax.set_xlabel("Issue quarter")
    _save(fig, "09_time_split.png")


def main():
    df = load_model_table()
    splits = time_split(df)

    # Overlap check (also enforced in tests/)
    ids = {k: set(v.id) for k, v in splits.items()}
    assert not (ids["train"] & ids["valid"]) and not (ids["train"] & ids["test"]) and not (ids["valid"] & ids["test"])
    assert splits["train"].issue_date.max() < splits["valid"].issue_date.min()
    assert splits["valid"].issue_date.max() < splits["test"].issue_date.min()

    plot_split(df)
    summary = split_summary(splits)
    summary.to_csv(TAB / "split_summary.csv", index=False)
    exp = window_experiment(df)
    exp.to_csv(TAB / "train_window_experiment.csv", index=False)

    pd.set_option("display.width", 200)
    print(summary.round(4).to_string(), "\n")
    print(exp.round(4).to_string())


if __name__ == "__main__":
    main()
