"""Phase 3: score every model on validation (2014) and test (2015).

Metrics: ROC-AUC and KS (ranking), PR-AUC (ranking focused on defaults), Brier (probability accuracy),
plus mean predicted PD vs actual default rate and calibration curves.
Accuracy is not reported: with 86% of loans paid off, "approve everyone" is 86% accurate and useless.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

from src.eda import BLUE, FIG, INK2, MUTED, ORANGE, _pct, _save, plt
from src.features import load_model_table, time_split
from src.train import MODELS, PRED_PATH, TAB

MODEL_LABELS = {
    "lc_subgrade_lookup": "LendingClub sub-grade only (benchmark)",
    "logistic_no_lc": "Logistic, without LC grade/rate",
    "xgboost_no_lc": "XGBoost, without LC grade/rate",
    "logistic_with_lc": "Logistic, with LC grade/rate",
    "xgboost_with_lc": "XGBoost, with LC grade/rate",
}


def ks_statistic(y, p) -> float:
    """Max gap between the cumulative score distributions of defaulters and non-defaulters."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def metrics(y, p) -> dict:
    brier = brier_score_loss(y, p)
    # Raw Brier is dominated by the base rate, so also compare to a hard baseline: predicting the
    # period's actual default rate for every loan (which even "knows" the future average).
    base = np.mean(y) * (1 - np.mean(y))
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "ks": ks_statistic(y, p),
        "brier": brier,
        "brier_skill": 1 - brier / base,
        "mean_pd": float(np.mean(p)),
        "actual_default_rate": float(np.mean(y)),
    }


def subgrade_benchmark(splits) -> pd.DataFrame:
    """LendingClub's own pricing as a model: PD = the 2007-2013 default rate of the loan's sub-grade."""
    lookup = splits["train"].groupby("sub_grade").is_default.mean()
    rows = []
    for name in ["valid", "test"]:
        d = splits[name]
        rows.append(pd.DataFrame({"id": d.id, "split": name, "model": "lc_subgrade_lookup",
                                  "pd": d.sub_grade.map(lookup).fillna(splits["train"].is_default.mean())}))
    return pd.concat(rows)


def calibration_table(y, p, bins=10) -> pd.DataFrame:
    """Group loans into equal-size bins of predicted PD and compare predicted vs actual default rate."""
    d = pd.DataFrame({"y": y, "p": p})
    d["bin"] = pd.qcut(d.p, bins, labels=False, duplicates="drop")
    return d.groupby("bin").agg(mean_pd=("p", "mean"), actual_rate=("y", "mean"), n=("y", "size")).reset_index()


def plot_calibration(scored: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    for i, split in enumerate(["valid", "test"]):
        for j, tag in enumerate(["no_lc", "with_lc"]):
            ax = axes[i, j]
            ax.plot([0, 0.45], [0, 0.45], color=MUTED, lw=1, ls="--")
            for model, color in [(f"logistic_{tag}", BLUE), (f"xgboost_{tag}", ORANGE)]:
                s = scored[(scored.split == split) & (scored.model == model)]
                c = calibration_table(s.is_default, s.pd)
                ax.plot(c.mean_pd, c.actual_rate, color=color, lw=2, marker="o", ms=5,
                        label="Logistic" if model.startswith("logistic") else "XGBoost")
            ax.set_title(f"{'Validation 2014' if split == 'valid' else 'Test 2015'} · "
                         f"{'without' if tag == 'no_lc' else 'with'} LC grade/rate", fontsize=11)
            ax.grid(axis="x")
            _pct(ax); _pct(ax, "x")
            ax.set_xlim(0, 0.45); ax.set_ylim(0, 0.45)
            if i == 1: ax.set_xlabel("Predicted default rate (decile mean)")
            if j == 0: ax.set_ylabel("Actual default rate")
            ax.legend(frameon=False, loc="upper left")
    fig.suptitle("Calibration: dots above the dashed line = model under-predicts risk",
                 x=0.02, ha="left", fontsize=12, fontweight="bold")
    _save(fig, "10_calibration.png")


def plot_auc_comparison(table: pd.DataFrame):
    t = table[table.split == "test"].set_index("model").loc[list(MODEL_LABELS)]
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [MUTED] + [BLUE] * 2 + [ORANGE] * 2
    bars = ax.barh([MODEL_LABELS[m] for m in t.index], t.roc_auc, color=colors, height=0.6)
    ax.invert_yaxis(); ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    for b, v in zip(bars, t.roc_auc):
        ax.text(b.get_width() + 0.002, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center", fontsize=9, color=INK2)
    ax.set_xlim(0.5, t.roc_auc.max() + 0.03)
    ax.set_xlabel("ROC-AUC on 2015 test set (0.5 = random)")
    ax.set_title("Model ranking power vs LendingClub's own sub-grade")
    _save(fig, "11_auc_comparison.png")


def logistic_coefficients() -> pd.DataFrame:
    bundle = joblib.load(MODELS / "logistic_no_lc.joblib")
    pipe = bundle["model"]
    names = pipe.named_steps["pre"].get_feature_names_out()
    coef = pipe.named_steps["lr"].coef_[0]
    out = pd.DataFrame({"feature": names, "coef_per_sd": coef, "odds_ratio": np.exp(coef)})
    return out.reindex(out.coef_per_sd.abs().sort_values(ascending=False).index)


def xgb_importance() -> pd.DataFrame:
    """Share of total split gain per feature: which inputs the trees lean on most."""
    rows = []
    for tag in ["no_lc", "with_lc"]:
        booster = joblib.load(MODELS / f"xgboost_{tag}.joblib")["model"].get_booster()
        gain = pd.Series(booster.get_score(importance_type="total_gain"))
        rows.append(pd.DataFrame({"model": f"xgboost_{tag}", "feature": gain.index,
                                  "gain_share": (gain / gain.sum()).values}))
    return pd.concat(rows).sort_values(["model", "gain_share"], ascending=[True, False])


def main():
    df = load_model_table()
    splits = time_split(df)
    preds = pd.concat([pd.read_parquet(PRED_PATH), subgrade_benchmark(splits)])
    scored = preds.merge(df[["id", "is_default"]], on="id")

    rows = []
    for (model, split), g in scored.groupby(["model", "split"]):
        rows.append({"model": model, "split": split, **metrics(g.is_default, g.pd)})
    table = pd.DataFrame(rows)
    table["model"] = pd.Categorical(table.model, list(MODEL_LABELS))
    table = table.sort_values(["split", "model"], ascending=[False, True])
    table.to_csv(TAB / "model_metrics.csv", index=False)

    cal = []
    for (model, split), g in scored.groupby(["model", "split"]):
        cal.append(calibration_table(g.is_default, g.pd).assign(model=model, split=split))
    pd.concat(cal).to_csv(TAB / "calibration_deciles.csv", index=False)

    coefs = logistic_coefficients()
    coefs.to_csv(TAB / "logistic_no_lc_coefficients.csv", index=False)

    imp = xgb_importance()
    imp.to_csv(TAB / "xgboost_feature_importance.csv", index=False)

    plot_calibration(scored)
    plot_auc_comparison(table)

    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False), "\n")
    print(coefs.head(15).round(3).to_string(index=False), "\n")
    print(imp.groupby("model").head(8).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
