"""Phase 3: train logistic regression and XGBoost, each with and without LendingClub's risk outputs.

Everything is fit on 2007-2013 (train). Every choice - regularization strength, tree depth,
class weighting, number of trees - is made on 2014 (valid). 2015 (test) is only scored.
"""
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# Clipper lives in src.features so pickled models load from any entry point (app, evaluate).
from src.features import Clipper, feature_columns, load_model_table, time_split

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
TAB = ROOT / "reports" / "tables"
PRED_PATH = ROOT / "data" / "processed" / "predictions.parquet"

# Heavily right-skewed: a log makes their effect on log-odds closer to linear.
LOG_FEATURES = ["annual_inc", "revol_bal"]


def make_logistic(num: list[str], cat: list[str], C: float) -> Pipeline:
    logged = [c for c in num if c in LOG_FEATURES]
    plain = [c for c in num if c not in LOG_FEATURES]
    numeric_steps = lambda: [Clipper(), SimpleImputer(strategy="median", add_indicator=True), StandardScaler()]
    pre = ColumnTransformer([
        ("log", make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"), *numeric_steps()), logged),
        ("num", make_pipeline(*numeric_steps()), plain),
        # min_frequency groups rare levels (e.g. 'educational') so they don't get noisy coefficients
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100), cat),
    ])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(C=C, max_iter=3000))])


def xgb_frame(df: pd.DataFrame, num: list[str], cat: list[str], categories: dict | None = None):
    """XGBoost handles categoricals natively; unseen levels (e.g. state ND in 2015) become missing."""
    X = df[num + cat].copy()
    if categories is None:
        categories = {c: sorted(X[c].dropna().unique()) for c in cat}
    for c in cat:
        X[c] = pd.Categorical(X[c].where(X[c].isin(categories[c])), categories=categories[c])
    return X, categories


def tune_logistic(train, valid, num, cat):
    rows, best = [], None
    for C in [0.01, 0.1, 1.0]:
        m = make_logistic(num, cat, C).fit(train[num + cat], train.is_default)
        auc = roc_auc_score(valid.is_default, m.predict_proba(valid[num + cat])[:, 1])
        rows.append({"C": C, "valid_roc_auc": auc})
        if best is None or auc > best[0]:
            best = (auc, C, m)
    return best[2], {"C": best[1]}, pd.DataFrame(rows)


def tune_xgb(train, valid, num, cat):
    Xtr, cats = xgb_frame(train, num, cat)
    Xva, _ = xgb_frame(valid, num, cat, cats)
    ytr, yva = train.is_default, valid.is_default
    # Imbalance: compare no weighting vs weighting defaults up to balance the classes.
    balance = (ytr == 0).sum() / (ytr == 1).sum()
    grid = itertools.product([3, 4, 5, 6], [1, 50], [1.0, round(balance, 2)])
    rows, best = [], None
    for depth, mcw, spw in grid:
        m = XGBClassifier(
            n_estimators=2000, learning_rate=0.05, max_depth=depth, min_child_weight=mcw,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=spw,
            # One-vs-rest splits on categories: the default partitions 50 states into arbitrary
            # subsets and overfit on 2014 validation (-0.002 to -0.003 AUC).
            enable_categorical=True, max_cat_to_onehot=64, tree_method="hist", eval_metric="auc",
            early_stopping_rounds=100, random_state=42, n_jobs=-1,
        )
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        p = m.predict_proba(Xva)[:, 1]
        rows.append({"max_depth": depth, "min_child_weight": mcw, "scale_pos_weight": spw,
                     "best_n_trees": m.best_iteration + 1, "valid_roc_auc": roc_auc_score(yva, p),
                     "valid_mean_pd": p.mean()})
        if best is None or rows[-1]["valid_roc_auc"] > best[0]:
            best = (rows[-1]["valid_roc_auc"], rows[-1], m)
    params = {k: best[1][k] for k in ["max_depth", "min_child_weight", "scale_pos_weight", "best_n_trees"]}
    return best[2], params, cats, pd.DataFrame(rows).sort_values("valid_roc_auc", ascending=False)


def main():
    MODELS.mkdir(exist_ok=True)
    splits = time_split(load_model_table())
    train, valid, test = splits["train"], splits["valid"], splits["test"]

    preds, chosen = [], {}
    for include_lc in [False, True]:
        tag = "with_lc" if include_lc else "no_lc"
        num, cat = feature_columns(include_lc)

        lr, lr_params, lr_tab = tune_logistic(train, valid, num, cat)
        lr_tab.to_csv(TAB / f"tuning_logistic_{tag}.csv", index=False)
        joblib.dump({"model": lr, "num": num, "cat": cat}, MODELS / f"logistic_{tag}.joblib")

        xgb, xgb_params, cats, xgb_tab = tune_xgb(train, valid, num, cat)
        xgb_tab.to_csv(TAB / f"tuning_xgboost_{tag}.csv", index=False)
        joblib.dump({"model": xgb, "num": num, "cat": cat, "categories": cats}, MODELS / f"xgboost_{tag}.joblib")

        chosen[f"logistic_{tag}"] = lr_params
        chosen[f"xgboost_{tag}"] = xgb_params
        for name, d in [("valid", valid), ("test", test)]:
            p_lr = lr.predict_proba(d[num + cat])[:, 1]
            p_xgb = xgb.predict_proba(xgb_frame(d, num, cat, cats)[0])[:, 1]
            for model, p in [(f"logistic_{tag}", p_lr), (f"xgboost_{tag}", p_xgb)]:
                preds.append(pd.DataFrame({"id": d.id, "split": name, "model": model, "pd": p}))
        print(tag, "logistic", lr_params, "| xgboost", xgb_params)

    pd.concat(preds).to_parquet(PRED_PATH, index=False)
    (MODELS / "chosen_params.json").write_text(json.dumps(chosen, indent=2, default=float))


if __name__ == "__main__":
    main()
