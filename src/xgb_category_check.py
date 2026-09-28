"""Phase 3 side experiment (validation only): how should XGBoost split on categorical features?

The default partitions many levels (50 states) into arbitrary subsets per split, which can overfit.
Compared on 2014 validation: default partitioning vs one-vs-rest splits.
Result drives max_cat_to_onehot=64 in src/train.py.
(An earlier run, when addr_state was still a feature, also tested dropping it; state was later removed
from the model entirely - see src/state_check.py.)
"""
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.features import feature_columns, load_model_table, time_split
from src.train import TAB, xgb_frame


def main():
    s = time_split(load_model_table())
    tr, va = s["train"], s["valid"]
    rows = []
    for include_lc in [False, True]:
        num, cat = feature_columns(include_lc)
        for label, extra in [("native partition (default)", {}),
                             ("one-vs-rest", {"max_cat_to_onehot": 64})]:
            Xtr, cats = xgb_frame(tr, num, cat)
            Xva, _ = xgb_frame(va, num, cat, cats)
            m = XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=5, min_child_weight=50,
                              subsample=0.8, colsample_bytree=0.8, enable_categorical=True,
                              tree_method="hist", eval_metric="auc", early_stopping_rounds=100,
                              random_state=42, n_jobs=-1, **extra)
            m.fit(Xtr, tr.is_default, eval_set=[(Xva, va.is_default)], verbose=False)
            rows.append({"features": "with LC" if include_lc else "without LC", "category_handling": label,
                         "n_trees": m.best_iteration + 1,
                         "valid_roc_auc": roc_auc_score(va.is_default, m.predict_proba(Xva)[:, 1])})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "xgb_category_experiment.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
