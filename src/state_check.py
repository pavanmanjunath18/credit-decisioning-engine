"""Phase 5 decision: does the decision model need addr_state?

Decline reasons must reflect what drives the decision, and "your state of residence" is not a
defensible reason. So instead of hiding state from the reasons, test dropping it from the model.
Pre-committed rule (set before looking at results): drop state if, on 2015,
  * test ROC-AUC falls by at most 0.002, and
  * profit at 4% cost of funds, using the cutoff chosen on 2014, falls by at most $1M.
The no-state model is re-tuned on 2014 with the same grid as the original.
"""
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.features import feature_columns, load_model_table, time_split
from src.strategy import DEFAULT_COF_RATE, compare_strategies, curve, load_scored
from src.train import TAB, tune_xgb, xgb_frame

AUC_TOLERANCE = 0.002
PROFIT_TOLERANCE = 1_000_000


def main():
    splits = time_split(load_model_table())
    train, valid, test = splits["train"], splits["valid"], splits["test"]
    num, cat = feature_columns(include_lc_risk=True)
    # Both variants are built explicitly, so this check reproduces after state left the feature list:
    # "with state" adds addr_state back; "no state" is the final feature set. Each is tuned on 2014.
    variants = {
        "xgboost_with_lc_with_state": [c for c in cat if c != "addr_state"] + ["addr_state"],
        "xgboost_with_lc_no_state": [c for c in cat if c != "addr_state"],
    }
    scored = load_scored()
    for name, cats_used in variants.items():
        model, params, cats, _ = tune_xgb(train, valid, num, cats_used)
        print(name, "tuned params:", params)
        for split in ["valid", "test"]:
            full = splits[split]
            p = pd.Series(model.predict_proba(xgb_frame(full, num, cats_used, cats)[0])[:, 1], index=full.id)
            scored[split][name] = scored[split].id.map(p).to_numpy()

    rows = []
    for m in variants:
        row = {"model": m}
        for name in ["valid", "test"]:
            d = scored[name]
            row[f"{name}_roc_auc"] = roc_auc_score(d.is_default, d[m])
        comp = compare_strategies(scored["valid"], scored["test"], m, DEFAULT_COF_RATE).set_index("policy")
        for policy, key in [("PD cutoff chosen on 2014", "profit_pd_cutoff"),
                            ("Approval-rate cutoff chosen on 2014", "profit_approval_cutoff"),
                            ("Model, same approval rate as E-G rule", "profit_same_volume_as_EG"),
                            ("Best possible 2015 cutoff (hindsight)", "profit_hindsight")]:
            row[key] = comp.loc[policy, "total_profit"]
        row["approval_chosen_on_2014"] = comp.iloc[0].chosen_on_2014_approval
        t = curve(scored["test"], m, DEFAULT_COF_RATE).set_index("target_approval")
        row["profit_at_84pct"] = t.loc[0.84, "total_profit"]
        rows.append(row)
    out = pd.DataFrame(rows)
    delta = out.iloc[1].drop("model") - out.iloc[0].drop("model")
    out = pd.concat([out, pd.DataFrame([{"model": "change (no_state - with_state)", **delta}])], ignore_index=True)
    out.to_csv(TAB / "state_ablation.csv", index=False)

    auc_drop = -delta["test_roc_auc"]
    profit_drop = -min(delta["profit_pd_cutoff"], delta["profit_approval_cutoff"])
    small = auc_drop <= AUC_TOLERANCE and profit_drop <= PROFIT_TOLERANCE
    pd.set_option("display.width", 250)
    print(out.T.to_string(), "\n")
    print(f"AUC drop {auc_drop:.4f} (tolerance {AUC_TOLERANCE}); worst profit drop ${profit_drop:,.0f} "
          f"(tolerance ${PROFIT_TOLERANCE:,.0f}) -> {'SMALL: drop state' if small else 'LARGE: review before deciding'}")


if __name__ == "__main__":
    main()
