"""Phase 5b: approval and default rates across income bands, home ownership, and state on 2015.

Two operating points:
  * the policy chosen on 2014 (4% cost of funds, approval-rate cutoff) - what would actually run
  * a stress test at 80% approval - the chosen policy declines very few people, so group gaps
    only become visible when the model declines a meaningful share

Limitation: the data has no protected attributes (race, sex, age, ...) and none are inferred. Income,
home ownership and state are not protected classes; they can correlate with protected traits, so
gaps here are a prompt for a proper fair-lending review, not a verdict either way.
"""
import duckdb
import pandas as pd

from src.eda import BLUE, INK2, MUTED, ORANGE, _pct, _save, plt
from src.explain import REASONS_PATH
from src.features import load_model_table, read_sql, time_split
from src.train import TAB


def decisions_table() -> pd.DataFrame:
    test = time_split(load_model_table())["test"]
    scores = pd.read_parquet(REASONS_PATH)[["id", "pd"]]
    d = test[["id", "annual_inc", "home_ownership", "addr_state", "is_default"]].merge(scores, on="id")
    comp = pd.read_csv(TAB / "strategy_comparison.csv")
    chosen = comp[(comp.model == "xgboost_with_lc") & (comp.cof_rate == 0.04)].iloc[0].chosen_on_2014_approval
    rank = d.pd.rank(method="first") / len(d)
    frames = []
    for label, rate in [(f"chosen on 2014 (approve {chosen:.0%})", chosen), ("stress test (approve 80%)", 0.80)]:
        frames.append(d.assign(operating_point=label, approved=rank <= rate))
    return pd.concat(frames, ignore_index=True)


def plot_fairness(res: pd.DataFrame):
    r = res[res.operating_point.str.startswith("stress")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.2, 0.8, 2.4]})
    for ax, dim, title in [(axes[0], "income_band", "Income band"), (axes[1], "home_ownership", "Home ownership")]:
        g = r[(r.dimension == dim) & (r.n_applicants >= 100)].sort_values("grp")
        labels = g.grp.str.replace(r"^\d: ", "", regex=True)
        ax.bar(labels, g.approval_rate, color=BLUE, width=0.6)
        for x, v in enumerate(g.approval_rate):
            ax.text(x, v + 0.01, f"{v:.0%}", ha="center", fontsize=8, color=INK2)
        ax.axhline(0.8 * g.approval_rate.max(), color=ORANGE, lw=1, ls="--")
        _pct(ax); ax.set_ylim(0, 1.05); ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", rotation=30)
    ax = axes[2]
    s = r[(r.dimension == "addr_state")].sort_values("approval_rate")
    colors = [MUTED if small else BLUE for small in s.small_group]
    # Dots, not bars: the axis does not start at zero, and bar length would exaggerate the gaps.
    ax.scatter(s.grp, s.approval_rate, color=colors, s=28, zorder=3)
    ax.grid(axis="x", visible=False)
    ax.axhline(0.8 * s[~s.small_group].approval_rate.max(), color=ORANGE, lw=1, ls="--")
    _pct(ax); ax.set_ylim(0.6, 0.9); ax.set_title("State (gray = fewer than 1,000 applicants)", fontsize=11)
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    fig.suptitle("Approval rate by group at 80% overall approval, 2015 (dashed = 4/5 of the best group)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    _save(fig, "15_fairness_approval.png")


def main():
    con = duckdb.connect()
    con.register("decisions", decisions_table())
    res = con.sql(read_sql("07_fairness.sql")).df()
    res.to_csv(TAB / "fairness_by_group.csv", index=False)
    plot_fairness(res)

    pd.set_option("display.width", 250)
    cols = ["dimension", "grp", "n_applicants", "approval_rate", "approval_ratio", "default_rate_all",
            "default_rate_approved", "mean_pd_approved", "underprediction_pp"]
    for op, g in res.groupby("operating_point"):
        print(f"=== {op}")
        print(g[g.dimension != "addr_state"][cols].round(4).to_string(index=False))
        st = g[(g.dimension == "addr_state") & ~g.small_group].sort_values("approval_rate")
        print(f"states with >=1000 applicants: {len(st)}; lowest approval:")
        print(st[cols].head(5).round(4).to_string(index=False))
        print("highest approval:")
        print(st[cols].tail(3).round(4).to_string(index=False))
        print("groups below 0.80 approval ratio (n>=1000):",
              g[(g.approval_ratio < 0.8) & ~g.small_group][["dimension", "grp", "approval_ratio"]].round(3).values.tolist(), "\n")


if __name__ == "__main__":
    main()
