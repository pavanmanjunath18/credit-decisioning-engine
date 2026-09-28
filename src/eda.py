"""Phase 1: data checks and EDA. Writes tables to reports/tables/ and charts to reports/figures/."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.features import (
    CATEGORICAL_FEATURES, LC_PRICE_DERIVED, LC_RISK_COLS, NUMERIC_FEATURES, build_model_table, connect, read_sql,
)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
TAB = ROOT / "reports" / "tables"

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e6e5e0"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
})


def _pct(ax, axis="y"):
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=None)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def bar_default_rate(seg: pd.DataFrame, title: str, name: str, horizontal=False, overall=None):
    fig, ax = plt.subplots(figsize=(8, 4.5 if not horizontal else 6))
    labels, rates = seg["bucket"].astype(str), seg["default_rate"]
    if horizontal:
        bars = ax.barh(labels, rates, color=BLUE, height=0.6)
        ax.invert_yaxis()
        ax.grid(axis="x"); ax.grid(axis="y", visible=False)
        _pct(ax, "x")
        for b, r, n in zip(bars, rates, seg["n_loans"]):
            ax.text(b.get_width() + 0.003, b.get_y() + b.get_height() / 2,
                    f"{r:.1%}  (n={n:,})", va="center", fontsize=8, color=INK2)
        ax.set_xlim(0, rates.max() * 1.35)
        if overall is not None:
            ax.axvline(overall, color=MUTED, lw=1, ls="--")
    else:
        bars = ax.bar(labels, rates, color=BLUE, width=0.6)
        _pct(ax)
        for b, r in zip(bars, rates):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003, f"{r:.1%}",
                    ha="center", fontsize=8, color=INK2)
        ax.set_ylim(0, rates.max() * 1.18)
        if overall is not None:
            ax.axhline(overall, color=MUTED, lw=1, ls="--")
    ax.set_title(title)
    if overall is not None:
        ax.text(0.99, 0.97, f"dashed line = overall {overall:.1%}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=MUTED)
    _save(fig, name)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    loans = build_model_table()
    con = connect()

    # --- 1. Survivorship / maturity check -------------------------------------------------
    mat = con.sql(read_sql("01_maturity_check.sql")).df()
    mat.to_csv(TAB / "maturity_by_year_term.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    for term, color, dy in [(36, BLUE, -0.04), (60, ORANGE, 0.04)]:
        m = mat[mat.term_months == term]
        ax.plot(m.issue_year, m.share_unfinished, color=color, lw=2, marker="o", ms=5)
        last = m.iloc[-1]
        ax.text(last.issue_year + 0.2, last.share_unfinished + dy, f"{term}-mo", color=INK2, va="center")
    ax.set_xlim(2006.5, 2019.3)
    ax.axvspan(2006.5, 2015.5, color=BLUE, alpha=0.06)
    ax.text(2007, 0.93, "kept: 36-mo, 2007–2015", color=INK2, fontsize=9)
    _pct(ax); ax.set_ylim(0, 1)
    ax.set_title("Share of loans still unfinished at data snapshot (Mar-2019)")
    ax.set_xlabel("Issue year")

    ax = axes[1]
    m36 = mat[mat.term_months == 36]
    ax.plot(m36.issue_year, m36.default_rate_among_finished, color=BLUE, lw=2, marker="o", ms=5)
    ax.axvspan(2015.5, 2018.5, color=ORANGE, alpha=0.08)
    ax.text(2015.7, m36.default_rate_among_finished.max() * 0.97, "biased:\nmostly unfinished",
            color=INK2, fontsize=9, va="top")
    _pct(ax); ax.set_ylim(0, m36.default_rate_among_finished.max() * 1.1)
    ax.set_title("36-mo default rate, counting only finished loans")
    ax.set_xlabel("Issue year")
    _save(fig, "01_maturity_survivorship.png")

    # --- 2. Payment definition check -----------------------------------------------------
    pay = con.sql(read_sql("02_payment_check.sql")).df()
    pay.to_csv(TAB / "payment_includes_recoveries_check.csv", index=False)

    # --- 3. Missing values (model candidate columns) ------------------------------------
    cand = NUMERIC_FEATURES + LC_PRICE_DERIVED + CATEGORICAL_FEATURES + LC_RISK_COLS + ["zip3"]
    miss = (loans[cand].isna().mean().rename("missing_share").to_frame()
            .assign(n_missing=loans[cand].isna().sum()).sort_values("missing_share", ascending=False))
    miss.to_csv(TAB / "missing_values.csv")
    nz = miss[miss.missing_share > 0]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(nz) + 1.2))
    bars = ax.barh(nz.index, nz.missing_share, color=BLUE, height=0.6)
    ax.invert_yaxis(); ax.grid(axis="x"); ax.grid(axis="y", visible=False); _pct(ax, "x")
    for b, s, n in zip(bars, nz.missing_share, nz.n_missing):
        ax.text(b.get_width() + 0.001, b.get_y() + b.get_height() / 2, f"{s:.2%}  ({n:,})",
                va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, nz.missing_share.max() * 1.4)
    ax.set_title(f"Missing values in model inputs (n={len(loans):,} loans)")
    _save(fig, "02_missing_values.png")

    # --- 4. Class balance ------------------------------------------------------------------
    bal = loans["loan_status"].value_counts().rename("n_loans").to_frame()
    bal["share"] = bal.n_loans / bal.n_loans.sum()
    bal.to_csv(TAB / "class_balance.csv")
    overall = loans["is_default"].mean()

    # --- 5. Default rate by segment (SQL) -----------------------------------------------
    seg = con.sql(read_sql("04_default_rate_by_segment.sql")).df()
    seg.to_csv(TAB / "default_rate_by_segment.csv", index=False)
    s = lambda d: seg[seg.dimension == d].reset_index(drop=True)
    bar_default_rate(s("issue_year"), "Default rate by vintage (issue year), 36-mo loans",
                     "03_default_by_vintage.png", overall=overall)
    bar_default_rate(s("grade"), "Default rate by LendingClub grade", "04_default_by_grade.png",
                     overall=overall)
    bar_default_rate(s("fico_band"), "Default rate by FICO band (midpoint of range)",
                     "05_default_by_fico.png", overall=overall)
    bar_default_rate(s("dti_band").query("bucket != 'missing'"), "Default rate by debt-to-income band",
                     "06_default_by_dti.png", overall=overall)
    bar_default_rate(s("purpose"), "Default rate by loan purpose", "07_default_by_purpose.png",
                     horizontal=True, overall=overall)

    # --- 6. Term (on raw data: model table is 36-mo only) -------------------------------
    term = con.sql(read_sql("05_default_rate_by_term.sql")).df()
    term.to_csv(TAB / "default_rate_by_term.csv", index=False)
    term_plot = term.assign(bucket=term.term_months.astype(str) + "-mo\n(issued "
                            + term.first_year.astype(str) + "–" + term.last_year.astype(str) + ")")
    bar_default_rate(term_plot, "Default rate by term (matured loans only)", "08_default_by_term.png")

    # --- Console summary ----------------------------------------------------------------
    pd.set_option("display.width", 160)
    print(f"Model table: {len(loans):,} loans, {loans.issue_year.min()}–{loans.issue_year.max()}")
    print(bal, "\n")
    print(miss[miss.missing_share > 0], "\n")
    print(seg.to_string(), "\n")
    print(term.to_string())


if __name__ == "__main__":
    main()
