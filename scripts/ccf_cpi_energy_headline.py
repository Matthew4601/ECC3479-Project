"""
Cross-Correlation Function (CCF) plots: CPI Energy vs Headline CPI
at lags 0–12 months for Canada, UK, and USA.

The CCF measures how strongly energy CPI at time t predicts headline CPI
at time t+k (positive lag = energy leads headline).

Data are first-differenced (month-on-month change) before computing
correlations to remove trend non-stationarity from the index levels.

95% confidence bounds: ± 1.96 / sqrt(n)

Sources:
  Clean 3/Domestic Consumption/CPI Energy Combined Index Final.csv
  Clean 3/Domestic Consumption/CPI_Combined_Final.csv

Output: Correlation Maps/CPI Energy vs CPI Headline LAG.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

ROOT      = Path(__file__).resolve().parent.parent
ENERGY_F  = ROOT / "Clean 3" / "Domestic Consumption" / "CPI Energy Combined Index Final.csv"
HEADLINE_F= ROOT / "Clean 3" / "Domestic Consumption" / "CPI_Combined_Final.csv"
OUT_PATH  = ROOT / "Correlation Maps" / "CPI Energy vs CPI Headline LAG.png"

COUNTRIES = {
    "CAN": {
        "energy_col":      "canada_cpi_energy_idx2015",
        "headline_series": "CPI All-items Index (Canada national, 2002=100)",
        "label":           "Canada",
        "abbrev":          "CAN",
        "color":           "#D62728",
    },
    "UK": {
        "energy_col":      "uk_cpi_energy_idx2015",
        "headline_series": "CPIH Index, All Items 2015=100 (L522)",
        "label":           "United Kingdom",
        "abbrev":          "UK",
        "color":           "#1F77B4",
    },
    "USA": {
        "energy_col":      "usa_cpi_energy_idx2015",
        "headline_series": "CPI-U All Items Index (1982-84=100, FRED CPIAUCSL)",
        "label":           "United States",
        "abbrev":          "USA",
        "color":           "#2CA02C",
    },
}

MAX_LAG = 12


def build_series(energy, headline, code, meta):
    en = energy[["date", meta["energy_col"]]].rename(
        columns={meta["energy_col"]: "energy"})
    hl = headline[
        (headline["country"] == code) &
        (headline["series"]  == meta["headline_series"]) &
        (headline["unit"]    == "index (2015=100)")
    ][["date", "value"]].rename(columns={"value": "headline"})

    df = en.merge(hl, on="date", how="inner").sort_values("date")
    # First-difference to remove trend (month-on-month change)
    df["d_energy"]   = df["energy"].diff()
    df["d_headline"] = df["headline"].diff()
    return df.dropna()


def compute_ccf(x, y, max_lag):
    """
    Cross-correlation of x (energy) with y (headline) at lags 0..max_lag.
    Lag k means: corr(x[t], y[t+k]) — energy leads headline by k months.
    """
    n = len(x)
    x_norm = (x - x.mean()) / x.std()
    y_norm = (y - y.mean()) / y.std()
    ccf_vals = []
    p_vals   = []
    for lag in range(0, max_lag + 1):
        if lag == 0:
            r, p = stats.pearsonr(x_norm, y_norm)
        else:
            r, p = stats.pearsonr(x_norm.iloc[:-lag].values,
                                  y_norm.iloc[lag:].values)
        ccf_vals.append(r)
        p_vals.append(p)
    return np.array(ccf_vals), np.array(p_vals)


def main():
    energy   = pd.read_csv(ENERGY_F,   parse_dates=["date"])
    headline = pd.read_csv(HEADLINE_F, parse_dates=["date"])

    lags = np.arange(0, MAX_LAG + 1)
    ci   = None  # computed per country

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle(
        "Cross-Correlation Function (CCF): CPI Energy → Headline CPI\n"
        "Lags 0–12 months  |  Monthly first-differenced series  |  95% confidence bounds shown",
        fontsize=13, fontweight="bold"
    )

    for ax, (code, meta) in zip(axes, COUNTRIES.items()):
        df = build_series(energy, headline, code, meta)
        n  = len(df)
        ci_bound = 1.96 / np.sqrt(n)

        ccf_vals, p_vals = compute_ccf(
            df["d_energy"], df["d_headline"], MAX_LAG)

        c = meta["color"]

        # ── Bar chart of CCF values ──────────────────────────────────────────
        bar_colors = [c if p < 0.05 else "lightgrey"
                      for p in p_vals]
        bars = ax.bar(lags, ccf_vals, color=bar_colors,
                      edgecolor="white", linewidth=0.5,
                      width=0.65, zorder=3)

        # 95% CI bands
        ax.axhline( ci_bound, color="navy", linewidth=1.2,
                    linestyle="--", alpha=0.7, zorder=4,
                    label=f"95% CI (±{ci_bound:.3f})")
        ax.axhline(-ci_bound, color="navy", linewidth=1.2,
                    linestyle="--", alpha=0.7, zorder=4)
        ax.fill_between([-0.5, MAX_LAG + 0.5],
                        -ci_bound, ci_bound,
                        color="navy", alpha=0.06, zorder=2)

        # Zero reference
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5, zorder=4)

        # Annotate each bar with its correlation value
        for lag, val, p in zip(lags, ccf_vals, p_vals):
            va    = "bottom" if val >= 0 else "top"
            y_off = 0.008  if val >= 0 else -0.008
            weight = "bold" if p < 0.05 else "normal"
            ax.text(lag, val + y_off, f"{val:.2f}",
                    ha="center", va=va, fontsize=7.5,
                    color=c if p < 0.05 else "grey",
                    fontweight=weight, zorder=5)

        # Highlight the peak significant lag
        sig_mask = p_vals < 0.05
        if sig_mask.any():
            peak_lag = lags[sig_mask][np.argmax(np.abs(ccf_vals[sig_mask]))]
            peak_val = ccf_vals[peak_lag]
            ax.annotate(
                f"Peak: lag {peak_lag}\nr = {peak_val:.3f}",
                xy=(peak_lag, peak_val),
                xytext=(peak_lag + (1.5 if peak_lag < 8 else -3),
                        peak_val + (0.06 if peak_val > 0 else -0.06)),
                fontsize=8.5, color=c, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=c, lw=1.0),
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=c, alpha=0.9, lw=0.8)
            )

        # Legend patches: significant vs not
        sig_patch  = mpatches.Patch(color=c,           label="Significant (p < 0.05)")
        nsig_patch = mpatches.Patch(color="lightgrey",  label="Not significant")

        ax.set_title(
            f"{meta['label']} ({meta['abbrev']})\nn = {n} months",
            fontsize=11, fontweight="bold"
        )
        ax.set_xlabel("Lag (months)\nEnergy CPI leads Headline CPI →", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Cross-Correlation Coefficient", fontsize=10)
        ax.set_xticks(lags)
        ax.set_xlim(-0.6, MAX_LAG + 0.6)
        ax.set_ylim(-0.25, 0.85)
        ax.grid(True, axis="y", alpha=0.25, linestyle=":")
        ax.legend(
            handles=[sig_patch, nsig_patch,
                     mpatches.Patch(color="navy", alpha=0.4,
                                    label=f"95% CI (±{ci_bound:.3f})")],
            fontsize=8, loc="upper right", framealpha=0.9
        )

        # Sample period note
        ax.text(0.01, 0.02,
                f"{df['date'].min().strftime('%Y-%m')} – "
                f"{df['date'].max().strftime('%Y-%m')}",
                transform=ax.transAxes, fontsize=7.5,
                color="grey", va="bottom")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
