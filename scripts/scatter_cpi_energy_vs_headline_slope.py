"""
Slope comparison: CPI Energy Index vs Headline CPI Index for Canada, UK, and USA.

Two-panel figure:
  Left  — Scatter of all three countries on shared axes, with OLS regression
           lines overlaid. 2022 observations highlighted per country.
  Right — Slope coefficients with 95% confidence intervals plotted as a
           dot-and-whisker chart for direct visual comparison.

Sources:
  Clean 3/Domestic Consumption/CPI Energy Combined Index Final.csv
  Clean 3/Domestic Consumption/CPI_Combined_Final.csv

Output: Scatter Plots/CPI Energy vs CPI Headline Slope.png
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
OUT_PATH  = ROOT / "Scatter Plots" / "CPI Energy vs CPI Headline Slope.png"

COUNTRIES = {
    "CAN": {
        "energy_col":      "canada_cpi_energy_idx2015",
        "headline_series": "CPI All-items Index (Canada national, 2002=100)",
        "label":           "Canada",
        "color":           "#D62728",
        "marker":          "o",
    },
    "UK": {
        "energy_col":      "uk_cpi_energy_idx2015",
        "headline_series": "CPIH Index, All Items 2015=100 (L522)",
        "label":           "United Kingdom",
        "color":           "#1F77B4",
        "marker":          "s",
    },
    "USA": {
        "energy_col":      "usa_cpi_energy_idx2015",
        "headline_series": "CPI-U All Items Index (1982-84=100, FRED CPIAUCSL)",
        "label":           "United States",
        "color":           "#2CA02C",
        "marker":          "^",
    },
}


def build_country_df(energy, headline, code, meta):
    en = energy[["date", meta["energy_col"]]].rename(
        columns={meta["energy_col"]: "cpi_energy"})
    hl = headline[
        (headline["country"] == code) &
        (headline["series"]  == meta["headline_series"]) &
        (headline["unit"]    == "index (2015=100)")
    ][["date", "value"]].rename(columns={"value": "cpi_headline"})
    df = en.merge(hl, on="date", how="inner").dropna().sort_values("date")
    df["year"] = df["date"].dt.year
    return df


def ols_with_ci(x, y, confidence=0.95):
    """Return slope, intercept, slope_ci_low, slope_ci_high, r_squared."""
    slope, intercept, r, p, se = stats.linregress(x, y)
    n     = len(x)
    dof   = n - 2
    t_crit = stats.t.ppf((1 + confidence) / 2, dof)
    ci    = t_crit * se
    return slope, intercept, slope - ci, slope + ci, r ** 2


def main():
    energy   = pd.read_csv(ENERGY_F,   parse_dates=["date"])
    headline = pd.read_csv(HEADLINE_F, parse_dates=["date"])

    fig, (ax_scatter, ax_slope) = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={"width_ratios": [2, 1]}
    )
    fig.suptitle(
        "CPI Energy vs Headline CPI — Slope Comparison by Country\n(Monthly observations, 2015=100)",
        fontsize=13, fontweight="bold"
    )

    results = {}
    x_global_min, x_global_max = np.inf, -np.inf

    for code, meta in COUNTRIES.items():
        df = build_country_df(energy, headline, code, meta)
        slope, intercept, ci_lo, ci_hi, r2 = ols_with_ci(
            df["cpi_energy"], df["cpi_headline"])
        results[code] = dict(slope=slope, intercept=intercept,
                             ci_lo=ci_lo, ci_hi=ci_hi, r2=r2, df=df)
        x_global_min = min(x_global_min, df["cpi_energy"].min())
        x_global_max = max(x_global_max, df["cpi_energy"].max())

    # ── Left panel: scatter + regression lines ──────────────────────────────
    x_line = np.linspace(x_global_min - 5, x_global_max + 5, 300)

    for code, meta in COUNTRIES.items():
        res = results[code]
        df  = res["df"]
        c   = meta["color"]

        # Background scatter — non-2022
        non_2022 = df[df["year"] != 2022]
        ax_scatter.scatter(
            non_2022["cpi_energy"], non_2022["cpi_headline"],
            color=c, s=12, alpha=0.25, marker=meta["marker"], zorder=2
        )

        # 2022 highlighted
        yr_2022 = df[df["year"] == 2022]
        ax_scatter.scatter(
            yr_2022["cpi_energy"], yr_2022["cpi_headline"],
            color=c, s=55, alpha=1.0, marker=meta["marker"],
            edgecolors="black", linewidths=0.6, zorder=4
        )

        # Regression line
        y_line = res["slope"] * x_line + res["intercept"]
        ax_scatter.plot(
            x_line, y_line, color=c, linewidth=2.2, zorder=3,
            label=f"{meta['label']}  (slope={res['slope']:.3f}, R²={res['r2']:.3f})"
        )

    # 2022 marker legend entry
    ax_scatter.scatter([], [], color="grey", s=55, edgecolors="black",
                       linewidths=0.6, label="● 2022 observations")

    ax_scatter.axvline(100, color="grey", linestyle="--", linewidth=0.7, alpha=0.5)
    ax_scatter.axhline(100, color="grey", linestyle="--", linewidth=0.7, alpha=0.5)
    ax_scatter.set_xlabel("CPI Energy Index (2015=100)", fontsize=11)
    ax_scatter.set_ylabel("Headline CPI Index (2015=100)", fontsize=11)
    ax_scatter.set_title("Scatter with OLS Regression Lines", fontsize=11)
    ax_scatter.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    ax_scatter.grid(True, alpha=0.3)

    # ── Right panel: slope coefficients + 95% CI ────────────────────────────
    country_labels = [COUNTRIES[c]["label"] for c in COUNTRIES]
    slopes  = [results[c]["slope"]  for c in COUNTRIES]
    ci_los  = [results[c]["ci_lo"]  for c in COUNTRIES]
    ci_his  = [results[c]["ci_hi"]  for c in COUNTRIES]
    colors  = [COUNTRIES[c]["color"] for c in COUNTRIES]
    y_pos   = np.arange(len(COUNTRIES))

    for i, (code, s, lo, hi, col) in enumerate(
            zip(COUNTRIES, slopes, ci_los, ci_his, colors)):
        ax_slope.plot([lo, hi], [i, i], color=col, linewidth=2.5, zorder=2)
        ax_slope.scatter(s, i, color=col, s=120, zorder=3,
                         edgecolors="black", linewidths=0.7)
        ax_slope.text(hi + 0.003, i, f"{s:.3f}", va="center",
                      fontsize=9.5, color=col, fontweight="bold")

    ax_slope.set_yticks(y_pos)
    ax_slope.set_yticklabels(country_labels, fontsize=10)
    ax_slope.set_xlabel("OLS Slope Coefficient\n(Δ Headline CPI per Δ Energy CPI)", fontsize=10)
    ax_slope.set_title("Slope Coefficients\nwith 95% Confidence Intervals", fontsize=11)
    ax_slope.axvline(0, color="black", linewidth=0.8, alpha=0.4)
    ax_slope.grid(True, axis="x", alpha=0.3)
    ax_slope.set_xlim(min(ci_los) - 0.05, max(ci_his) + 0.06)

    # R² annotations inside right panel
    for i, code in enumerate(COUNTRIES):
        ax_slope.text(min(ci_los) - 0.045, i,
                      f"R²={results[code]['r2']:.3f}",
                      va="center", ha="left", fontsize=8, color="grey")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
