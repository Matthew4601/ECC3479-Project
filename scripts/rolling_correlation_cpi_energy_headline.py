"""
Rolling 12-month correlation between CPI Energy Index and Headline CPI Index
for Canada, UK, and USA.

Sources:
  Clean 3/Domestic Consumption/CPI Energy Combined Index Final.csv
  Clean 3/Domestic Consumption/CPI_Combined_Final.csv

Output: Correlation Maps/CPI Energy vs CPI Headline Correlation.png
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT      = Path(__file__).resolve().parent.parent
ENERGY_F  = ROOT / "Clean 3" / "Domestic Consumption" / "CPI Energy Combined Index Final.csv"
HEADLINE_F= ROOT / "Clean 3" / "Domestic Consumption" / "CPI_Combined_Final.csv"
OUT_PATH  = ROOT / "Correlation Maps" / "CPI Energy vs CPI Headline Correlation.png"

COUNTRIES = {
    "CAN": {
        "energy_col":      "canada_cpi_energy_idx2015",
        "headline_series": "CPI All-items Index (Canada national, 2002=100)",
        "label":           "Canada",
        "abbrev":          "CAN",
        "color":           "#D62728",
        "annot_offset":    (40, 18),   # (x, y) offset in points for 2022 annotation
    },
    "UK": {
        "energy_col":      "uk_cpi_energy_idx2015",
        "headline_series": "CPIH Index, All Items 2015=100 (L522)",
        "label":           "United Kingdom",
        "abbrev":          "UK",
        "color":           "#1F77B4",
        "annot_offset":    (40, 4),
    },
    "USA": {
        "energy_col":      "usa_cpi_energy_idx2015",
        "headline_series": "CPI-U All Items Index (1982-84=100, FRED CPIAUCSL)",
        "label":           "United States",
        "abbrev":          "USA",
        "color":           "#2CA02C",
        "annot_offset":    (40, -12),
    },
}

WINDOW = 12


def build_rolling_corr(energy, headline, code, meta):
    en = energy[["date", meta["energy_col"]]].rename(
        columns={meta["energy_col"]: "cpi_energy"})
    hl = headline[
        (headline["country"] == code) &
        (headline["series"]  == meta["headline_series"]) &
        (headline["unit"]    == "index (2015=100)")
    ][["date", "value"]].rename(columns={"value": "cpi_headline"})

    df = en.merge(hl, on="date", how="inner").sort_values("date").set_index("date")
    df["rolling_corr"] = (
        df["cpi_energy"]
        .rolling(WINDOW)
        .corr(df["cpi_headline"])
    )
    return df.dropna(subset=["rolling_corr"])


def main():
    energy   = pd.read_csv(ENERGY_F,   parse_dates=["date"])
    headline = pd.read_csv(HEADLINE_F, parse_dates=["date"])

    fig, (ax_main, ax_dist) = plt.subplots(
        1, 2, figsize=(16, 6),
        gridspec_kw={"width_ratios": [3, 1]}
    )
    fig.suptitle(
        f"Rolling {WINDOW}-Month Correlation: CPI Energy vs Headline CPI by Country",
        fontsize=13, fontweight="bold"
    )

    country_data = {}
    for code, meta in COUNTRIES.items():
        df = build_rolling_corr(energy, headline, code, meta)
        country_data[code] = df

        ax_main.plot(
            df.index, df["rolling_corr"],
            color=meta["color"], linewidth=1.8,
            label=meta["label"], alpha=0.9, zorder=3
        )

    # 2022 shaded band
    ax_main.axvspan(
        pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"),
        color="gold", alpha=0.25, zorder=1, label="2022"
    )

    # Reference lines
    ax_main.axhline(1.0,  color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax_main.axhline(0.0,  color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax_main.axhline(-1.0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)

    # Annotate 2022 peak correlation — staggered offsets per country for legibility
    for code, meta in COUNTRIES.items():
        df     = country_data[code]
        yr2022 = df[df.index.year == 2022]
        if not yr2022.empty:
            peak_date = yr2022["rolling_corr"].idxmax()
            peak_val  = yr2022["rolling_corr"].max()
            ax_main.annotate(
                f"{meta['abbrev']}: {peak_val:.2f}",
                xy=(peak_date, peak_val),
                xytext=meta["annot_offset"], textcoords="offset points",
                fontsize=9, color=meta["color"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=meta["color"], lw=0.9),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=meta["color"],
                          alpha=0.85, lw=0.8)
            )

    ax_main.set_xlabel("Date", fontsize=11)
    ax_main.set_ylabel(f"Rolling {WINDOW}-Month Correlation", fontsize=11)
    ax_main.set_title("Correlation Over Time", fontsize=11)
    ax_main.set_ylim(-0.3, 1.15)
    ax_main.legend(fontsize=9.5, loc="lower left", framealpha=0.9)
    ax_main.grid(True, alpha=0.3)

    # ── Right panel: distribution of rolling correlations ───────────────────
    for code, meta in COUNTRIES.items():
        df   = country_data[code]
        vals = df["rolling_corr"].dropna()

        ax_dist.hist(
            vals, bins=30, orientation="horizontal",
            color=meta["color"], alpha=0.45, edgecolor="white", linewidth=0.3
        )
        median = vals.median()
        ax_dist.axhline(median, color=meta["color"], linewidth=1.8,
                        linestyle="--", alpha=0.9)
        ax_dist.text(
            0.97, median + 0.01,
            f"{meta['abbrev']} med={median:.2f}",
            transform=ax_dist.get_yaxis_transform(),
            ha="right", fontsize=8, color=meta["color"], fontweight="bold"
        )

    ax_dist.axhline(0.0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax_dist.set_xlabel("Frequency", fontsize=10)
    ax_dist.set_ylabel(f"Rolling {WINDOW}-Month Correlation", fontsize=10)
    ax_dist.set_title("Correlation Distribution\n(dashed = median)", fontsize=11)
    ax_dist.grid(True, alpha=0.3)
    ax_dist.set_ylim(ax_main.get_ylim())

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
