"""
Individual rolling 12-month correlation charts: CPI Energy vs Headline CPI,
one panel per country (Canada, UK, USA).

Each panel shows:
  - Rolling 12-month correlation over time
  - 2022 shaded in gold
  - Annotated peak correlation value in 2022
  - Distribution histogram of all rolling correlations (inset)
  - Median and mean reference lines
  - Shaded +/- 1 std band around the mean

Sources:
  Clean 3/Domestic Consumption/CPI Energy Combined Index Final.csv
  Clean 3/Domestic Consumption/CPI_Combined_Final.csv

Output: Correlation Maps/Energy CPI vs CPI Headline Individual.png
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

ROOT      = Path(__file__).resolve().parent.parent
ENERGY_F  = ROOT / "Clean 3" / "Domestic Consumption" / "CPI Energy Combined Index Final.csv"
HEADLINE_F= ROOT / "Clean 3" / "Domestic Consumption" / "CPI_Combined_Final.csv"
OUT_PATH  = ROOT / "Correlation Maps" / "Energy CPI vs CPI Headline Individual.png"

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
        df["cpi_energy"].rolling(WINDOW).corr(df["cpi_headline"])
    )
    return df.dropna(subset=["rolling_corr"])


def main():
    energy   = pd.read_csv(ENERGY_F,   parse_dates=["date"])
    headline = pd.read_csv(HEADLINE_F, parse_dates=["date"])

    fig = plt.figure(figsize=(18, 15))
    fig.suptitle(
        f"Rolling {WINDOW}-Month Correlation: CPI Energy vs Headline CPI\n"
        "Individual Country Analysis",
        fontsize=14, fontweight="bold", y=0.98
    )

    outer = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45)

    for row_idx, (code, meta) in enumerate(COUNTRIES.items()):
        df   = build_rolling_corr(energy, headline, code, meta)
        vals = df["rolling_corr"].dropna()
        c    = meta["color"]

        # Each row: main time-series (left, wider) + histogram (right, narrower)
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[row_idx],
            width_ratios=[3, 1], wspace=0.08
        )
        ax_line = fig.add_subplot(inner[0])
        ax_hist = fig.add_subplot(inner[1])

        # ── Time-series panel ────────────────────────────────────────────────
        # Mean ± 1 std band
        mean_val = vals.mean()
        std_val  = vals.std()
        ax_line.axhspan(mean_val - std_val, mean_val + std_val,
                        color=c, alpha=0.08, zorder=1)

        # Rolling correlation line
        ax_line.plot(df.index, df["rolling_corr"],
                     color=c, linewidth=1.8, alpha=0.95, zorder=3)

        # Fill above/below zero
        ax_line.fill_between(df.index, df["rolling_corr"], 0,
                             where=df["rolling_corr"] >= 0,
                             color=c, alpha=0.12, zorder=2)
        ax_line.fill_between(df.index, df["rolling_corr"], 0,
                             where=df["rolling_corr"] < 0,
                             color="grey", alpha=0.2, zorder=2)

        # 2022 shaded band
        ax_line.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"),
                        color="gold", alpha=0.30, zorder=1, label="2022")

        # Reference lines
        for yval, ls in [(1.0, "--"), (0.0, "-"), (-1.0, "--")]:
            ax_line.axhline(yval, color="black", linewidth=0.6,
                            linestyle=ls, alpha=0.35)

        # Mean and median lines
        median_val = vals.median()
        ax_line.axhline(mean_val,   color=c, linewidth=1.2,
                        linestyle=":", alpha=0.8, label=f"Mean: {mean_val:.2f}")
        ax_line.axhline(median_val, color=c, linewidth=1.2,
                        linestyle=(0, (5, 2)), alpha=0.8,
                        label=f"Median: {median_val:.2f}")

        # Annotate 2022 peak
        yr2022 = df[df.index.year == 2022]
        if not yr2022.empty:
            peak_date = yr2022["rolling_corr"].idxmax()
            peak_val  = yr2022["rolling_corr"].max()
            ax_line.annotate(
                f"2022 peak: {peak_val:.3f}",
                xy=(peak_date, peak_val),
                xytext=(30, -22), textcoords="offset points",
                fontsize=9, color="darkred", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="darkred", lw=1.0),
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="darkred", alpha=0.9, lw=0.8)
            )

        # Annotate min correlation
        min_date = vals.idxmin()
        min_val  = vals.min()
        ax_line.annotate(
            f"Min: {min_val:.3f}",
            xy=(min_date, min_val),
            xytext=(30, 10), textcoords="offset points",
            fontsize=8, color="grey",
            arrowprops=dict(arrowstyle="->", color="grey", lw=0.7),
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec="grey", alpha=0.85, lw=0.6)
        )

        date_range = (f"{df.index.min().strftime('%Y-%m')} – "
                      f"{df.index.max().strftime('%Y-%m')}")
        ax_line.set_title(
            f"{meta['label']} ({meta['abbrev']})  |  {date_range}",
            fontsize=11, fontweight="bold", color=c, loc="left"
        )
        ax_line.set_xlabel("Date", fontsize=10)
        ax_line.set_ylabel(f"Rolling {WINDOW}-Month Correlation", fontsize=10)
        ax_line.set_ylim(-0.45, 1.15)
        ax_line.legend(fontsize=8.5, loc="lower left", framealpha=0.9, ncol=3)
        ax_line.grid(True, alpha=0.25)

        # ── Histogram panel ──────────────────────────────────────────────────
        ax_hist.hist(vals, bins=30, orientation="horizontal",
                     color=c, alpha=0.55, edgecolor="white", linewidth=0.3)

        ax_hist.axhline(median_val, color=c, linewidth=1.8,
                        linestyle=(0, (5, 2)), alpha=0.9)
        ax_hist.axhline(mean_val,   color=c, linewidth=1.8,
                        linestyle=":",       alpha=0.9)
        ax_hist.axhline(0.0, color="black", linewidth=0.6,
                        linestyle="-", alpha=0.35)

        # Labels inside histogram
        x_max = ax_hist.get_xlim()[1]
        ax_hist.text(0.97, median_val + 0.02,
                     f"Median: {median_val:.2f}",
                     transform=ax_hist.get_yaxis_transform(),
                     ha="right", fontsize=8, color=c, fontweight="bold")
        ax_hist.text(0.97, mean_val - 0.05,
                     f"Mean: {mean_val:.2f}",
                     transform=ax_hist.get_yaxis_transform(),
                     ha="right", fontsize=8, color=c)

        # Std annotation
        ax_hist.text(0.97, 0.04,
                     f"Std: {std_val:.2f}",
                     transform=ax_hist.transAxes,
                     ha="right", va="bottom", fontsize=8, color="grey")

        ax_hist.set_xlabel("Frequency", fontsize=10)
        ax_hist.set_ylabel("")
        ax_hist.set_title("Distribution", fontsize=10)
        ax_hist.set_ylim(-0.45, 1.15)
        ax_hist.yaxis.set_ticklabels([])
        ax_hist.grid(True, alpha=0.25)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
