"""
Scatter plots: Headline CPI Index vs Domestic Consumption Index over time,
one panel per country — EQUAL PERIOD VERSION.

Only the quarters shared by all three countries are included:
2007 Q1 – 2023 Q3 (67 quarters), binding constraints:
  - Start: USA consumption begins 2007 Q1
  - End:   Canada & UK consumption data ends 2023 Q3

Both series are indexed to 2015=100. CPI (monthly) is aggregated to
quarterly means before merging. Points coloured by year (plasma gradient);
2022 observations highlighted in red.

Sources:
  Clean 3/Domestic Consumption/Domestic_Consumption_Combined.csv
  Clean 3/Domestic Consumption/CPI_Combined_Final.csv

Output: Scatter Plots/CPI Headline vs Con EQUAL.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy import stats

ROOT     = Path(__file__).resolve().parent.parent
CON_F    = ROOT / "Clean 3" / "Domestic Consumption" / "Domestic_Consumption_Combined.csv"
CPI_F    = ROOT / "Clean 3" / "Domestic Consumption" / "CPI_Combined_Final.csv"
OUT_PATH = ROOT / "Scatter Plots" / "CPI Headline vs Con EQUAL.png"

# Common overlap period across all three countries
OVERLAP_START = pd.Timestamp("2007-01-01")
OVERLAP_END   = pd.Timestamp("2023-07-01")

COUNTRIES = {
    "CAN": {
        "cpi_series": "CPI All-items Index (Canada national, 2002=100)",
        "label":      "Canada",
        "abbrev":     "CAN",
        "color":      "#D62728",
        "marker":     "o",
    },
    "UK": {
        "cpi_series": "CPIH Index, All Items 2015=100 (L522)",
        "label":      "United Kingdom",
        "abbrev":     "UK",
        "color":      "#1F77B4",
        "marker":     "s",
    },
    "USA": {
        "cpi_series": "CPI-U All Items Index (1982-84=100, FRED CPIAUCSL)",
        "label":      "United States",
        "abbrev":     "USA",
        "color":      "#2CA02C",
        "marker":     "^",
    },
}


def to_quarter_start(date):
    qm = ((date.month - 1) // 3) * 3 + 1
    return pd.Timestamp(date.year, qm, 1)


def build_df(con_df, cpi_df, code, meta):
    # Consumption — quarterly, index (2015=100)
    con = con_df[con_df["country"] == code][["date", "value"]].rename(
        columns={"value": "consumption"})

    # CPI — aggregate monthly to quarterly mean
    cpi_raw = cpi_df[
        (cpi_df["country"] == code) &
        (cpi_df["series"]  == meta["cpi_series"]) &
        (cpi_df["unit"]    == "index (2015=100)")
    ][["date", "value"]].copy()
    cpi_raw["date"] = cpi_raw["date"].apply(to_quarter_start)
    cpi_q = cpi_raw.groupby("date", as_index=False)["value"].mean().rename(
        columns={"value": "cpi_headline"})

    df = con.merge(cpi_q, on="date", how="inner").sort_values("date")

    # Restrict to common overlap window
    df = df[(df["date"] >= OVERLAP_START) & (df["date"] <= OVERLAP_END)]
    df["year"] = df["date"].dt.year
    return df


def main():
    con_df = pd.read_csv(CON_F, parse_dates=["date"])
    cpi_df = pd.read_csv(CPI_F, parse_dates=["date"])

    datasets = {code: build_df(con_df, cpi_df, code, meta)
                for code, meta in COUNTRIES.items()}

    # Consistent colourbar across all panels
    year_min = min(df["year"].min() for df in datasets.values())
    year_max = max(df["year"].max() for df in datasets.values())
    norm = mcolors.Normalize(vmin=year_min, vmax=year_max)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Headline CPI Index vs Domestic Consumption Index — Equal Period Comparison\n"
        f"(Quarterly observations, 2015=100  |  Common period: "
        f"{OVERLAP_START.strftime('%Y Q1')} – {OVERLAP_END.strftime('%Y Q3')}  |  67 quarters)",
        fontsize=12, fontweight="bold"
    )

    for ax, (code, meta) in zip(axes, COUNTRIES.items()):
        df = datasets[code]
        c  = meta["color"]

        non_2022 = df[df["year"] != 2022]
        yr_2022  = df[df["year"] == 2022]

        # Scatter — non-2022
        ax.scatter(
            non_2022["cpi_headline"], non_2022["consumption"],
            c=non_2022["year"], cmap="plasma", norm=norm,
            s=32, alpha=0.80, linewidths=0, zorder=2,
            marker=meta["marker"]
        )

        # 2022 highlighted
        ax.scatter(
            yr_2022["cpi_headline"], yr_2022["consumption"],
            color="red", s=75, alpha=1.0, marker=meta["marker"],
            edgecolors="darkred", linewidths=0.8, zorder=4, label="2022"
        )

        # OLS regression line
        slope, intercept, r, p, _ = stats.linregress(
            df["cpi_headline"], df["consumption"])
        x_fit = np.linspace(df["cpi_headline"].min() - 2,
                            df["cpi_headline"].max() + 2, 200)
        ax.plot(x_fit, slope * x_fit + intercept,
                color=c, linewidth=1.8, linestyle="--", alpha=0.85, zorder=3,
                label=f"OLS  slope={slope:.3f}  R²={r**2:.3f}")

        # Annotate 2022
        if not yr_2022.empty:
            ax.annotate(
                "2022",
                xy=(yr_2022["cpi_headline"].iloc[0],
                    yr_2022["consumption"].iloc[0]),
                xytext=(8, 8), textcoords="offset points",
                fontsize=9, color="darkred", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="darkred", alpha=0.85, lw=0.7)
            )

        # Reference lines at 2015=100
        ax.axvline(100, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.axhline(100, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)

        ax.set_title(f"{meta['label']} ({meta['abbrev']})\n"
                     f"n = {len(df)} quarters",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Headline CPI Index (2015=100)", fontsize=10)
        ax.set_ylabel("Domestic Consumption Index (2015=100)", fontsize=10)
        ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
        ax.grid(True, alpha=0.28)

    # Colourbar to the left of the Canada panel
    plt.subplots_adjust(left=0.10, right=0.97, top=0.85,
                        bottom=0.10, wspace=0.30)
    sm = cm.ScalarMappable(cmap="plasma", norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.01, 0.15, 0.018, 0.65])
    cbar    = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cbar.set_label("Year", fontsize=10, labelpad=6)
    cbar.set_ticks([year_min, 2010, 2015, 2020, year_max])
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
