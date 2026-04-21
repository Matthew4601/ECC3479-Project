"""
Scatter plots: Headline CPI Index vs Domestic Consumption Index over time,
one panel per country (Canada, UK, USA).

Both series are indexed to 2015=100, making them directly comparable.
CPI is monthly and is aggregated to quarterly means to match the
Domestic Consumption frequency before merging.

Points are coloured by year (plasma gradient). 2022 observations are
highlighted in red with a labelled annotation.

Sources:
  Clean 3/Domestic Consumption/Domestic_Consumption_Combined.csv
  Clean 3/Domestic Consumption/CPI_Combined_Final.csv

Output: Scatter Plots/CPI Headline vs Con.png
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
OUT_PATH = ROOT / "Scatter Plots" / "CPI Headline vs Con.png"

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
    # Consumption — already quarterly index (2015=100)
    con = con_df[con_df["country"] == code][["date", "value"]].rename(
        columns={"value": "consumption"})

    # CPI — monthly, aggregate to quarterly mean
    cpi_raw = cpi_df[
        (cpi_df["country"] == code) &
        (cpi_df["series"]  == meta["cpi_series"]) &
        (cpi_df["unit"]    == "index (2015=100)")
    ][["date", "value"]].copy()
    cpi_raw["date"] = cpi_raw["date"].apply(to_quarter_start)
    cpi_q = cpi_raw.groupby("date", as_index=False)["value"].mean().rename(
        columns={"value": "cpi_headline"})

    df = con.merge(cpi_q, on="date", how="inner").sort_values("date")
    df["year"] = df["date"].dt.year
    return df


def main():
    con_df = pd.read_csv(CON_F, parse_dates=["date"])
    cpi_df = pd.read_csv(CPI_F, parse_dates=["date"])

    # Global year range for consistent colourbar
    all_years = []
    datasets  = {}
    for code, meta in COUNTRIES.items():
        df = build_df(con_df, cpi_df, code, meta)
        datasets[code] = df
        all_years.extend(df["year"].tolist())

    year_min, year_max = min(all_years), max(all_years)
    norm = mcolors.Normalize(vmin=year_min, vmax=year_max)
    cmap = cm.get_cmap("plasma")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Headline CPI Index vs Domestic Consumption Index by Country\n"
        "(Quarterly observations, 2015=100)",
        fontsize=13, fontweight="bold"
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
            s=28, alpha=0.75, linewidths=0, zorder=2
        )

        # 2022 highlighted
        ax.scatter(
            yr_2022["cpi_headline"], yr_2022["consumption"],
            color="red", s=70, alpha=1.0, marker=meta["marker"],
            edgecolors="darkred", linewidths=0.8, zorder=4, label="2022"
        )

        # OLS regression line
        slope, intercept, r, p, _ = stats.linregress(
            df["cpi_headline"], df["consumption"])
        x_fit = np.linspace(df["cpi_headline"].min() - 2,
                            df["cpi_headline"].max() + 2, 200)
        ax.plot(x_fit, slope * x_fit + intercept,
                color=c, linewidth=1.8, linestyle="--",
                alpha=0.8, zorder=3,
                label=f"OLS  slope={slope:.3f}  R²={r**2:.3f}")

        # Annotate 2022 if present
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

        date_range = (f"{df['date'].min().strftime('%Y-Q%q' if False else '%Y')} – "
                      f"{df['date'].max().strftime('%Y')}")
        start_yr = df["date"].min().year
        end_yr   = df["date"].max().year

        ax.set_title(f"{meta['label']} ({meta['abbrev']})",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Headline CPI Index (2015=100)", fontsize=10)
        ax.set_ylabel("Domestic Consumption Index (2015=100)", fontsize=10)
        ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
        ax.grid(True, alpha=0.28)
        ax.text(0.98, 0.03, f"{start_yr}–{end_yr}",
                transform=ax.transAxes, fontsize=8,
                color="grey", ha="right")

    # Colourbar to the left of Canada panel
    plt.subplots_adjust(left=0.10, right=0.97, top=0.88,
                        bottom=0.10, wspace=0.30)
    sm = cm.ScalarMappable(cmap="plasma", norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.01, 0.15, 0.018, 0.65])
    cbar    = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cbar.set_label("Year", fontsize=10, labelpad=6)
    tick_years = sorted(set(
        [year_min] +
        [y for y in [1990, 1995, 2000, 2005, 2010, 2015, 2020] if year_min < y < year_max] +
        [year_max]
    ))
    cbar.set_ticks(tick_years)
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
