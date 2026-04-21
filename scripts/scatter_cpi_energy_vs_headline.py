"""
Scatter plots: CPI Energy Index vs Headline CPI Index over time, per country.

For each country (Canada, UK, USA), plots monthly observations as a scatter
where x = CPI Energy (2015=100) and y = Headline CPI (2015=100).
Points are coloured by year; 2022 observations are highlighted in red with
a larger marker and labelled.

Sources:
  Clean 3/Domestic Consumption/CPI Energy Combined Index Final.csv
  Clean 3/Domestic Consumption/CPI_Combined_Final.csv

Output: Scatter Plots/CPI Energy vs CPI Headline.png
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

ROOT      = Path(__file__).resolve().parent.parent
ENERGY_F  = ROOT / "Clean 3" / "Domestic Consumption" / "CPI Energy Combined Index Final.csv"
HEADLINE_F= ROOT / "Clean 3" / "Domestic Consumption" / "CPI_Combined_Final.csv"
OUT_PATH  = ROOT / "Scatter Plots" / "CPI Energy vs CPI Headline.png"

COUNTRIES = {
    "CAN": {
        "energy_col":   "canada_cpi_energy_idx2015",
        "headline_series": "CPI All-items Index (Canada national, 2002=100)",
        "label":        "Canada",
        "color":        "#D62728",
    },
    "UK": {
        "energy_col":   "uk_cpi_energy_idx2015",
        "headline_series": "CPIH Index, All Items 2015=100 (L522)",
        "label":        "United Kingdom",
        "color":        "#1F77B4",
    },
    "USA": {
        "energy_col":   "usa_cpi_energy_idx2015",
        "headline_series": "CPI-U All Items Index (1982-84=100, FRED CPIAUCSL)",
        "label":        "United States",
        "color":        "#2CA02C",
    },
}


def load_data():
    energy   = pd.read_csv(ENERGY_F, parse_dates=["date"])
    headline = pd.read_csv(HEADLINE_F, parse_dates=["date"])
    return energy, headline


def build_country_df(energy, headline, country_code, meta):
    # Energy series
    en = energy[["date", meta["energy_col"]]].rename(
        columns={meta["energy_col"]: "cpi_energy"})

    # Headline index series for this country
    hl = headline[
        (headline["country"] == country_code) &
        (headline["series"] == meta["headline_series"]) &
        (headline["unit"] == "index (2015=100)")
    ][["date", "value"]].rename(columns={"value": "cpi_headline"})

    df = en.merge(hl, on="date", how="inner").sort_values("date")
    df["year"] = df["date"].dt.year
    return df


def main():
    energy, headline = load_data()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "CPI Energy Index vs Headline CPI Index by Country\n(Monthly observations, 2015=100)",
        fontsize=14, fontweight="bold", y=1.02
    )

    all_years = sorted(set(
        yr for meta in COUNTRIES.values()
        for yr in build_country_df(energy, headline,
            [k for k, v in COUNTRIES.items() if v == meta][0], meta)["year"]
    ))
    year_min, year_max = min(all_years), max(all_years)
    norm  = mcolors.Normalize(vmin=year_min, vmax=year_max)
    cmap  = cm.get_cmap("plasma")

    for ax, (country_code, meta) in zip(axes, COUNTRIES.items()):
        df = build_country_df(energy, headline, country_code, meta)

        # All non-2022 points
        non_2022 = df[df["year"] != 2022]
        sc = ax.scatter(
            non_2022["cpi_energy"], non_2022["cpi_headline"],
            c=non_2022["year"], cmap="plasma", norm=norm,
            s=18, alpha=0.7, linewidths=0, zorder=2
        )

        # 2022 highlighted
        yr_2022 = df[df["year"] == 2022]
        ax.scatter(
            yr_2022["cpi_energy"], yr_2022["cpi_headline"],
            color="red", s=60, alpha=1.0, edgecolors="darkred",
            linewidths=0.8, zorder=4, label="2022"
        )

        # Annotate first 2022 point
        if not yr_2022.empty:
            ax.annotate(
                "2022",
                xy=(yr_2022["cpi_energy"].iloc[0], yr_2022["cpi_headline"].iloc[0]),
                xytext=(8, 8), textcoords="offset points",
                fontsize=9, color="darkred", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8)
            )

        # Reference lines at 100
        ax.axvline(100, color="grey", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.axhline(100, color="grey", linestyle="--", linewidth=0.7, alpha=0.5)

        ax.set_title(meta["label"], fontsize=12, fontweight="bold")
        ax.set_xlabel("CPI Energy Index (2015=100)", fontsize=10)
        ax.set_ylabel("Headline CPI Index (2015=100)", fontsize=10)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)

        date_range = f"{df['date'].min().strftime('%Y')}–{df['date'].max().strftime('%Y')}"
        ax.text(0.98, 0.03, date_range, transform=ax.transAxes,
                fontsize=8, color="grey", ha="right")

    # Leave room on the left for the colourbar, then lay out the subplots
    plt.subplots_adjust(left=0.10, right=0.97, top=0.91, bottom=0.10,
                        wspace=0.32)

    # Colourbar placed to the left of the Canada subplot, clear of all text
    sm = cm.ScalarMappable(cmap="plasma", norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.01, 0.15, 0.018, 0.65])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cbar.set_label("Year", fontsize=10, labelpad=6)
    cbar.set_ticks([year_min, 1980, 1990, 2000, 2010, 2020, year_max])
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
