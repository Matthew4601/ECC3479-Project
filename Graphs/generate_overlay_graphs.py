from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent

# ── Load data ─────────────────────────────────────────────────────────────────

cpi_df = pd.read_csv(ROOT / "data/clean/CPI_Combined_Final.csv", parse_dates=["date"])
energy_df = pd.read_csv(ROOT / "data/clean/combined_energy_cpi.csv", parse_dates=["date"])
cons_df = pd.read_csv(
    ROOT / "Clean 3/Domestic Consumption/Domestic_Consumption_Combined.csv",
    parse_dates=["date"]
)

# Headline CPI index series per country
HEADLINE_SERIES = {
    "CAN": "CPI All-items Index (Canada national, 2002=100)",
    "UK":  "CPIH Index, All Items 2015=100 (L522)",
    "USA": "CPI-U All Items Index (1982-84=100, FRED CPIAUCSL)",
}

ENERGY_COL = {
    "CAN": "cpi_energy_yoy_canada",
    "UK":  "cpi_energy_yoy_uk",
    "USA": "cpi_energy_yoy_usa",
}

COUNTRY_LABELS = {"CAN": "Canada", "UK": "United Kingdom", "USA": "United States"}
OUTPUT_NAMES   = {"CAN": "CAN Overlay", "UK": "UK Overlay", "USA": "USA Overlay"}

# Each country's first national lockdown date
LOCKDOWN_DATES = {
    "CAN": (pd.Timestamp("2020-03-17"), "Canada Lockdown\n(17 Mar 2020)"),
    "UK":  (pd.Timestamp("2020-03-23"), "UK Lockdown\n(23 Mar 2020)"),
    "USA": (pd.Timestamp("2020-03-13"), "US National Emergency\n(13 Mar 2020)"),
}
SHOCK_DATE = pd.Timestamp("2022-10-01")

# Colour palette
C_CPI  = "#2166AC"   # blue  — headline CPI index
C_CONS = "#4DAC26"   # green — domestic consumption index
C_ENRG = "#D6604D"   # red   — energy CPI YoY %


def build_series(country):
    # Headline CPI (index 2015=100)
    cpi = (
        cpi_df[
            (cpi_df["country"] == country) &
            (cpi_df["series"] == HEADLINE_SERIES[country]) &
            (cpi_df["unit"] == "index (2015=100)")
        ][["date", "value"]]
        .rename(columns={"value": "cpi"})
        .sort_values("date")
    )

    # Energy CPI (YoY %)
    enrg = (
        energy_df[["date", ENERGY_COL[country]]]
        .rename(columns={ENERGY_COL[country]: "energy_yoy"})
        .sort_values("date")
    )

    # Domestic consumption (index 2015=100)
    cons = (
        cons_df[cons_df["country"] == country][["date", "value"]]
        .rename(columns={"value": "consumption"})
        .sort_values("date")
    )

    return cpi, enrg, cons


def make_overlay(country):
    cpi, enrg, cons = build_series(country)

    # Common date range: start where all three overlap, end at latest common point
    start = max(cpi["date"].min(), enrg["date"].min(), cons["date"].min())
    end   = min(cpi["date"].max(), enrg["date"].max(), cons["date"].max())
    cpi   = cpi[(cpi["date"] >= start)   & (cpi["date"] <= end)]
    enrg  = enrg[(enrg["date"] >= start) & (enrg["date"] <= end)]
    cons  = cons[(cons["date"] >= start) & (cons["date"] <= end)]

    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax2 = ax1.twinx()

    # Left axis: headline CPI and domestic consumption (both index 2015=100)
    ax1.plot(cpi["date"],  cpi["cpi"],         color=C_CPI,  linewidth=2,   label="Headline CPI (index 2015=100)")
    ax1.plot(cons["date"], cons["consumption"], color=C_CONS, linewidth=2,   label="Domestic Consumption (index 2015=100)")

    # Right axis: energy CPI YoY %
    ax2.plot(enrg["date"], enrg["energy_yoy"],  color=C_ENRG, linewidth=1.5, linestyle="--", label="Energy CPI YoY %", alpha=0.85)
    ax2.axhline(0, color=C_ENRG, linewidth=0.6, linestyle=":", alpha=0.5)

    # Annotations
    ymin_left = ax1.get_ylim()[0]
    ymax_left = max(cpi["cpi"].max(), cons["consumption"].max())
    yrange    = ymax_left - ymin_left

    lockdown_ts, lockdown_label = LOCKDOWN_DATES[country]
    ax1.axvline(lockdown_ts, color="#555555", linestyle=":", linewidth=1.6)
    ax1.text(lockdown_ts, ymin_left + yrange * 0.08,
             f"  {lockdown_label}", fontsize=8, color="#555555", va="bottom")

    ax1.axvline(SHOCK_DATE, color="#8B0000", linestyle=":", linewidth=1.6)
    ax1.text(SHOCK_DATE, ymin_left + yrange * 0.08,
             "  Energy Shock\n  (Oct 2022)", fontsize=8, color="#8B0000", va="bottom")

    # Axes styling
    ax1.set_xlim(start, end)
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax1.set_ylabel("Index (2015 = 100)", fontsize=10)
    ax2.set_ylabel("Energy CPI — YoY % Change", fontsize=10, color=C_ENRG)
    ax2.tick_params(axis="y", labelcolor=C_ENRG)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    ax1.set_xlabel("Date", fontsize=10)
    ax1.set_title(
        f"{COUNTRY_LABELS[country]}: Headline CPI, Energy CPI & Domestic Consumption",
        fontsize=13, fontweight="bold", pad=12
    )

    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)

    # Combined legend
    handles = [
        Line2D([0], [0], color=C_CPI,  linewidth=2,   label="Headline CPI (index 2015=100)"),
        Line2D([0], [0], color=C_CONS, linewidth=2,   label="Domestic Consumption (index 2015=100)"),
        Line2D([0], [0], color=C_ENRG, linewidth=1.5, linestyle="--", label="Energy CPI YoY % (right axis)"),
    ]
    ax1.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    out = OUT_DIR / f"{OUTPUT_NAMES[country]}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


for country in ["CAN", "UK", "USA"]:
    make_overlay(country)
