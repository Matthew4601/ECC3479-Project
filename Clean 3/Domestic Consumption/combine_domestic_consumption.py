"""
Combine household/private final consumption expenditure data for
Canada, UK, and USA into a single long-format dataset.

Sources (all from FRED):
  CAN Domestic Consumption.xlsx  — NAEXKP02CAQ189S
    Private Final Consumption Expenditure, Canada
    Constant prices, Canadian Dollars, Quarterly, Seasonally Adjusted
    Range: 1981 Q1 – 2023 Q3

  UK Domestic Consumption.xlsx   — NAEXKP02GBQ189S
    Private Final Consumption Expenditure, United Kingdom
    Constant prices, Pound Sterling, Quarterly, Seasonally Adjusted
    Range: 1995 Q1 – 2023 Q3

  US Domestic Consumption.xlsx   — PCEC96
    Real Personal Consumption Expenditures, USA
    Billions of Chained 2017 USD, Monthly, Seasonally Adjusted Annual Rate
    Range: 2007-01 – 2026-02

Note: USA data is monthly; it is aggregated to quarterly means to match
      the frequency of Canada and UK before combining.

Common overlap (all three, quarterly): 2007 Q1 – 2023 Q3

Each country's series is rebased so that the mean of calendar-year 2015 = 100,
removing currency differences and making cross-country trajectories directly
comparable.

Output: Domestic_Consumption_Combined.csv  (same folder as this script)
Schema : date | country | series_id | description | unit | value
"""

from pathlib import Path
from datetime import datetime

import openpyxl
import pandas as pd

ROOT      = Path(__file__).resolve().parent
OUT       = ROOT / "Domestic_Consumption_Combined.csv"
BASE_YEAR = 2015

SOURCES = {
    "CAN": {
        "file":        ROOT / "CAN Domestic Consumption.xlsx",
        "sheet":       "Quarterly",
        "series_id":   "NAEXKP02CAQ189S",
        "description": "Private Final Consumption Expenditure (constant prices)",
        "unit":        "CAD (constant prices, seasonally adjusted)",
        "frequency":   "quarterly",
    },
    "UK": {
        "file":        ROOT / "UK Domestic Consumption.xlsx",
        "sheet":       "Quarterly",
        "series_id":   "NAEXKP02GBQ189S",
        "description": "Private Final Consumption Expenditure (constant prices)",
        "unit":        "GBP (constant prices, seasonally adjusted)",
        "frequency":   "quarterly",
    },
    "USA": {
        "file":        ROOT / "US Domestic Consumption.xlsx",
        "sheet":       "Monthly",
        "series_id":   "PCEC96",
        "description": "Real Personal Consumption Expenditures",
        "unit":        "Billions of Chained 2017 USD (SAAR)",
        "frequency":   "monthly -> aggregated to quarterly mean",
    },
}


def load_fred_xlsx(path: Path, sheet: str) -> pd.DataFrame:
    wb = openpyxl.load_workbook(path)
    ws = wb[sheet]
    rows = list(ws.iter_rows(min_row=2, values_only=True))  # skip header
    records = []
    for date_val, value in rows:
        if isinstance(date_val, datetime) and value is not None:
            records.append({"date": datetime(date_val.year, date_val.month, 1), "value": float(value)})
    return pd.DataFrame(records)


def to_quarter_start(date: pd.Timestamp) -> pd.Timestamp:
    """Map any date to the first month of its quarter."""
    quarter_start_month = ((date.month - 1) // 3) * 3 + 1
    return pd.Timestamp(date.year, quarter_start_month, 1)


def main():
    parts = []

    for country, meta in SOURCES.items():
        df = load_fred_xlsx(meta["file"], meta["sheet"])
        df["date"] = pd.to_datetime(df["date"])

        # Aggregate USA monthly -> quarterly
        if country == "USA":
            df["date"] = df["date"].apply(to_quarter_start)
            df = df.groupby("date", as_index=False)["value"].mean()

        # Rebase to 2015=100
        base_mean = df.loc[df["date"].dt.year == BASE_YEAR, "value"].mean()
        df["value"] = (df["value"] / base_mean) * 100

        df["country"]     = country
        df["series_id"]   = meta["series_id"]
        df["description"] = meta["description"]
        df["unit"]        = f"index ({BASE_YEAR}=100)"

        parts.append(df[["date", "country", "series_id", "description", "unit", "value"]])

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.sort_values(["country", "date"]).reset_index(drop=True)

    # Summary
    print("Series loaded:")
    for country, grp in combined.groupby("country"):
        print(f"  {country}: {grp['date'].min().strftime('%Y-%m')} to "
              f"{grp['date'].max().strftime('%Y-%m')} ({len(grp)} quarters)")

    overlap = combined.groupby("date").filter(lambda x: x["country"].nunique() == 3)
    if not overlap.empty:
        print(f"\nCommon overlap (all 3 countries): "
              f"{overlap['date'].min().strftime('%Y-%m')} to "
              f"{overlap['date'].max().strftime('%Y-%m')} "
              f"({overlap['date'].nunique()} quarters)")

    print(f"\nTotal rows: {len(combined)}")
    combined.to_csv(OUT, index=False, date_format="%Y-%m-%d")
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()
