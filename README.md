# ECC3479-Project
Repository Structure
Files under the raw folder are the orginal untampered downloads from the original sources.
Files under the clean folder are those that have undergone transforamtions to allow them to be more comparable with the data from their counterpart nations.
Scripts include the codes used to transform the files from those in the raw folder to those in the clean.

CPI Data
What the data is,
Value of the consumer price index (CPI) within the constituent economies.
Data files used
CPIAUCSL.csv USA CPI2.csv
UK_CPI_index.csv
1810025601-eng(CAN CPI).xlsx

Where the data is gathered from,
The data comes from the relevatn governmetn statistic offices for each nation. The ONS for the Uk, 


Software used and required
Two non-standard packages were utilised alongside python to clean the data
To install these packages use the following code:
                  
                    pip3 install pandas openpyxl

How to clean the data
Open a python compatible coding environment such as VS Code.
Load the raw data files into the environment.

Collate the files together at the finest level (monthly) and eliminate the Australian data.
Script


Rebase the data so that all the datapoints are able to be compared accurately without the influence of differing
Script:
"""
Rebase UK, Canada, and USA CPI index series to a common base (2015=100).

Source : data/clean/cpi_clean.csv  (long format)
Output : data/clean/cpi_rebased.csv (long format, same schema)

Approach:
- Australia is excluded.
- Index series are rebased so that the mean of calendar-year 2015 = 100
  for each country/series combination.
- YoY % change and quarterly % change series are carried through unchanged
  (they are already comparable across countries and need no rebasing).
- Output is trimmed to Jan 1988 – Feb 2026, the common overlap across all
  three countries (binding constraint: UK CPIH starts Jan 1988).
"""

from pathlib import Path
import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
IN_PATH  = ROOT / "data" / "clean" / "cpi_clean.csv"
OUT_PATH = ROOT / "data" / "clean" / "cpi_rebased.csv"

BASE_YEAR = 2015


def rebase(df_series: pd.DataFrame) -> pd.DataFrame:
    """Rebase a single index series so that mean(BASE_YEAR) = 100."""
    df_series = df_series.copy()
    df_series["date"] = pd.to_datetime(df_series["date"])
    base_mean = df_series.loc[df_series["date"].dt.year == BASE_YEAR, "value"].mean()
    df_series["value"] = (df_series["value"] / base_mean) * 100
    df_series["unit"] = f"index ({BASE_YEAR}=100)"
    return df_series


def main():
    df = pd.read_csv(IN_PATH, parse_dates=["date"])

    # Drop Australia
    df = df[df["country"] != "AUS"].copy()

    index_mask = df["unit"] == "index"
    pct_mask   = ~index_mask

    # Rebase each index series independently
    rebased_parts = []
    for (country, series), group in df[index_mask].groupby(["country", "series"]):
        rebased_parts.append(rebase(group))

    rebased_index = pd.concat(rebased_parts, ignore_index=True)
    pct_df        = df[pct_mask].copy()

    combined = pd.concat([rebased_index, pct_df], ignore_index=True)

    # Trim to common overlap: Jan 1988 – Feb 2026
    combined = combined[(combined["date"] >= "1988-01-01") & (combined["date"] <= "2026-02-01")]
    combined = combined.sort_values(["country", "series", "date"]).reset_index(drop=True)

    # Summary
    print("Rebased index series (2015=100):")
    for (country, series), grp in rebased_index.groupby(["country", "series"]):
        base_check = grp.loc[grp["date"].dt.year == BASE_YEAR, "value"].mean()
        print(f"  {country} | {series[:60]}")
        print(f"    Range: {grp['date'].min().strftime('%Y-%m')} – {grp['date'].max().strftime('%Y-%m')} "
              f"| 2015 mean check: {base_check:.4f}")

    print(f"\nPassthrough % change series: {pct_df['series'].nunique()} series, {len(pct_df)} rows")
    print(f"\nOutput: {len(combined)} rows × {len(combined.columns)} cols")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, index=False, date_format="%Y-%m-%d")
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()

Produced File
CPI_Combined_Final.csv


Combined_energy
What the data is,
The weighting of energy production methods within the economies.
Data files used:
International Energy Agency - total energy supply in United Kingdom
International Energy Agency - total energy supply in United States.csv
International Energy Agency - total energy supply in Canada.csv

Origins of the raw data,
The original raw data for each country was sourced from the International Energy Agency (IEA), which breaks down each nations energy production methods into subcategories and tracks how these subcategories have expanded or contracted over the five year period of 2020-2025.

Software used and required
Two non-standard packages were utilised alongside python to clean the data
To install these packages use the following code:
                   
                    pip3 install pandas openpyxl

How to clean the data
Open a python compatible coding environment such as VS Code.
Load the raw data files into the environment.
All the relevant uploaded data files were collated at the finest level into the same file, set to a standard interval of dates and includes the percentage weighting of each energy generation form.
Script:
"""
Combine FRED monthly CPI Energy series for Canada, UK, and USA.

Sources:
  CANCPIENGMINMEI.xlsx CAN ENERGY.xlsx  — Canada CPI Energy, Index 2015=100,        1961-01 to 2025-03
  GBRCPIENGMINMEI.xlsx UK ENERGY.xlsx   — UK CPI Energy,     Index 2015=100,        1970-01 to 2025-03
  CPGREN01USM659N.xlsx US ENERGY.xlsx   — USA CPI Energy,    YoY % growth rate,     1958-01 to 2025-04

Note: Canada and UK are index levels (2015=100); USA is a YoY % growth rate.
      Column names reflect units to avoid confusion.

Overlapping period (inner join): 1970-01 to 2025-03 (binding constraint: UK starts Jan 1970)
Output: data/clean/combined_energy_cpi_index.csv
"""

from pathlib import Path
from datetime import datetime

import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

SOURCES = {
    "canada_cpi_energy_idx2015":    ROOT / "data" / "raw" / "CANCPIENGMINMEI.xlsx CAN ENERGY.xlsx",
    "uk_cpi_energy_idx2015":        ROOT / "data" / "raw" / "GBRCPIENGMINMEI.xlsx UK ENERGY.xlsx",
    "usa_cpi_energy_yoy_pct":       ROOT / "data" / "raw" / "CPGREN01USM659N.xlsx US ENERGY.xlsx",
}

OUT_PATH = ROOT / "data" / "clean" / "combined_energy_cpi_index.csv"


def load_fred_xlsx(path: Path, col_name: str) -> pd.DataFrame:
    wb = openpyxl.load_workbook(path)
    ws = wb["Monthly"]
    rows = list(ws.iter_rows(min_row=2, values_only=True))  # skip header
    records = []
    for date_val, value in rows:
        if isinstance(date_val, datetime) and value is not None:
            records.append({
                "date": datetime(date_val.year, date_val.month, 1),
                col_name: float(value),
            })
    return pd.DataFrame(records)


def main():
    dfs = [load_fred_xlsx(path, col) for col, path in SOURCES.items()]

    combined = dfs[0]
    for df in dfs[1:]:
        combined = combined.merge(df, on="date", how="inner")

    combined = combined.sort_values("date").reset_index(drop=True)

    print(f"Overlapping period : {combined['date'].min().strftime('%Y-%m')} to {combined['date'].max().strftime('%Y-%m')} ({len(combined)} months)")
    print(f"Columns            : {list(combined.columns)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, index=False, date_format="%Y-%m-%d")
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()


Energy CPI
What the data is,
Price of energy for consumers in the constituent countries expressed as a year on year change in the price between the months in the current and previous years.
Data files used:
CPGREN01CAM659N.xlsx(UK).xlsx
CPGREN01GBM659N.xlsx(CAN).xlsx
CPGREN01USM659N.xlsx(USA).xlsx

Where the data is gathered from
The data is gathered from the Federal Reserve of St. Louis (FRED). To eliminate any issues in currency conversions, differing scales or differing collection methods all of the data for each of the countries in this subcategory was obtained from this same source. 

Software used and required
Two non-standard packages were utilised alongside python to clean the data
To install these packages use the following code:
                    
                    pip3 install pandas openpyxl

How to clean the data
Open a python compatible coding environment such as VS Code.
Load the raw data files into the environment.
The data was collated into a single file at the finest level, ensuring that each dataset is compared monthly and over the same time period.
Script:
"""
Combine FRED monthly CPI Energy series for Canada, UK, and USA.

All three are YoY % growth rate, monthly, not seasonally adjusted (COICOP 1999).

Sources (note: filename suffixes in raw files are mislabelled):
  CPGREN01CAM659N.xlsx(UK).xlsx  — Canada  (series CPGREN01CAM659N, 1962–2025)
  CPGREN01GBM659N.xlsx(CAN).xlsx — UK      (series CPGREN01GBM659N, 1971–2025)
  CPGREN01USM659N.xlsx(USA).xlsx — USA     (series CPGREN01USM659N, 1958–2025)

Overlapping period (inner join): 1971-01 to 2025-03  (binding constraint: UK starts Jan 1971)
Output: data/clean/combined_energy_cpi.csv
Units: % YoY growth rate
"""

from pathlib import Path
from datetime import datetime

import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

SOURCES = {
    "canada": ROOT / "data" / "raw" / "CPGREN01CAM659N.xlsx(UK).xlsx",
    "uk":     ROOT / "data" / "raw" / "CPGREN01GBM659N.xlsx(CAN).xlsx",
    "usa":    ROOT / "data" / "raw" / "CPGREN01USM659N.xlsx(USA).xlsx",
}

OUT_PATH = ROOT / "data" / "clean" / "combined_energy_cpi.csv"


def load_fred_xlsx(path: Path, country: str) -> pd.DataFrame:
    wb = openpyxl.load_workbook(path)
    ws = wb["Monthly"]
    rows = list(ws.iter_rows(min_row=2, values_only=True))  # skip header
    records = []
    for date_val, value in rows:
        if isinstance(date_val, datetime) and value is not None:
            records.append({
                "date": datetime(date_val.year, date_val.month, 1),
                f"cpi_energy_yoy_{country}": float(value),
            })
    return pd.DataFrame(records)


def main():
    dfs = [load_fred_xlsx(path, country) for country, path in SOURCES.items()]

    combined = dfs[0]
    for df in dfs[1:]:
        combined = combined.merge(df, on="date", how="inner")

    combined = combined.sort_values("date").reset_index(drop=True)

    print(f"Overlapping period : {combined['date'].min().strftime('%Y-%m')} to {combined['date'].max().strftime('%Y-%m')} ({len(combined)} months)")
    print(f"Columns            : {list(combined.columns)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, index=False, date_format="%Y-%m-%d")
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
