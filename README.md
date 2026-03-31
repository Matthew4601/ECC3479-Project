# ECC3479-Project
Repository Structure
All the data is kept under the data folder which contains a further two folders, one for raw data and one for clean data.
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
The data comes from the relevant government statistic offices for each nation. The ONS for the Uk, Statistique Canada for Canada and the FRED for the United States.


Software used and required
Two non-standard packages were utilised alongside python to clean the data
To install these packages use the following code:
                  
                    pip3 install pandas openpyxl

How to clean the data
Open a python compatible coding environment such as VS Code.
Load the raw data files into the environment.

Collate the files together at the finest level (monthly) and eliminate the Australian data.
Script:
"""
Combine CPI data for Canada (Statistics Canada 18-10-0256-01),
UK (ONS CPIH index), and USA (BLS CPI components).

Sources:
  data/raw/1810025601-eng(CAN CPI).xlsx  — Canada core CPI measures, wide format, Jan 2020–Feb 2026
  data/raw/UK_CPI_index.csv              — UK CPIH all-items index (2015=100), monthly
  data/raw/USA CPI.xlsx                  — USA CPI 18-component series (YoY % change), monthly

Overlapping period (inner join): Jan 2020 – Feb 2026 (74 months, all three sources match exactly)
Output: data/clean/combined_cpi_all_countries.csv
"""

import re
from pathlib import Path
from datetime import datetime

import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CAN_PATH = ROOT / "data" / "raw" / "1810025601-eng(CAN CPI).xlsx"
UK_PATH  = ROOT / "data" / "raw" / "UK_CPI_index.csv"
USA_PATH = ROOT / "data" / "raw" / "USA CPI.xlsx"
OUT_PATH = ROOT / "data" / "clean" / "combined_cpi_all_countries.csv"

MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

UK_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

MONTHLY_RE = re.compile(r"^(\d{4}) ([A-Z]{3})$")

# Short slugs for each Canadian measure row (rows 12–26 in the sheet, 0-indexed)
CAN_MEASURE_SLUGS = [
    "can_cpi_common_yoy_pct",
    "can_cpi_median_yoy_pct",
    "can_cpi_trim_yoy_pct",
    "can_cpi_median_idx198901",
    "can_cpi_trim_idx198901",
    "can_cpi_ex8volatile_idx2002",
    "can_cpi_ex8volatile_notax_idx2002",
    "can_cpi_ex_indirect_tax_idx2002",
    "can_cpi_ex_food_energy_tax_idx2002",
    "can_cpi_ex8volatile_sa_notax_idx2002",
    "can_cpi_ex8volatile_sa_idx2002",
    "can_cpi_ex_indirect_tax_sa_idx2002",
    "can_cpi_ex_food_energy_tax_sa_idx2002",
]


def load_canada() -> pd.DataFrame:
    wb = openpyxl.load_workbook(CAN_PATH)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))

    # Row index 10 = header; cols 1+ are date strings like "January 2020"
    date_row = rows[10]
    date_cols = {}  # col_index -> datetime
    for i, val in enumerate(date_row[1:], start=1):
        if isinstance(val, str) and val in MONTH_MAP:
            pass  # shouldn't happen
        if isinstance(val, str):
            parts = val.split()
            if len(parts) == 2 and parts[0] in MONTH_MAP:
                date_cols[i] = datetime(int(parts[1]), MONTH_MAP[parts[0]], 1)

    # Data rows start at index 12; collect only the measure rows (stop before footnotes at row 31)
    data_rows = [r for r in rows[12:31] if r[0] is not None]

    records = {dt: {"date": dt} for dt in date_cols.values()}
    for measure_row, slug in zip(data_rows, CAN_MEASURE_SLUGS):
        for col_idx, dt in date_cols.items():
            val = measure_row[col_idx]
            records[dt][slug] = float(val) if val is not None else None

    return pd.DataFrame(list(records.values())).sort_values("date").reset_index(drop=True)


def load_uk() -> pd.DataFrame:
    rows = []
    with open(UK_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip().strip('"')
            parts = line.split('","')
            if len(parts) != 2:
                continue
            label, value = parts[0].strip(), parts[1].strip()
            m = MONTHLY_RE.match(label)
            if m:
                year, mon = int(m.group(1)), UK_MONTH_MAP[m.group(2)]
                rows.append({"date": datetime(year, mon, 1), "uk_cpih_all_items_idx2015": float(value)})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def load_usa() -> pd.DataFrame:
    wb = openpyxl.load_workbook(USA_PATH)
    ws = wb.active
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    header = rows[0]
    col_names = [str(h) if h is not None else f"col_{i}" for i, h in enumerate(header)]

    records = []
    for row in rows[1:]:
        date_val = row[2]
        if not isinstance(date_val, datetime):
            continue
        record = {"date": datetime(date_val.year, date_val.month, 1)}
        for c in range(3, len(col_names)):
            slug = "usa_" + col_names[c].lower().replace(" ", "_").replace("(", "").replace(")", "")
            record[slug] = row[c]
        records.append(record)

    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)


def main():
    canada = load_canada()
    uk     = load_uk()
    usa    = load_usa()

    combined = canada.merge(uk, on="date", how="inner").merge(usa, on="date", how="inner")
    combined = combined.sort_values("date").reset_index(drop=True)

    print(f"Overlapping period : {combined['date'].min().strftime('%Y-%m')} to {combined['date'].max().strftime('%Y-%m')} ({len(combined)} months)")
    print(f"Columns            : {list(combined.columns)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, index=False, date_format="%Y-%m-%d")
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()

Rebase the data so that all the datapoints are able to be compared accurately by setting a common baseline (the year 2015).
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
