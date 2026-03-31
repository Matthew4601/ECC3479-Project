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
