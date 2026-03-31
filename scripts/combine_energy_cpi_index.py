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
