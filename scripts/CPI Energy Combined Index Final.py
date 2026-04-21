"""
Combine CPI Energy index series (2015=100) for Canada, UK, and USA.

All three are OECD COICOP 1999 CPI Energy index levels, base year 2015=100,
monthly, not seasonally adjusted. Units are directly comparable with no
rebasing required.

Sources (FRED / OECD):
  data/raw/CANCPIENGMINMEI.xlsx CAN ENERGY.xlsx  — Canada,  Jan 1961 – Mar 2025
  data/raw/GBRCPIENGMINMEI.xlsx UK ENERGY.xlsx   — UK,      Jan 1970 – Mar 2025
  data/raw/USA Energy CPI Indexed.xlsx           — USA,     Jan 1957 – Apr 2025

Overlapping period (inner join): Jan 1970 – Mar 2025
Binding constraint: UK series starts Jan 1970

Output: data/Clean 3/CPI Energy Combined Index Final.csv
Schema : date | canada_cpi_energy_idx2015 | uk_cpi_energy_idx2015 | usa_cpi_energy_idx2015
"""

from pathlib import Path
from datetime import datetime

import openpyxl
import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "Clean 3" / "CPI Energy Combined Index Final.csv"

SOURCES = {
    "canada_cpi_energy_idx2015": ROOT / "data" / "raw" / "CANCPIENGMINMEI.xlsx CAN ENERGY.xlsx",
    "uk_cpi_energy_idx2015":     ROOT / "data" / "raw" / "GBRCPIENGMINMEI.xlsx UK ENERGY.xlsx",
    "usa_cpi_energy_idx2015":    ROOT / "data" / "raw" / "USA Energy CPI Indexed.xlsx",
}


def load_fred_xlsx(path: Path, col_name: str) -> pd.DataFrame:
    wb = openpyxl.load_workbook(path)
    ws = wb["Monthly"]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    records = []
    for date_val, value in rows:
        if isinstance(date_val, datetime) and value is not None:
            records.append({
                "date":   datetime(date_val.year, date_val.month, 1),
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
    print(f"\n2015 mean check (should be ~100 for each):")
    for col in combined.columns[1:]:
        mean_2015 = combined.loc[combined["date"].dt.year == 2015, col].mean()
        print(f"  {col}: {mean_2015:.4f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, index=False, date_format="%Y-%m-%d")
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
