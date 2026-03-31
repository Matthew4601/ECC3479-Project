"""
Combine UK and USA monthly CPI data for the overlapping period.

UK source : data/raw/UK_CPI_index.csv  — CPIH All Items index (2015=100)
USA source : data/raw/USA CPI.xlsx     — 18 component series (YoY % change)
Output    : data/clean/combined_cpi.csv
"""

import re
from pathlib import Path
from datetime import datetime

import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
UK_PATH = ROOT / "data" / "raw" / "UK_CPI_index.csv"
USA_PATH = ROOT / "data" / "raw" / "USA CPI.xlsx"
OUT_PATH = ROOT / "data" / "clean" / "combined_cpi.csv"

MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

MONTHLY_RE = re.compile(r"^(\d{4}) ([A-Z]{3})$")


def load_uk() -> pd.DataFrame:
    rows = []
    with open(UK_PATH, encoding="utf-8") as f:
        for line in f:
            # Strip surrounding quotes and whitespace
            line = line.strip().strip('"')
            # Split on '","'
            parts = line.split('","')
            if len(parts) != 2:
                continue
            label, value = parts[0].strip(), parts[1].strip()
            m = MONTHLY_RE.match(label)
            if m:
                year, mon = int(m.group(1)), MONTH_MAP[m.group(2)]
                rows.append({"date": datetime(year, mon, 1), "uk_cpih_all_items": float(value)})
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def load_usa() -> pd.DataFrame:
    wb = openpyxl.load_workbook(USA_PATH)
    ws = wb.active
    rows = list(ws.iter_rows(min_row=2, values_only=True))

    # Row index 0 is the header row
    header = rows[0]
    col_names = [str(h) if h is not None else f"col_{i}" for i, h in enumerate(header)]
    # Column 2 (index 2) is 'Month'; columns 3+ are data series
    date_col = 2
    data_cols = list(range(3, len(col_names)))

    records = []
    for row in rows[1:]:
        date_val = row[date_col]
        if not isinstance(date_val, datetime):
            continue
        record = {"date": datetime(date_val.year, date_val.month, 1)}
        for c in data_cols:
            record[f"usa_{col_names[c].lower().replace(' ', '_').replace('(', '').replace(')', '')}"] = row[c]
        records.append(record)

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    return df


def main():
    uk = load_uk()
    usa = load_usa()

    combined = pd.merge(uk, usa, on="date", how="inner")
    combined = combined.sort_values("date").reset_index(drop=True)

    overlap_start = combined["date"].min().strftime("%Y-%m")
    overlap_end = combined["date"].max().strftime("%Y-%m")
    print(f"Overlapping period : {overlap_start} to {overlap_end} ({len(combined)} months)")
    print(f"Columns            : {list(combined.columns)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, index=False, date_format="%Y-%m-%d")
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
