"""
Combine CPI index-level series for Canada, UK, and USA.

All three are index levels (different base years), monthly frequency.

Sources:
  data/raw/1810025601-eng(CAN CPI).xlsx     — Canada core CPI measures (Statistics Canada),
                                               wide format, Jan 2020–Feb 2026
  data/raw/UK_CPI_index.csv                 — UK CPIH all-items index (ONS, 2015=100),
                                               Jan 1988–Feb 2026
  data/raw/CPIAUCSL.csv USA CPI 2.csv       — USA CPI All Urban Consumers index (FRED/BLS,
                                               1982-84=100), Jan 1947–Feb 2026 (NaN: Oct 2025)

Overlapping period (inner join, NaN rows dropped): Jan 2020 – Feb 2026
Note: Oct 2025 is excluded due to a missing value in the CPIAUCSL series (73 months total).

Output: data/clean/combined_cpi_index_levels.csv
"""

import re
from pathlib import Path
from datetime import datetime

import openpyxl
import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
CAN_PATH = ROOT / "data" / "raw" / "1810025601-eng(CAN CPI).xlsx"
UK_PATH  = ROOT / "data" / "raw" / "UK_CPI_index.csv"
USA_PATH = ROOT / "data" / "raw" / "CPIAUCSL.csv USA CPI 2.csv"
OUT_PATH = ROOT / "data" / "clean" / "combined_cpi_index_levels.csv"

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

# Short slugs for each Canadian measure row (rows 12–24 in sheet, 0-indexed)
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

    # Row 10: header with date strings "January 2020" etc. in cols 1+
    date_row = rows[10]
    date_cols = {}
    for i, val in enumerate(date_row[1:], start=1):
        if isinstance(val, str):
            parts = val.split()
            if len(parts) == 2 and parts[0] in MONTH_MAP:
                date_cols[i] = datetime(int(parts[1]), MONTH_MAP[parts[0]], 1)

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
    df = pd.read_csv(USA_PATH, parse_dates=["observation_date"])
    df = df.dropna(subset=["CPIAUCSL"])
    df["date"] = df["observation_date"].apply(lambda d: datetime(d.year, d.month, 1))
    return df[["date", "CPIAUCSL"]].rename(columns={"CPIAUCSL": "usa_cpi_all_urban_idx198284"})


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
