"""
Combine IEA total energy supply data for the UK, USA, and Canada.

All three sources share the same 7 energy products and cover 2000–2024 annually.
Output is wide format: one row per year, columns prefixed by country.

Sources:
  data/raw/International Energy Agency - total energy supply in United States.csv
  data/raw/International Energy Agency - total energy supply in Canada.csv
  data/raw/International Energy Agency - total energy supply in United Kingdom.xlsx
Output:
  data/clean/combined_energy.csv
Units: TJ (terajoules) throughout.
"""

from pathlib import Path

import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
USA_PATH = ROOT / "data" / "raw" / "International Energy Agency - total energy supply in United States.csv"
CAN_PATH = ROOT / "data" / "raw" / "International Energy Agency - total energy supply in Canada.csv"
UK_PATH  = ROOT / "data" / "raw" / "International Energy Agency - total energy supply in United Kingdom.xlsx"
OUT_PATH = ROOT / "data" / "clean" / "combined_energy.csv"

PRODUCT_SLUG = {
    "Coal and coal products":          "coal",
    "Oil and oil products":            "oil",
    "Natural gas":                     "natural_gas",
    "Nuclear":                         "nuclear",
    "Hydropower":                      "hydropower",
    "Solar, wind and other renewables":"solar_wind_renewables",
    "Biofuels and waste":              "biofuels_waste",
}


def load_csv(path: Path, country: str) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=1, header=None, names=["product", "value", "year", "units"])
    df["product"] = df["product"].str.strip()
    df["year"] = df["year"].astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[df["product"].isin(PRODUCT_SLUG)]
    df["column"] = country + "_" + df["product"].map(PRODUCT_SLUG)
    return df.pivot_table(index="year", columns="column", values="value").reset_index()


def load_uk_xlsx(path: Path) -> pd.DataFrame:
    wb = openpyxl.load_workbook(path)
    ws = wb["Sheet 1 - International Energy "]
    rows = [row for row in ws.iter_rows(values_only=True) if any(c is not None for c in row)]

    # Skip title row; data rows start at index 1 (header is rows[1])
    records = []
    for row in rows[2:]:  # rows[0]=title, rows[1]=header, rows[2:]=data
        product, value, year, *_ = row
        product = str(product).strip() if product else None
        if product in PRODUCT_SLUG and year is not None:
            records.append({"product": product, "value": int(value), "year": int(year)})

    df = pd.DataFrame(records)
    df["column"] = "uk_" + df["product"].map(PRODUCT_SLUG)
    return df.pivot_table(index="year", columns="column", values="value").reset_index()


def main():
    usa = load_csv(USA_PATH, "usa")
    can = load_csv(CAN_PATH, "canada")
    uk  = load_uk_xlsx(UK_PATH)

    # Inner join on year — all three cover 2000–2024 so nothing is lost
    combined = usa.merge(can, on="year").merge(uk, on="year")
    combined = combined.sort_values("year").reset_index(drop=True)
    # Clean up pandas column index name artifact
    combined.columns.name = None

    print(f"Years     : {combined['year'].min()} – {combined['year'].max()} ({len(combined)} rows)")
    print(f"Columns   : {list(combined.columns)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
