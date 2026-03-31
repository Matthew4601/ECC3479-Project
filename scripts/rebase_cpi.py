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
