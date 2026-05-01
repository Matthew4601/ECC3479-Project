from pathlib import Path
import openpyxl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "UK Energy Price Cap.xlsx"
OUT_PATH  = Path(__file__).resolve().parent / "OFGEM Cap graph.png"

wb = openpyxl.load_workbook(DATA_PATH)
ws = wb["Quarterly Price Cap Data"]
rows = list(ws.iter_rows(min_row=2, values_only=True))

records = []
for row in rows:
    period, start, end, _, ofgem, epg, effective, *_ = row
    if not isinstance(start, (str, datetime)) or ofgem is None:
        continue
    date = datetime.strptime(start, "%Y-%m-%d") if isinstance(start, str) else start
    records.append({
        "date":      date,
        "ofgem":     float(ofgem),
        "effective": float(effective) if effective is not None else float(ofgem),
    })

df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

# Use period midpoints for a smoother step plot
df["end_date"] = df["date"].shift(-1).fillna(df["date"] + pd.DateOffset(months=3))

fig, ax = plt.subplots(figsize=(13, 6))

for _, row in df.iterrows():
    xs = [row["date"], row["end_date"]]
    ax.plot(xs, [row["ofgem"],     row["ofgem"]],     color="#E05C5C", linewidth=2.5)
    ax.plot(xs, [row["effective"], row["effective"]], color="#3A7DC9", linewidth=2.5, linestyle="--")

# Connect steps vertically
for i in range(len(df) - 1):
    ax.plot([df.loc[i+1, "date"], df.loc[i+1, "date"]],
            [df.loc[i, "ofgem"], df.loc[i+1, "ofgem"]], color="#E05C5C", linewidth=2.5)
    ax.plot([df.loc[i+1, "date"], df.loc[i+1, "date"]],
            [df.loc[i, "effective"], df.loc[i+1, "effective"]], color="#3A7DC9", linewidth=2.5, linestyle="--")

# Lockdown annotation — UK lockdown began 23 March 2020
lockdown_date = datetime(2020, 3, 23)
ax.axvline(lockdown_date, color="#555555", linestyle=":", linewidth=1.5)
ax.text(lockdown_date, 1550,
        "  UK Lockdown\n  (Mar 2020)", fontsize=8.5, color="#555555", va="bottom")

# Price shock annotation — Ofgem +80% rise, Oct 2022
shock_date = datetime(2022, 10, 1)
ax.axvline(shock_date, color="#C0392B", linestyle=":", linewidth=1.5)
ax.text(shock_date, 4400, "  Energy Price\n  Shock (Oct 2022)", fontsize=8.5, color="#C0392B", va="top")

# EPG shaded region
epg_start = datetime(2022, 10, 1)
epg_end   = datetime(2023, 6, 30)
ax.axvspan(epg_start, epg_end, alpha=0.08, color="#F39C12", label="EPG active (Oct 2022–Jun 2023)")

ax.set_xlim(df["date"].min(), df["end_date"].max())
ax.set_ylim(800, 4700)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{int(x):,}"))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], color="#E05C5C", linewidth=2.5, label="Ofgem Price Cap (£/yr, typical use)"),
    Line2D([0], [0], color="#3A7DC9", linewidth=2.5, linestyle="--", label="Effective Consumer Rate (£/yr)"),
    plt.Rectangle((0, 0), 1, 1, fc="#F39C12", alpha=0.2, label="EPG active (Oct 2022–Jun 2023)"),
]
ax.legend(handles=legend_handles, loc="upper left", fontsize=9, framealpha=0.9)

ax.set_title("Ofgem Energy Price Cap & Effective Consumer Rate (2019–2026)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Date", fontsize=10)
ax.set_ylabel("Annual Bill — Typical Household (£/yr)", fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
