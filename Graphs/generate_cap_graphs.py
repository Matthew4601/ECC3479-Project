from pathlib import Path
import openpyxl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from datetime import datetime

ROOT      = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "UK Energy Price Cap.xlsx"
OUT_DIR   = Path(__file__).resolve().parent

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
df["end_date"] = df["date"].shift(-1).fillna(df["date"] + pd.DateOffset(months=3))

LOCKDOWN_DATE = datetime(2020, 3, 23)
SHOCK_DATE    = datetime(2022, 10, 1)
EPG_START     = datetime(2022, 10, 1)
EPG_END       = datetime(2023, 6, 30)


def draw_step(ax, df, col, color, linestyle="-"):
    for _, row in df.iterrows():
        ax.plot([row["date"], row["end_date"]], [row[col], row[col]],
                color=color, linewidth=2.5, linestyle=linestyle)
    for i in range(len(df) - 1):
        ax.plot([df.loc[i+1, "date"], df.loc[i+1, "date"]],
                [df.loc[i, col], df.loc[i+1, col]],
                color=color, linewidth=2.5, linestyle=linestyle)


def apply_common_style(ax, title, ylabel, ylim=(800, 4700)):
    ax.set_xlim(df["date"].min(), df["end_date"].max())
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{int(x):,}"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)


# ── Graph 1: OFGEM Cap ────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(13, 6))

draw_step(ax1, df, "ofgem", "#E05C5C")

ax1.axvline(LOCKDOWN_DATE, color="#555555", linestyle=":", linewidth=1.5)
ax1.text(LOCKDOWN_DATE, 1550, "  UK Lockdown\n  (Mar 2020)",
         fontsize=8.5, color="#555555", va="bottom")

ax1.axvline(SHOCK_DATE, color="#C0392B", linestyle=":", linewidth=1.5)
ax1.text(SHOCK_DATE, 4400, "  Energy Price\n  Shock (Oct 2022)",
         fontsize=8.5, color="#C0392B", va="top")

ax1.axvspan(EPG_START, EPG_END, alpha=0.08, color="#F39C12")

legend_handles = [
    Line2D([0], [0], color="#E05C5C", linewidth=2.5, label="Ofgem Price Cap (£/yr, typical use)"),
    plt.Rectangle((0, 0), 1, 1, fc="#F39C12", alpha=0.2, label="EPG active (Oct 2022–Jun 2023)"),
]
ax1.legend(handles=legend_handles, loc="upper left", fontsize=9, framealpha=0.9)

apply_common_style(ax1,
    title="Ofgem Energy Price Cap (2019–2026)",
    ylabel="Ofgem Price Cap — Typical Household (£/yr)")

plt.tight_layout()
out1 = OUT_DIR / "Graph OFGEM Cap.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved → {out1}")
plt.close(fig1)


# ── Graph 2: Effective Consumer Rate ─────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(13, 6))

draw_step(ax2, df, "effective", "#3A7DC9", linestyle="--")

ax2.axvline(LOCKDOWN_DATE, color="#555555", linestyle=":", linewidth=1.5)
ax2.text(LOCKDOWN_DATE, 1550, "  UK Lockdown\n  (Mar 2020)",
         fontsize=8.5, color="#555555", va="bottom")

ax2.axvline(SHOCK_DATE, color="#C0392B", linestyle=":", linewidth=1.5)
ax2.text(SHOCK_DATE, 4400, "  Energy Price\n  Shock (Oct 2022)",
         fontsize=8.5, color="#C0392B", va="top")

ax2.axvspan(EPG_START, EPG_END, alpha=0.08, color="#F39C12")

legend_handles2 = [
    Line2D([0], [0], color="#3A7DC9", linewidth=2.5, linestyle="--",
           label="Effective Consumer Rate (£/yr)"),
    plt.Rectangle((0, 0), 1, 1, fc="#F39C12", alpha=0.2, label="EPG active (Oct 2022–Jun 2023)"),
]
ax2.legend(handles=legend_handles2, loc="upper left", fontsize=9, framealpha=0.9)

apply_common_style(ax2,
    title="Effective Consumer Energy Rate (2019–2026)",
    ylabel="Effective Consumer Rate — Typical Household (£/yr)")

plt.tight_layout()
out2 = OUT_DIR / "Graph Effective Rate.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved → {out2}")
plt.close(fig2)
