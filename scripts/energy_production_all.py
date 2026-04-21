"""
Visualise IEA total energy supply by fuel type for Canada, UK, and USA.

Two rows per country:
  Row 1 — Stacked area chart: absolute supply in TJ
  Row 2 — Stacked area chart: percentage share of total supply

Produces a single 3-column × 2-row figure (6 panels total).
The year 2022 is marked with a vertical dashed line on every panel.

Source : Clean 3/Domestic Consumption/combined_energy.csv
Output : Clean 3/Domestic Consumption/Energy Production All.png
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT     = Path(__file__).resolve().parent.parent
DATA_F   = ROOT / "Clean 3" / "Domestic Consumption" / "combined_energy.csv"
OUT_PATH = ROOT / "Clean 3" / "Domestic Consumption" / "Energy Production All.png"

FUEL_TYPES = {
    "coal":                  ("Coal",                 "#4E4E4E"),
    "oil":                   ("Oil",                  "#8B4513"),
    "natural_gas":           ("Natural Gas",           "#FF8C00"),
    "nuclear":               ("Nuclear",               "#9467BD"),
    "hydropower":            ("Hydropower",            "#1F77B4"),
    "solar_wind_renewables": ("Solar, Wind & Other",   "#2CA02C"),
    "biofuels_waste":        ("Biofuels & Waste",      "#8FBC8F"),
}

COUNTRIES = {
    "canada": "Canada",
    "uk":     "United Kingdom",
    "usa":    "United States",
}


def get_country_df(df, prefix):
    cols = {f"{prefix}_{fuel}": label
            for fuel, (label, _) in FUEL_TYPES.items()}
    sub  = df[["year"] + list(cols.keys())].copy()
    sub  = sub.rename(columns=cols).set_index("year")
    return sub


def add_2022_line(ax, y_max, is_pct=False):
    ax.axvline(2022, color="red", linewidth=1.5, linestyle="--",
               alpha=0.85, zorder=5)
    y_pos = 93 if is_pct else y_max * 0.93
    ax.text(2021.7, y_pos, "2022", color="red", fontsize=8.5,
            fontweight="bold", ha="right", va="top", zorder=6)


def main():
    df     = pd.read_csv(DATA_F)
    colors = [c for _, (_, c) in FUEL_TYPES.items()]
    labels = [lbl for _, (lbl, _) in FUEL_TYPES.items()]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(22, 13),
        gridspec_kw={"hspace": 0.50, "wspace": 0.32}
    )

    fig.suptitle(
        "IEA Total Energy Supply by Fuel Type  (2000–2024)",
        fontsize=15, fontweight="bold", y=1.002
    )

    # Row titles using a text object in figure coordinates
    fig.text(0.01, 0.74, "Absolute Supply (TJ)", fontsize=11,
             fontweight="bold", rotation=90, va="center", ha="center")
    fig.text(0.01, 0.27, "Share of Total Supply (%)", fontsize=11,
             fontweight="bold", rotation=90, va="center", ha="center")

    for col_idx, (prefix, country_name) in enumerate(COUNTRIES.items()):
        sub = get_country_df(df, prefix)
        pct = sub.div(sub.sum(axis=1), axis=0) * 100

        # ── Row 0: absolute TJ ───────────────────────────────────────────────
        ax_abs = axes[0, col_idx]
        ax_abs.stackplot(sub.index, sub.T.values,
                         labels=labels, colors=colors, alpha=0.88)

        y_max = sub.sum(axis=1).max()
        add_2022_line(ax_abs, y_max, is_pct=False)

        ax_abs.set_title(country_name, fontsize=12, fontweight="bold", pad=8)
        ax_abs.set_xlabel("Year", fontsize=10)
        ax_abs.set_ylabel("Energy Supply (TJ)", fontsize=10)
        ax_abs.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M TJ"))
        ax_abs.set_xlim(sub.index.min(), sub.index.max())
        ax_abs.tick_params(axis="x", labelsize=9)
        ax_abs.tick_params(axis="y", labelsize=9)
        ax_abs.grid(True, axis="y", alpha=0.25, linestyle=":")
        ax_abs.margins(x=0)

        # ── Row 1: percentage share ──────────────────────────────────────────
        ax_pct = axes[1, col_idx]
        ax_pct.stackplot(pct.index, pct.T.values,
                         labels=labels, colors=colors, alpha=0.88)

        add_2022_line(ax_pct, 100, is_pct=True)

        # Label each segment at its midpoint in the final year — inside only
        cumsum = 0.0
        for fuel_lbl, color in zip(labels, colors):
            val = pct[fuel_lbl].iloc[-1]
            if val >= 5:
                y_mid = cumsum + val / 2
                ax_pct.text(
                    sub.index.max() - 0.3, y_mid,
                    f"{val:.0f}%",
                    fontsize=8, color="white", fontweight="bold",
                    va="center", ha="right", zorder=6
                )
            cumsum += val

        ax_pct.set_xlabel("Year", fontsize=10)
        ax_pct.set_ylabel("Share of Total (%)", fontsize=10)
        ax_pct.set_ylim(0, 100)
        ax_pct.set_xlim(pct.index.min(), pct.index.max())
        ax_pct.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax_pct.tick_params(axis="x", labelsize=9)
        ax_pct.tick_params(axis="y", labelsize=9)
        ax_pct.grid(True, axis="y", alpha=0.25, linestyle=":")
        ax_pct.margins(x=0)

    # Shared legend centred below all panels
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels,
        loc="lower center", ncol=7,
        fontsize=10, framealpha=0.95,
        bbox_to_anchor=(0.5, -0.032),
        columnspacing=1.2, handlelength=1.5
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
