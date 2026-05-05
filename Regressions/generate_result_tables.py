"""
Generate clean result tables for each regression.
One file per regression, one table per country, showing:
  regressor name | coefficient | standard error | p-value
  plus R², N, and sample date range.
"""

from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
import openpyxl

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent

# ── Load data ─────────────────────────────────────────────────────────────────

cpi_df    = pd.read_csv(ROOT / "data/clean/CPI_Combined_Final.csv",    parse_dates=["date"])
energy_df = pd.read_csv(ROOT / "data/clean/combined_energy_cpi.csv",   parse_dates=["date"])
cons_df   = pd.read_csv(
    ROOT / "Clean 3/Domestic Consumption/Domestic_Consumption_Combined.csv",
    parse_dates=["date"],
)

HEADLINE_SERIES = {
    "CAN": "CPI All-items YoY % Change (Canada national)",
    "UK":  "CPIH Annual Rate, All Items 2015=100 (L55O)",
    "USA": "CPI-U All Items YoY % Change",
}
ENERGY_COL     = {"CAN": "cpi_energy_yoy_canada", "UK": "cpi_energy_yoy_uk", "USA": "cpi_energy_yoy_usa"}
COUNTRY_LABELS = {"CAN": "Canada", "UK": "United Kingdom", "USA": "United States"}
SHOCK_DATE     = pd.Timestamp("2022-10-01")
EPG_START      = pd.Timestamp("2022-10-01")
EPG_END        = pd.Timestamp("2023-06-01")

REGRESSOR_LABELS = {
    "const":                "Constant",
    "energy_yoy":           "Energy CPI (t)",
    "energy_lag1":          "Energy CPI (t-1)",
    "energy_lag2":          "Energy CPI (t-2)",
    "energy_lag3":          "Energy CPI (t-3)",
    "headline_yoy_lag1":    "Headline CPI (t-1)",
    "cons_yoy_lag1":        "Consumption (t-1)",
    "post_shock":           "Post-Shock Dummy",
    "epg_dummy":            "EPG Dummy",
    "CAN_x_energy":         "Canada × Energy CPI",
    "UK_x_energy":          "UK × Energy CPI",
    "CAN_dummy":            "Canada FE",
    "UK_dummy":             "UK FE",
}

# ── Data helpers ──────────────────────────────────────────────────────────────

def get_headline(country):
    return (
        cpi_df[(cpi_df["country"] == country) & (cpi_df["series"] == HEADLINE_SERIES[country])]
        [["date","value"]].rename(columns={"value":"headline_yoy"}).set_index("date").sort_index()
    )

def get_energy(country, quarterly=False):
    col = ENERGY_COL[country]
    df  = energy_df[["date", col]].rename(columns={col:"energy_yoy"}).set_index("date").sort_index()
    return df.resample("QS").mean() if quarterly else df

def get_cons(country):
    df = cons_df[cons_df["country"] == country][["date","value"]].set_index("date").sort_index()
    df["cons_yoy"] = df["value"].pct_change(4) * 100
    return df[["cons_yoy"]].dropna()

def build_cpi_data(country):
    df = get_headline(country).join(get_energy(country), how="inner").dropna()
    for k in range(1, 4):
        df[f"energy_lag{k}"] = df["energy_yoy"].shift(k)
    df["headline_yoy_lag1"] = df["headline_yoy"].shift(1)
    df["post_shock"]        = (df.index >= SHOCK_DATE).astype(int)
    return df.dropna()

def build_cons_data(country):
    df = get_cons(country).join(get_energy(country, quarterly=True), how="inner").dropna()
    for k in range(1, 4):
        df[f"energy_lag{k}"] = df["energy_yoy"].shift(k)
    df["cons_yoy_lag1"] = df["cons_yoy"].shift(1)
    df["post_shock"]    = (df.index >= SHOCK_DATE).astype(int)
    return df.dropna()

def run(df, y_col, x_cols):
    y = df[y_col]
    X = sm.add_constant(df[x_cols])
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})

# ── Table formatter ───────────────────────────────────────────────────────────

def fmt_table(result, x_cols, y_col, date_start, date_end, extra_rows=None):
    """Return a formatted string table for one regression result."""
    rows = []
    all_cols = ["const"] + x_cols
    for col in all_cols:
        label = REGRESSOR_LABELS.get(col, col)
        coef  = result.params[col]
        se    = result.bse[col]
        pval  = result.pvalues[col]
        stars = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.10 else ""))
        rows.append((label, f"{coef:+.4f}{stars}", f"{se:.4f}", f"{pval:.4f}"))

    # Cumulative pass-through
    energy_cols = [c for c in x_cols if "energy" in c and "dummy" not in c
                   and "CAN" not in c and "UK" not in c]
    cum_pt = sum(result.params[c] for c in energy_cols)

    col_w = [max(len(r[i]) for r in rows + [("Regressor","Coefficient","Std. Error","p-value")]) + 2
             for i in range(4)]
    col_w[0] = max(col_w[0], 28)

    sep  = "+" + "+".join("-" * w for w in col_w) + "+"
    head = "|" + "|".join(h.center(col_w[i]) for i, h in
                           enumerate(["Regressor","Coefficient","Std. Error","p-value"])) + "|"

    lines = [sep, head, sep]
    for r in rows:
        lines.append("|" + "|".join(v.center(col_w[i]) for i, v in enumerate(r)) + "|")
    lines.append(sep)

    lines.append(f"  Cumulative energy pass-through : {cum_pt:+.4f}")
    lines.append(f"  R-squared                      : {result.rsquared:.4f}")
    lines.append(f"  N (observations)               : {int(result.nobs)}")
    lines.append(f"  Sample period                  : {date_start} to {date_end}")
    if extra_rows:
        for row in extra_rows:
            lines.append(f"  {row}")
    lines.append("  * p<0.10   ** p<0.05   *** p<0.01   (HAC SE, Newey-West m=3)")

    return "\n".join(lines)

# ── Regression 1: Headline CPI ────────────────────────────────────────────────

x_cpi = ["energy_yoy","energy_lag1","energy_lag2","energy_lag3",
          "headline_yoy_lag1","post_shock"]

with open(OUT_DIR / "reg1_headline_cpi_tables.txt", "w") as f:
    f.write("=" * 70 + "\n")
    f.write("REGRESSION 1 — ENERGY CPI PASS-THROUGH TO HEADLINE CPI\n")
    f.write("Dependent variable: Headline CPI YoY %  |  Frequency: Monthly\n")
    f.write("=" * 70 + "\n")

    for country in ["CAN", "UK", "USA"]:
        df  = build_cpi_data(country)
        res = run(df, "headline_yoy", x_cpi)
        d0  = df.index.min().strftime("%b %Y")
        d1  = df.index.max().strftime("%b %Y")

        f.write(f"\n{'─'*70}\n")
        f.write(f"Country: {COUNTRY_LABELS[country]}\n")
        f.write(f"{'─'*70}\n")
        f.write(fmt_table(res, x_cpi, "headline_yoy", d0, d1))
        f.write("\n")

print("Saved reg1_headline_cpi_tables.txt")

# ── Regression 2: Consumption ─────────────────────────────────────────────────

x_cons = ["energy_yoy","energy_lag1","energy_lag2","energy_lag3",
           "cons_yoy_lag1","post_shock"]

with open(OUT_DIR / "reg2_consumption_tables.txt", "w") as f:
    f.write("=" * 70 + "\n")
    f.write("REGRESSION 2 — ENERGY CPI PASS-THROUGH TO DOMESTIC CONSUMPTION\n")
    f.write("Dependent variable: Consumption YoY %  |  Frequency: Quarterly\n")
    f.write("Note: Energy CPI resampled to quarterly mean\n")
    f.write("=" * 70 + "\n")

    for country in ["CAN", "UK", "USA"]:
        df  = build_cons_data(country)
        res = run(df, "cons_yoy", x_cons)
        d0  = df.index.min().strftime("%b %Y")
        d1  = df.index.max().strftime("%b %Y")

        f.write(f"\n{'─'*70}\n")
        f.write(f"Country: {COUNTRY_LABELS[country]}\n")
        f.write(f"{'─'*70}\n")
        f.write(fmt_table(res, x_cons, "cons_yoy", d0, d1))
        f.write("\n")

print("Saved reg2_consumption_tables.txt")

# ── Regression 3: Pooled panel ────────────────────────────────────────────────

x_panel = ["energy_yoy","CAN_x_energy","UK_x_energy",
            "CAN_dummy","UK_dummy","headline_yoy_lag1","post_shock"]

with open(OUT_DIR / "reg3_panel_tables.txt", "w") as f:
    f.write("=" * 70 + "\n")
    f.write("REGRESSION 3 — POOLED PANEL: DIFFERENTIAL PASS-THROUGH\n")
    f.write("Dependent variable: Headline CPI YoY %  |  Baseline: USA\n")
    f.write("Frequency: Monthly  |  All countries pooled\n")
    f.write("=" * 70 + "\n")

    panels = []
    for country in ["CAN", "UK", "USA"]:
        df = build_cpi_data(country)[["headline_yoy","energy_yoy",
                                       "headline_yoy_lag1","post_shock"]].copy()
        df["country"] = country
        panels.append(df)

    panel = pd.concat(panels).sort_index()
    panel["CAN_dummy"]    = (panel["country"] == "CAN").astype(int)
    panel["UK_dummy"]     = (panel["country"] == "UK").astype(int)
    panel["CAN_x_energy"] = panel["CAN_dummy"] * panel["energy_yoy"]
    panel["UK_x_energy"]  = panel["UK_dummy"]  * panel["energy_yoy"]

    res_p = run(panel, "headline_yoy", x_panel)
    d0    = panel.index.min().strftime("%b %Y")
    d1    = panel.index.max().strftime("%b %Y")

    b_usa = res_p.params["energy_yoy"]
    b_can = b_usa + res_p.params["CAN_x_energy"]
    b_uk  = b_usa + res_p.params["UK_x_energy"]

    extras = [
        f"Total pass-through — USA     : {b_usa:+.4f}",
        f"Total pass-through — Canada  : {b_can:+.4f}  (baseline {b_usa:+.4f} + differential {res_p.params['CAN_x_energy']:+.4f})",
        f"Total pass-through — UK      : {b_uk:+.4f}  (baseline {b_usa:+.4f} + differential {res_p.params['UK_x_energy']:+.4f})",
    ]

    f.write(f"\n{'─'*70}\n")
    f.write("Pooled (Canada + UK + USA)\n")
    f.write(f"{'─'*70}\n")
    f.write(fmt_table(res_p, x_panel, "headline_yoy", d0, d1, extra_rows=extras))
    f.write("\n")

print("Saved reg3_panel_tables.txt")

# ── Regression 4: UK with EPG dummy ──────────────────────────────────────────

x_base = ["energy_yoy","energy_lag1","energy_lag2","energy_lag3",
           "headline_yoy_lag1","post_shock"]
x_epg  = x_base + ["epg_dummy"]

with open(OUT_DIR / "reg4_uk_epg_tables.txt", "w") as f:
    f.write("=" * 70 + "\n")
    f.write("REGRESSION 4 — UK HEADLINE CPI WITH EPG POLICY DUMMY\n")
    f.write("Dependent variable: Headline CPI YoY %  |  Frequency: Monthly\n")
    f.write("EPG dummy = 1 during October 2022 – June 2023\n")
    f.write("=" * 70 + "\n")

    df_uk = build_cpi_data("UK")
    epg   = pd.Series(0, index=df_uk.index, name="epg_dummy")
    epg.loc[(df_uk.index >= EPG_START) & (df_uk.index <= EPG_END)] = 1
    df_uk = df_uk.join(epg.to_frame())
    d0    = df_uk.index.min().strftime("%b %Y")
    d1    = df_uk.index.max().strftime("%b %Y")

    res_base = run(df_uk, "headline_yoy", x_base)
    res_epg  = run(df_uk, "headline_yoy", x_epg)

    cpt_base = sum(res_base.params[c] for c in x_base if "energy" in c)
    cpt_epg  = sum(res_epg.params[c]  for c in x_epg  if "energy" in c
                   and c != "epg_dummy")

    f.write(f"\n{'─'*70}\n")
    f.write("United Kingdom — Without EPG Dummy\n")
    f.write(f"{'─'*70}\n")
    f.write(fmt_table(res_base, x_base, "headline_yoy", d0, d1))
    f.write("\n")

    f.write(f"\n{'─'*70}\n")
    f.write("United Kingdom — With EPG Dummy\n")
    f.write(f"{'─'*70}\n")
    f.write(fmt_table(res_epg, x_epg, "headline_yoy", d0, d1,
                      extra_rows=[
                          f"EPG effect on cumulative PT      : {cpt_epg - cpt_base:+.4f}",
                      ]))
    f.write("\n")

print("Saved reg4_uk_epg_tables.txt")
print("\nAll result tables generated.")
