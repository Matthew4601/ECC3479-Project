"""
Energy Price Shock Transmission Regressions
Research question: Did the 2022 energy price shock transmit more severely into
headline CPI and household consumption in Canada, USA or UK?

Outputs
-------
  regression_headline_cpi.txt   — country-specific distributed lag, headline CPI
  regression_consumption.txt    — country-specific distributed lag, consumption
  regression_panel.txt          — pooled panel with interaction terms (USA baseline)
  regression_uk_epg.txt         — UK with/without EPG policy dummy
  regression_summary_table.csv  — key coefficients side-by-side
  regression_coefficients.png   — visual coefficient comparison
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import openpyxl

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent

# ── Load raw data ─────────────────────────────────────────────────────────────

cpi_df    = pd.read_csv(ROOT / "data/clean/CPI_Combined_Final.csv", parse_dates=["date"])
energy_df = pd.read_csv(ROOT / "data/clean/combined_energy_cpi.csv", parse_dates=["date"])
cons_df   = pd.read_csv(
    ROOT / "Clean 3/Domestic Consumption/Domestic_Consumption_Combined.csv",
    parse_dates=["date"],
)

# ── Series definitions ────────────────────────────────────────────────────────

HEADLINE_SERIES = {
    "CAN": "CPI All-items YoY % Change (Canada national)",
    "UK":  "CPIH Annual Rate, All Items 2015=100 (L55O)",
    "USA": "CPI-U All Items YoY % Change",
}

ENERGY_COL = {
    "CAN": "cpi_energy_yoy_canada",
    "UK":  "cpi_energy_yoy_uk",
    "USA": "cpi_energy_yoy_usa",
}

COUNTRY_LABELS = {"CAN": "Canada", "UK": "United Kingdom", "USA": "United States"}

SHOCK_DATE    = pd.Timestamp("2022-10-01")
EPG_START     = pd.Timestamp("2022-10-01")
EPG_END       = pd.Timestamp("2023-06-01")


# ── Data preparation helpers ──────────────────────────────────────────────────

def get_headline_cpi(country):
    return (
        cpi_df[
            (cpi_df["country"] == country) &
            (cpi_df["series"]  == HEADLINE_SERIES[country])
        ][["date", "value"]]
        .rename(columns={"value": "headline_yoy"})
        .set_index("date").sort_index()
    )


def get_energy_cpi(country, quarterly=False):
    col = ENERGY_COL[country]
    df  = (
        energy_df[["date", col]]
        .rename(columns={col: "energy_yoy"})
        .set_index("date").sort_index()
    )
    if quarterly:
        df = df.resample("QS").mean()
    return df


def get_consumption_yoy(country):
    df = (
        cons_df[cons_df["country"] == country][["date", "value"]]
        .set_index("date").sort_index()
    )
    # All three countries are quarterly in the combined file
    df["cons_yoy"] = df["value"].pct_change(4) * 100
    return df[["cons_yoy"]].dropna()


def build_epg_dummy(index):
    """Monthly binary dummy: 1 during EPG active period."""
    epg = pd.Series(0, index=index, name="epg_dummy")
    epg.loc[(index >= EPG_START) & (index <= EPG_END)] = 1
    return epg.to_frame()


# ── Distributed lag dataset builder ──────────────────────────────────────────

def build_dl(country, dep="cpi", n_lags=3, quarterly=False):
    if dep == "cpi":
        y_df   = get_headline_cpi(country)
        y_col  = "headline_yoy"
        x_df   = get_energy_cpi(country, quarterly=False)
    else:
        y_df   = get_consumption_yoy(country)
        y_col  = "cons_yoy"
        x_df   = get_energy_cpi(country, quarterly=True)

    df = y_df.join(x_df, how="inner").dropna()

    # Lags of energy CPI
    for k in range(1, n_lags + 1):
        df[f"energy_lag{k}"] = df["energy_yoy"].shift(k)

    # AR(1) term
    df[f"{y_col}_lag1"] = df[y_col].shift(1)

    # Post-shock dummy
    df["post_shock"] = (df.index >= SHOCK_DATE).astype(int)

    return df.dropna(), y_col


# ── OLS with HAC standard errors ─────────────────────────────────────────────

def run_ols(df, y_col, x_cols, label, outfile, maxlags=3):
    y      = df[y_col]
    X      = sm.add_constant(df[x_cols])
    result = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})

    energy_cols = [c for c in x_cols if "energy" in c]
    cum_pt      = sum(result.params[c] for c in energy_cols)

    outfile.write(f"\n{'='*72}\n{label}\n{'='*72}\n")
    outfile.write(result.summary().as_text())
    outfile.write(f"\n  Cumulative energy pass-through (sum β_energy): {cum_pt:+.4f}\n")

    return result, cum_pt


# ── REGRESSION 1: Headline CPI — country-specific DL ─────────────────────────

results = {}

with open(OUT_DIR / "regression_headline_cpi.txt", "w") as f:
    f.write("REGRESSION 1: ENERGY CPI → HEADLINE CPI PASS-THROUGH\n")
    f.write("Model:  ΔHeadlineCPI_t = α + Σ(k=0..3) βk·ΔEnergyCPI_(t-k)\n")
    f.write("              + λ·HeadlineCPI_(t-1) + γ·PostShock_t + ε_t\n")
    f.write("SE:     HAC (Newey-West, maxlags=3)\n")
    f.write("Dep:    Headline CPI YoY %\n")
    f.write("Freq:   Monthly\n\n")

    x_cols = ["energy_yoy", "energy_lag1", "energy_lag2", "energy_lag3",
              "headline_yoy_lag1", "post_shock"]

    for country in ["CAN", "UK", "USA"]:
        df, y_col = build_dl(country, dep="cpi")
        label     = f"{COUNTRY_LABELS[country]}  |  n = {len(df)} monthly obs  |  {df.index.min().date()} – {df.index.max().date()}"
        res, cpt  = run_ols(df, y_col, x_cols, label, f)
        results[f"cpi_{country}"] = (res, cpt)

print("Regression 1 complete")

# ── REGRESSION 2: Domestic Consumption — country-specific DL (quarterly) ─────

with open(OUT_DIR / "regression_consumption.txt", "w") as f:
    f.write("REGRESSION 2: ENERGY CPI → DOMESTIC CONSUMPTION PASS-THROUGH\n")
    f.write("Model:  ΔCons_t = α + Σ(k=0..3) βk·ΔEnergyCPI_(t-k)\n")
    f.write("            + λ·Cons_(t-1) + γ·PostShock_t + ε_t\n")
    f.write("SE:     HAC (Newey-West, maxlags=3)\n")
    f.write("Dep:    Domestic Consumption YoY %\n")
    f.write("Freq:   Quarterly (energy CPI resampled to quarterly mean)\n\n")

    x_cols_q = ["energy_yoy", "energy_lag1", "energy_lag2", "energy_lag3",
                "cons_yoy_lag1", "post_shock"]

    for country in ["CAN", "UK", "USA"]:
        df, y_col = build_dl(country, dep="cons", quarterly=True)
        label     = f"{COUNTRY_LABELS[country]}  |  n = {len(df)} quarterly obs  |  {df.index.min().date()} – {df.index.max().date()}"
        res, cpt  = run_ols(df, y_col, x_cols_q, label, f)
        results[f"cons_{country}"] = (res, cpt)

print("Regression 2 complete")

# ── REGRESSION 3: Pooled panel with interaction terms ─────────────────────────

with open(OUT_DIR / "regression_panel.txt", "w") as f:
    f.write("REGRESSION 3: POOLED PANEL — DIFFERENTIAL PASS-THROUGH (USA BASELINE)\n")
    f.write("Model:  HeadlineCPI_it = α + β·EnergyCPI_it\n")
    f.write("            + δ_CAN·(CAN × EnergyCPI_it) + δ_UK·(UK × EnergyCPI_it)\n")
    f.write("            + country FE + λ·HeadlineCPI_(t-1) + γ·PostShock_t + ε_it\n")
    f.write("SE:     HAC (Newey-West, maxlags=3)\n")
    f.write("Note:   USA is the baseline country. Interaction coefficients show\n")
    f.write("        differential pass-through relative to USA.\n\n")

    panels = []
    for country in ["CAN", "UK", "USA"]:
        df, y_col = build_dl(country, dep="cpi")
        df        = df[["headline_yoy", "energy_yoy", "headline_yoy_lag1", "post_shock"]].copy()
        df["country"] = country
        panels.append(df)

    panel = pd.concat(panels).sort_index()
    panel["CAN_dummy"]    = (panel["country"] == "CAN").astype(int)
    panel["UK_dummy"]     = (panel["country"] == "UK").astype(int)
    panel["CAN_x_energy"] = panel["CAN_dummy"] * panel["energy_yoy"]
    panel["UK_x_energy"]  = panel["UK_dummy"]  * panel["energy_yoy"]

    y_p = panel["headline_yoy"]
    X_p = sm.add_constant(panel[["energy_yoy", "CAN_x_energy", "UK_x_energy",
                                  "CAN_dummy", "UK_dummy", "headline_yoy_lag1", "post_shock"]])
    res_panel = sm.OLS(y_p, X_p).fit(cov_type="HAC", cov_kwds={"maxlags": 3})

    f.write(res_panel.summary().as_text())

    b_usa = res_panel.params["energy_yoy"]
    b_can = res_panel.params["energy_yoy"] + res_panel.params["CAN_x_energy"]
    b_uk  = res_panel.params["energy_yoy"] + res_panel.params["UK_x_energy"]

    f.write("\n\nPASS-THROUGH SUMMARY (contemporaneous β, no lags):\n")
    f.write(f"  USA baseline pass-through:         {b_usa:+.4f}  (p = {res_panel.pvalues['energy_yoy']:.4f})\n")
    f.write(f"  Canada total pass-through:         {b_can:+.4f}  (differential vs USA: {res_panel.params['CAN_x_energy']:+.4f}, p = {res_panel.pvalues['CAN_x_energy']:.4f})\n")
    f.write(f"  UK total pass-through:             {b_uk:+.4f}  (differential vs USA: {res_panel.params['UK_x_energy']:+.4f}, p = {res_panel.pvalues['UK_x_energy']:.4f})\n")

    results["panel"] = res_panel
    print("Regression 3 complete")

# ── REGRESSION 4: UK with EPG policy dummy ────────────────────────────────────

with open(OUT_DIR / "regression_uk_epg.txt", "w") as f:
    f.write("REGRESSION 4: UK — HEADLINE CPI WITH AND WITHOUT EPG POLICY DUMMY\n")
    f.write("Tests whether the EPG (Oct 2022–Jun 2023) significantly reduced\n")
    f.write("energy price pass-through to headline CPI in the UK.\n")
    f.write("EPG dummy = 1 during Oct 2022 – Jun 2023, else 0.\n\n")

    df_uk, y_col = build_dl("UK", dep="cpi")
    epg_d        = build_epg_dummy(df_uk.index)
    df_uk        = df_uk.join(epg_d)

    x_base = ["energy_yoy", "energy_lag1", "energy_lag2", "energy_lag3",
               "headline_yoy_lag1", "post_shock"]
    x_epg  = x_base + ["epg_dummy"]

    res_base, cpt_base = run_ols(df_uk, y_col, x_base, "UK — without EPG dummy", f)
    res_epg,  cpt_epg  = run_ols(df_uk, y_col, x_epg,  "UK — with EPG dummy",    f)

    epg_coef = res_epg.params["epg_dummy"]
    epg_pval = res_epg.pvalues["epg_dummy"]

    f.write("\nEPG POLICY EFFECT SUMMARY:\n")
    f.write(f"  Cumulative energy pass-through WITHOUT EPG control: {cpt_base:+.4f}\n")
    f.write(f"  Cumulative energy pass-through WITH    EPG control: {cpt_epg:+.4f}\n")
    f.write(f"  EPG coefficient:  {epg_coef:+.4f}  (p = {epg_pval:.4f})\n")
    f.write(f"  Interpretation: During the EPG period, headline CPI was on average\n")
    f.write(f"  {epg_coef:+.2f} percentage points {'lower' if epg_coef < 0 else 'higher'} than the energy-CPI model alone predicts,\n")
    f.write(f"  consistent with {'the EPG suppressing energy pass-through.' if epg_coef < 0 else 'no suppression effect.'}\n")

    results["uk_base"] = (res_base, cpt_base)
    results["uk_epg"]  = (res_epg,  cpt_epg)
    print("Regression 4 complete")

# ── Summary comparison table ──────────────────────────────────────────────────

rows = []
for country in ["CAN", "UK", "USA"]:
    r_cpi,  cpt_cpi  = results[f"cpi_{country}"]
    r_cons, cpt_cons = results[f"cons_{country}"]

    rows.append({
        "Country":                        COUNTRY_LABELS[country],
        "CPI_CumPassThrough":             round(cpt_cpi,  4),
        "CPI_R2":                         round(r_cpi.rsquared, 4),
        "CPI_PostShock_coef":             round(r_cpi.params.get("post_shock", np.nan), 4),
        "CPI_PostShock_pval":             round(r_cpi.pvalues.get("post_shock", np.nan), 4),
        "CPI_N":                          int(r_cpi.nobs),
        "Cons_CumPassThrough":            round(cpt_cons, 4),
        "Cons_R2":                        round(r_cons.rsquared, 4),
        "Cons_PostShock_coef":            round(r_cons.params.get("post_shock", np.nan), 4),
        "Cons_PostShock_pval":            round(r_cons.pvalues.get("post_shock", np.nan), 4),
        "Cons_N":                         int(r_cons.nobs),
    })

summary = pd.DataFrame(rows)
summary.to_csv(OUT_DIR / "regression_summary_table.csv", index=False)
print("\nSummary table:")
print(summary.to_string(index=False))

# ── Coefficient comparison plot ───────────────────────────────────────────────

countries  = ["Canada", "United Kingdom", "United States"]
colors     = ["#2166AC", "#E05C5C", "#4DAC26"]
x          = np.arange(len(countries))
width      = 0.35

cpt_cpi_vals  = [results[f"cpi_{c}"][1]  for c in ["CAN", "UK", "USA"]]
cpt_cons_vals = [results[f"cons_{c}"][1] for c in ["CAN", "UK", "USA"]]

# Confidence intervals for cumulative pass-through (approximate: sum of individual CIs)
def cum_ci(res, energy_cols, alpha=0.05):
    coefs = [res.params[c] for c in energy_cols if c in res.params]
    ci    = res.conf_int(alpha=alpha)
    lo    = sum(ci.loc[c, 0] for c in energy_cols if c in ci.index)
    hi    = sum(ci.loc[c, 1] for c in energy_cols if c in ci.index)
    return lo, hi

energy_cols_m = ["energy_yoy", "energy_lag1", "energy_lag2", "energy_lag3"]

cpi_lo  = [cum_ci(results[f"cpi_{c}"][0],  energy_cols_m)[0] for c in ["CAN", "UK", "USA"]]
cpi_hi  = [cum_ci(results[f"cpi_{c}"][0],  energy_cols_m)[1] for c in ["CAN", "UK", "USA"]]
cons_lo = [cum_ci(results[f"cons_{c}"][0], energy_cols_m)[0] for c in ["CAN", "UK", "USA"]]
cons_hi = [cum_ci(results[f"cons_{c}"][0], energy_cols_m)[1] for c in ["CAN", "UK", "USA"]]

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

for ax, vals, lo, hi, title, ylabel in [
    (axes[0], cpt_cpi_vals,  cpi_lo,  cpi_hi,
     "Pass-Through: Energy CPI → Headline CPI",
     "Cumulative β (pp headline CPI per pp energy CPI)"),
    (axes[1], cpt_cons_vals, cons_lo, cons_hi,
     "Pass-Through: Energy CPI → Domestic Consumption",
     "Cumulative β (pp consumption YoY per pp energy CPI)"),
]:
    bars = ax.bar(x, vals, width=0.5, color=colors, alpha=0.85, zorder=3)

    # Error bars (95% CI)
    yerr_lo = [v - l for v, l in zip(vals, lo)]
    yerr_hi = [h - v for v, h in zip(vals, hi)]
    ax.errorbar(x, vals, yerr=[yerr_lo, yerr_hi],
                fmt="none", color="black", capsize=5, linewidth=1.2, zorder=4)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(countries, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + (0.003 if val >= 0 else -0.006),
                f"{val:+.3f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=9, fontweight="bold")

fig.suptitle(
    "Cumulative Energy Price Pass-Through (2022 Shock Period)\n"
    "Distributed Lag Model, 3 lags, HAC SE — 95% Confidence Intervals shown",
    fontsize=12, fontweight="bold", y=1.01
)
plt.tight_layout()
fig.savefig(OUT_DIR / "regression_coefficients.png", dpi=150, bbox_inches="tight")
print(f"\nSaved → {OUT_DIR / 'regression_coefficients.png'}")
plt.close(fig)

print("\nAll regressions complete. Files saved to:", OUT_DIR)
