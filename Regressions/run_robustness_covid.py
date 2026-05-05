"""
Robustness Check: Consumption Regression Excluding COVID Quarters
Excludes 2020 Q1 and Q2 (Jan–Jun 2020) — the extreme collapse/rebound
quarters that inflate kurtosis and distort standard errors in the main model.

Outputs
-------
  robustness_consumption_ex_covid.txt   — full regression output, original vs ex-COVID
  robustness_comparison.png             — side-by-side coefficient comparison
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent

energy_df = pd.read_csv(ROOT / "data/clean/combined_energy_cpi.csv", parse_dates=["date"])
cons_df   = pd.read_csv(
    ROOT / "Clean 3/Domestic Consumption/Domestic_Consumption_Combined.csv",
    parse_dates=["date"],
)

ENERGY_COL     = {"CAN": "cpi_energy_yoy_canada", "UK": "cpi_energy_yoy_uk", "USA": "cpi_energy_yoy_usa"}
COUNTRY_LABELS = {"CAN": "Canada", "UK": "United Kingdom", "USA": "United States"}
SHOCK_DATE     = pd.Timestamp("2022-10-01")

# COVID exclusion windows
COVID_Q1 = (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-03-31"))
COVID_Q2 = (pd.Timestamp("2020-04-01"), pd.Timestamp("2020-06-30"))


def get_energy_quarterly(country):
    col = ENERGY_COL[country]
    return (
        energy_df[["date", col]]
        .rename(columns={col: "energy_yoy"})
        .set_index("date").sort_index()
        .resample("QS").mean()
    )


def get_consumption_yoy(country):
    df = cons_df[cons_df["country"] == country][["date", "value"]].set_index("date").sort_index()
    df["cons_yoy"] = df["value"].pct_change(4) * 100
    return df[["cons_yoy"]].dropna()


def build_dataset(country, exclude_covid=False):
    y_df = get_consumption_yoy(country)
    x_df = get_energy_quarterly(country)
    df   = y_df.join(x_df, how="inner").dropna()

    for k in range(1, 4):
        df[f"energy_lag{k}"] = df["energy_yoy"].shift(k)
    df["cons_yoy_lag1"] = df["cons_yoy"].shift(1)
    df["post_shock"]    = (df.index >= SHOCK_DATE).astype(int)
    df = df.dropna()

    if exclude_covid:
        mask = ~(
            ((df.index >= COVID_Q1[0]) & (df.index <= COVID_Q1[1])) |
            ((df.index >= COVID_Q2[0]) & (df.index <= COVID_Q2[1]))
        )
        df = df[mask]

    return df


def run_ols(df, label, outfile):
    y_col  = "cons_yoy"
    x_cols = ["energy_yoy", "energy_lag1", "energy_lag2", "energy_lag3",
               "cons_yoy_lag1", "post_shock"]
    y      = df[y_col]
    X      = sm.add_constant(df[x_cols])
    result = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})

    energy_cols = [c for c in x_cols if "energy" in c]
    cum_pt      = sum(result.params[c] for c in energy_cols)

    outfile.write(f"\n{'='*72}\n{label}\n{'='*72}\n")
    outfile.write(result.summary().as_text())
    outfile.write(f"\n  Cumulative energy pass-through: {cum_pt:+.4f}\n")
    outfile.write(f"  Kurtosis of residuals:          {result.resid.kurt():.2f}\n")

    return result, cum_pt


# ── Run regressions and write output ─────────────────────────────────────────

store = {}   # (country, version) -> (result, cum_pt)

with open(OUT_DIR / "robustness_consumption_ex_covid.txt", "w") as f:
    f.write("ROBUSTNESS CHECK: CONSUMPTION REGRESSION EXCLUDING COVID QUARTERS\n")
    f.write("Excluded observations: 2020 Q1 (Jan–Mar) and 2020 Q2 (Apr–Jun)\n")
    f.write("Rationale: extreme consumption collapse/rebound inflates kurtosis\n")
    f.write("           and distorts standard errors in the main model.\n")
    f.write("All other specifications identical to Regression 2.\n")

    for country in ["CAN", "UK", "USA"]:
        name = COUNTRY_LABELS[country]

        df_full   = build_dataset(country, exclude_covid=False)
        df_ex     = build_dataset(country, exclude_covid=True)

        label_full = f"{name} — ORIGINAL  |  n = {len(df_full)} obs  |  kurtosis check below"
        label_ex   = f"{name} — EX-COVID  |  n = {len(df_ex)} obs  (dropped {len(df_full)-len(df_ex)} quarters)"

        res_full, cpt_full = run_ols(df_full, label_full, f)
        res_ex,   cpt_ex   = run_ols(df_ex,   label_ex,   f)

        # Change summary
        f.write(f"\n  ── CHANGE SUMMARY ({name}) ──\n")
        for var in ["energy_yoy", "energy_lag1", "energy_lag2", "energy_lag3",
                    "cons_yoy_lag1", "post_shock"]:
            c_full = res_full.params.get(var, np.nan)
            c_ex   = res_ex.params.get(var, np.nan)
            p_full = res_full.pvalues.get(var, np.nan)
            p_ex   = res_ex.pvalues.get(var, np.nan)
            f.write(f"  {var:<20}  orig: {c_full:+.4f} (p={p_full:.3f})  →  "
                    f"ex-COVID: {c_ex:+.4f} (p={p_ex:.3f})  Δ={c_ex-c_full:+.4f}\n")
        f.write(f"  {'Cumulative PT':<20}  orig: {cpt_full:+.4f}  →  ex-COVID: {cpt_ex:+.4f}  "
                f"Δ={cpt_ex-cpt_full:+.4f}\n")
        f.write(f"  {'R²':<20}  orig: {res_full.rsquared:.4f}  →  ex-COVID: {res_ex.rsquared:.4f}\n")

        store[(country, "full")] = (res_full, cpt_full)
        store[(country, "ex")]   = (res_ex,   cpt_ex)

print("Robustness regressions complete")

# ── Comparison plot ───────────────────────────────────────────────────────────

countries = ["CAN", "UK", "USA"]
labels    = [COUNTRY_LABELS[c] for c in countries]
x         = np.arange(len(countries))
width     = 0.32

# Cumulative pass-through
cpt_full_vals = [store[(c, "full")][1] for c in countries]
cpt_ex_vals   = [store[(c, "ex")][1]   for c in countries]

# Post-shock coefficients
ps_full_vals = [store[(c, "full")][0].params.get("post_shock", np.nan) for c in countries]
ps_ex_vals   = [store[(c, "ex")][0].params.get("post_shock", np.nan)   for c in countries]
ps_full_pval = [store[(c, "full")][0].pvalues.get("post_shock", np.nan) for c in countries]
ps_ex_pval   = [store[(c, "ex")][0].pvalues.get("post_shock", np.nan)   for c in countries]

# Kurtosis of residuals
kurt_full = [store[(c, "full")][0].resid.kurt() for c in countries]
kurt_ex   = [store[(c, "ex")][0].resid.kurt()   for c in countries]

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle(
    "Robustness Check: Excluding COVID Quarters (2020 Q1–Q2)\n"
    "Blue = Original | Orange = Ex-COVID",
    fontsize=12, fontweight="bold"
)

C_ORIG = "#2166AC"
C_EX   = "#F4A636"

# Panel 1: cumulative pass-through
ax = axes[0]
ax.bar(x - width/2, cpt_full_vals, width, color=C_ORIG, alpha=0.85, label="Original")
ax.bar(x + width/2, cpt_ex_vals,   width, color=C_EX,   alpha=0.85, label="Ex-COVID")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
ax.set_title("Cumulative Energy Pass-Through\nto Consumption", fontsize=10, fontweight="bold")
ax.set_ylabel("Sum of β_energy coefficients", fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
for i, (v_f, v_e) in enumerate(zip(cpt_full_vals, cpt_ex_vals)):
    ax.text(i - width/2, v_f + (0.002 if v_f >= 0 else -0.004), f"{v_f:+.3f}",
            ha="center", va="bottom" if v_f >= 0 else "top", fontsize=8)
    ax.text(i + width/2, v_e + (0.002 if v_e >= 0 else -0.004), f"{v_e:+.3f}",
            ha="center", va="bottom" if v_e >= 0 else "top", fontsize=8)

# Panel 2: post-shock coefficient
ax = axes[1]
ax.bar(x - width/2, ps_full_vals, width, color=C_ORIG, alpha=0.85, label="Original")
ax.bar(x + width/2, ps_ex_vals,   width, color=C_EX,   alpha=0.85, label="Ex-COVID")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
ax.set_title("Post-Shock Dummy Coefficient\n(Oct 2022+)", fontsize=10, fontweight="bold")
ax.set_ylabel("Coefficient (pp consumption YoY)", fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
for i, (v_f, p_f, v_e, p_e) in enumerate(zip(ps_full_vals, ps_full_pval, ps_ex_vals, ps_ex_pval)):
    star_f = "**" if p_f < 0.05 else ("†" if p_f < 0.10 else "")
    star_e = "**" if p_e < 0.05 else ("†" if p_e < 0.10 else "")
    offset = 0.08
    ax.text(i - width/2, v_f + offset, f"{v_f:+.2f}{star_f}", ha="center", fontsize=8)
    ax.text(i + width/2, v_e + offset, f"{v_e:+.2f}{star_e}", ha="center", fontsize=8)
ax.text(0.98, 0.02, "** p<0.05  † p<0.10", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=7, color="grey")

# Panel 3: kurtosis of residuals
ax = axes[2]
ax.bar(x - width/2, kurt_full, width, color=C_ORIG, alpha=0.85, label="Original")
ax.bar(x + width/2, kurt_ex,   width, color=C_EX,   alpha=0.85, label="Ex-COVID")
ax.axhline(3, color="black", linewidth=1, linestyle="--", alpha=0.5, label="Normal = 3")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
ax.set_title("Residual Kurtosis\n(Normal = 3)", fontsize=10, fontweight="bold")
ax.set_ylabel("Excess kurtosis of residuals", fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
for i, (k_f, k_e) in enumerate(zip(kurt_full, kurt_ex)):
    ax.text(i - width/2, k_f + 0.3, f"{k_f:.1f}", ha="center", fontsize=8)
    ax.text(i + width/2, k_e + 0.3, f"{k_e:.1f}", ha="center", fontsize=8)

for ax in axes:
    ax.legend(fontsize=8, loc="upper right")

plt.tight_layout()
out = OUT_DIR / "robustness_comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig)
