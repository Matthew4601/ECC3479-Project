import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
import warnings
warnings.filterwarnings('ignore')

OUTPUT = "NEW/NEW REGRESSIONS.txt"

# ── Data loading ──────────────────────────────────────────────────────────────

core = pd.read_excel("data/raw/ALL CORE.xlsx")
core["date"] = pd.to_datetime(core["date"])

def load_energy(path):
    df = pd.read_excel(path, sheet_name="Monthly", header=0)
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    return df

can_e = load_energy("data/raw/CANCPIENGMINMEI.xlsx CAN ENERGY.xlsx")
uk_e  = load_energy("data/raw/GBRCPIENGMINMEI.xlsx UK ENERGY.xlsx")
us_e  = load_energy("data/raw/CPGREN01USM659N.xlsx US ENERGY.xlsx")

can_e.columns = ["date", "Canada_Energy"]
uk_e.columns  = ["date", "UK_Energy"]
us_e.columns  = ["date", "USA_Energy"]

df = core.merge(can_e, on="date", how="left") \
         .merge(uk_e,  on="date", how="left") \
         .merge(us_e,  on="date", how="left")

df = df.dropna(subset=["Canada_Core","UK_Core","USA_Core",
                        "Canada_Energy","UK_Energy","USA_Energy"])
df = df.sort_values("date").reset_index(drop=True)

# Annual energy gen mix (for panel interaction)
gen = pd.read_excel("NEW/ENERGY GEN NEW.xlsx")
gen["year"] = gen["year"].astype(int)

COUNTRIES = ["Canada", "UK", "USA"]
CORE_COL  = {"Canada": "Canada_Core", "UK": "UK_Core", "USA": "USA_Core"}
ENERGY_COL = {"Canada": "Canada_Energy", "UK": "UK_Energy", "USA": "USA_Energy"}

SEP  = "\n" + "="*80 + "\n"
SEP2 = "\n" + "-"*60 + "\n"

lines = []

def hdr(title):
    lines.append(SEP)
    lines.append(title)
    lines.append("="*80)

def sub(title):
    lines.append(SEP2)
    lines.append(title)
    lines.append("-"*60)

def add(text):
    lines.append(str(text))

# ── REGRESSION 1 ─────────────────────────────────────────────────────────────
hdr("REGRESSION 1: BASELINE PASS-THROUGH (Country-by-Country OLS)")
add("Model: CoreCPI_t = α + β·EnergyCPI_t + ε_t")
add("Purpose: Establishes baseline contemporaneous pass-through per country.")

for country in COUNTRIES:
    sub(f"Country: {country}")
    y = df[CORE_COL[country]]
    X = sm.add_constant(df[ENERGY_COL[country]])
    res = sm.OLS(y, X).fit(cov_type="HC3")
    add(res.summary())

# ── REGRESSION 2 ─────────────────────────────────────────────────────────────
hdr("REGRESSION 2: ADL PASS-THROUGH WITH LAGS (12 lags, per country)")
add("Model: CoreCPI_t = α + Σ(k=0..12) βₖ·EnergyCPI_{t-k} + γ·CoreCPI_{t-1} + ε_t")
add("Purpose: Captures dynamic/delayed pass-through. Cumulative β = total effect.")

for country in COUNTRIES:
    sub(f"Country: {country}")
    c_col = CORE_COL[country]
    e_col = ENERGY_COL[country]
    tmp = df[["date", c_col, e_col]].copy().dropna()
    tmp["core_lag1"] = tmp[c_col].shift(1)
    for k in range(13):
        tmp[f"energy_lag{k}"] = tmp[e_col].shift(k)
    tmp = tmp.dropna()
    y = tmp[c_col]
    lag_cols = ["core_lag1"] + [f"energy_lag{k}" for k in range(13)]
    X = sm.add_constant(tmp[lag_cols])
    res = sm.OLS(y, X).fit(cov_type="HC3")
    add(res.summary())
    cumulative_beta = sum(res.params[f"energy_lag{k}"] for k in range(13))
    add(f"\n  >>> Cumulative pass-through (Σβ lags 0-12): {cumulative_beta:.4f}")
    add(f"  >>> Interpretation: a 1pp rise in energy CPI sustained over 12 months")
    add(f"      is associated with a {cumulative_beta:.4f}pp change in core CPI for {country}.")

# ── REGRESSION 3 ─────────────────────────────────────────────────────────────
hdr("REGRESSION 3: STRUCTURAL BREAK — 2022 REGIME INTERACTION (per country)")
add("Model: CoreCPI_t = α + β·EnergyCPI_t + δ·(D2022×EnergyCPI_t) + θ·D2022 + ε_t")
add("D2022 = 1 from January 2022 onwards.")
add("Purpose: Tests whether the pass-through coefficient changed in the 2022 episode.")
add("A significant δ means the transmission mechanism itself shifted, not just the shock size.")

for country in COUNTRIES:
    sub(f"Country: {country}")
    c_col = CORE_COL[country]
    e_col = ENERGY_COL[country]
    tmp = df[["date", c_col, e_col]].dropna().copy()
    tmp["D2022"] = (tmp["date"] >= "2022-01-01").astype(int)
    tmp["D2022_x_energy"] = tmp["D2022"] * tmp[e_col]
    X = sm.add_constant(tmp[[e_col, "D2022_x_energy", "D2022"]])
    res = sm.OLS(tmp[c_col], X).fit(cov_type="HC3")
    add(res.summary())
    beta_pre  = res.params[e_col]
    beta_post = res.params[e_col] + res.params["D2022_x_energy"]
    add(f"\n  >>> Pre-2021 pass-through (β):          {beta_pre:.4f}")
    add(f"  >>> Post-2021 pass-through (β + δ):     {beta_post:.4f}")
    add(f"  >>> Change in pass-through (δ):         {res.params['D2022_x_energy']:.4f}  "
        f"p={res.pvalues['D2022_x_energy']:.4f}")

# ── REGRESSION 4 ─────────────────────────────────────────────────────────────
hdr("REGRESSION 4: PANEL REGRESSION — COUNTRY FIXED EFFECTS + FOSSIL SHARE INTERACTION")
add("Model A: CoreCPI_it = αᵢ + β·EnergyCPI_it + εᵢₜ   [country FE only]")
add("Model B: CoreCPI_it = αᵢ + β·EnergyCPI_it + δ·(EnergyCPI_it × FossilShare_i) + εᵢₜ")
add("Purpose: Pools all three countries. FE absorbs structural level differences.")
add("Interaction tests whether higher fossil electricity share amplifies pass-through.")

# Build long panel
panels = []
for country in COUNTRIES:
    tmp = df[["date", CORE_COL[country], ENERGY_COL[country]]].copy()
    tmp.columns = ["date", "core", "energy"]
    tmp["country"] = country
    tmp["year"] = tmp["date"].dt.year
    panels.append(tmp)
panel = pd.concat(panels, ignore_index=True)

# Merge annual fossil share
fossil = gen[gen["country"].isin(COUNTRIES)][["country","year","fossil_share_elec","gas_share_elec","hydro_share_elec"]].copy()
panel = panel.merge(fossil, on=["country","year"], how="left")
panel = panel.dropna(subset=["core","energy","fossil_share_elec"])
panel = panel.set_index(["country","date"])

sub("Model A — Country Fixed Effects only")
exog_a = sm.add_constant(panel[["energy"]])
res_a = PanelOLS(panel["core"], exog_a, entity_effects=True).fit(cov_type="robust")
add(res_a.summary)

sub("Model B — Country FE + Fossil Share Interaction")
panel["energy_x_fossil"] = panel["energy"] * panel["fossil_share_elec"]
exog_b = sm.add_constant(panel[["energy","energy_x_fossil"]])
res_b = PanelOLS(panel["core"], exog_b, entity_effects=True).fit(cov_type="robust")
add(res_b.summary)
add(f"\n  >>> β (base pass-through):              {res_b.params['energy']:.4f}")
add(f"  >>> δ (fossil share amplification):     {res_b.params['energy_x_fossil']:.4f}  "
    f"p={res_b.pvalues['energy_x_fossil']:.4f}")
add("  >>> Interpretation: a 1pp higher fossil share of electricity multiplies")
add(f"      the energy→core pass-through by an additional {res_b.params['energy_x_fossil']:.4f} per pp.")

# ── REGRESSION 5 ─────────────────────────────────────────────────────────────
hdr("REGRESSION 5: ROLLING WINDOW PASS-THROUGH (60-month window, per country)")
add("Model: CoreCPI_t = α + β·EnergyCPI_t + ε_t  estimated in rolling 60-month windows.")
add("Purpose: Tracks when the pass-through relationship changed and whether it persisted.")

for country in COUNTRIES:
    sub(f"Country: {country}")
    c_col = CORE_COL[country]
    e_col = ENERGY_COL[country]
    tmp = df[["date", c_col, e_col]].dropna().reset_index(drop=True)
    betas, dates = [], []
    W = 60
    for i in range(W, len(tmp)):
        window = tmp.iloc[i-W:i]
        y_w = window[c_col]
        X_w = sm.add_constant(window[e_col])
        try:
            b = sm.OLS(y_w, X_w).fit().params[e_col]
        except Exception:
            b = np.nan
        betas.append(b)
        dates.append(tmp.loc[i, "date"])

    roll = pd.DataFrame({"date": dates, "beta": betas})
    roll["date"] = pd.to_datetime(roll["date"])

    # Summarise by key periods
    periods = [
        ("1976-01","1985-12","Post-1970s oil shocks"),
        ("1986-01","2000-12","Great Moderation"),
        ("2001-01","2010-12","2000s"),
        ("2011-01","2019-12","Pre-COVID"),
        ("2020-01","2022-12","COVID/2022 shock"),
        ("2023-01","2025-12","Post-shock"),
    ]
    add(f"\n  {'Period':<30} {'Mean β':>10} {'Min β':>10} {'Max β':>10}")
    add("  " + "-"*62)
    for start, end, label in periods:
        sub_r = roll[(roll["date"] >= start) & (roll["date"] <= end)]["beta"]
        if len(sub_r):
            add(f"  {label:<30} {sub_r.mean():>10.4f} {sub_r.min():>10.4f} {sub_r.max():>10.4f}")

    peak_row = roll.loc[roll["beta"].idxmax()]
    add(f"\n  >>> Peak β: {peak_row['beta']:.4f} in {peak_row['date'].strftime('%Y-%m')}")
    trough_row = roll.loc[roll["beta"].idxmin()]
    add(f"  >>> Trough β: {trough_row['beta']:.4f} in {trough_row['date'].strftime('%Y-%m')}")

# ── REGRESSION 6 ─────────────────────────────────────────────────────────────
hdr("REGRESSION 6: INFLATION EPISODE COMPARISON (Panel with episode dummies × energy)")
add("Model: CoreCPI_it = αᵢ + β·EnergyCPI_it + Σ γₑ·(Dₑ×EnergyCPI_it) + εᵢₜ")
add("Episodes: D1973 (1973-01 to 1975-12), D1979 (1979-01 to 1982-12),")
add("          D2022 (2022-01 to 2023-12).")
add("Purpose: Compares 2022 pass-through against the 1973 and 1979 oil shocks.")
add("A non-significant γ₂₀₂₂ vs γ₁₉₇₃ means 2022 was not structurally novel.")

panel2 = pd.concat(panels, ignore_index=True).dropna(subset=["core","energy"])

panel2["D1973"] = ((panel2["date"] >= "1973-01-01") & (panel2["date"] <= "1975-12-31")).astype(int)
panel2["D1979"] = ((panel2["date"] >= "1979-01-01") & (panel2["date"] <= "1982-12-31")).astype(int)
panel2["D2022"] = ((panel2["date"] >= "2022-01-01") & (panel2["date"] <= "2023-12-31")).astype(int)

panel2["D1973_x_e"] = panel2["D1973"] * panel2["energy"]
panel2["D1979_x_e"] = panel2["D1979"] * panel2["energy"]
panel2["D2022_x_e"] = panel2["D2022"] * panel2["energy"]

panel2 = panel2.set_index(["country","date"])

sub("Full panel (all three countries pooled)")
exog6 = sm.add_constant(panel2[["energy","D1973_x_e","D1979_x_e","D2022_x_e",
                                  "D1973","D1979","D2022"]])
res6 = PanelOLS(panel2["core"], exog6, entity_effects=True).fit(cov_type="robust")
add(res6.summary)

b      = res6.params["energy"]
g1973  = res6.params["D1973_x_e"]
g1979  = res6.params["D1979_x_e"]
g2022  = res6.params["D2022_x_e"]
p1973  = res6.pvalues["D1973_x_e"]
p1979  = res6.pvalues["D1979_x_e"]
p2022  = res6.pvalues["D2022_x_e"]

add(f"\n  >>> Baseline pass-through (β):             {b:.4f}")
add(f"  >>> Extra pass-through in 1973 ep (γ₁₉₇₃): {g1973:.4f}  p={p1973:.4f}")
add(f"  >>> Extra pass-through in 1979 ep (γ₁₉₇₉): {g1979:.4f}  p={p1979:.4f}")
add(f"  >>> Extra pass-through in 2022 ep (γ₂₀₂₂): {g2022:.4f}  p={p2022:.4f}")
add(f"\n  >>> Total 1973 pass-through (β + γ₁₉₇₃):  {b+g1973:.4f}")
add(f"  >>> Total 1979 pass-through (β + γ₁₉₇₉):  {b+g1979:.4f}")
add(f"  >>> Total 2022 pass-through (β + γ₂₀₂₂):  {b+g2022:.4f}")

# ── Write output ──────────────────────────────────────────────────────────────
with open(OUTPUT, "w") as f:
    f.write("NEW REGRESSIONS — Core Inflation Pass-Through Analysis\n")
    f.write("Canada · United Kingdom · United States  |  1971-01 to 2025-03\n")
    f.write("="*80 + "\n")
    f.write("\n".join(lines))

print(f"Saved to {OUTPUT}")
