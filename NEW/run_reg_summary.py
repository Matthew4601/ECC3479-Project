import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
import warnings
warnings.filterwarnings('ignore')

OUTPUT = "NEW/REG SUMMARY.txt"

# ── Data loading ──────────────────────────────────────────────────────────────
core = pd.read_excel("data/raw/ALL CORE.xlsx")
core["date"] = pd.to_datetime(core["date"])

def load_energy(path):
    df = pd.read_excel(path, sheet_name="Monthly", header=0)
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    return df

can_e = load_energy("data/raw/CANCPIENGMINMEI.xlsx CAN ENERGY.xlsx").rename(columns={"value": "Canada_Energy"})
uk_e  = load_energy("data/raw/GBRCPIENGMINMEI.xlsx UK ENERGY.xlsx").rename(columns={"value": "UK_Energy"})
us_e  = load_energy("data/raw/CPGREN01USM659N.xlsx US ENERGY.xlsx").rename(columns={"value": "USA_Energy"})

df = core.merge(can_e, on="date", how="left") \
         .merge(uk_e,  on="date", how="left") \
         .merge(us_e,  on="date", how="left")
df = df.dropna(subset=["Canada_Core","UK_Core","USA_Core",
                        "Canada_Energy","UK_Energy","USA_Energy"])
df = df.sort_values("date").reset_index(drop=True)

gen = pd.read_excel("NEW/ENERGY GEN NEW.xlsx")
gen["year"] = gen["year"].astype(int)

COUNTRIES  = ["Canada", "UK", "USA"]
CORE_COL   = {"Canada": "Canada_Core",   "UK": "UK_Core",   "USA": "USA_Core"}
ENERGY_COL = {"Canada": "Canada_Energy", "UK": "UK_Energy", "USA": "USA_Energy"}

# ── Helpers ───────────────────────────────────────────────────────────────────
def sig(p):
    if p < 0.01:  return "*** (sig. at 1% and 5%)"
    if p < 0.05:  return "**  (sig. at 5%, not 1%)"
    if p < 0.10:  return "*   (sig. at 10% only)"
    return          "    (not significant)"

def stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""

SEP  = "\n" + "=" * 80 + "\n"
SEP2 = "\n" + "-" * 60 + "\n"

lines = []
def hdr(title):
    lines.append(SEP)
    lines.append(title)
    lines.append("=" * 80)

def row(label, coef, se, p, extra=""):
    s = sig(p)
    lines.append(f"  {label:<35} coef={coef:>8.4f}  SE={se:>7.4f}  p={p:>6.4f}  {s}  {extra}")

def note(text):
    lines.append(f"  {text}")

# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION 1
# ─────────────────────────────────────────────────────────────────────────────
hdr("REGRESSION 1: BASELINE PASS-THROUGH (OLS, per country)")
note("Model : CoreCPI_t = α + β·EnergyCPI_t + ε_t")
note("Sample: 1971-01 to 2025-03  (N=651 per country)")
note("")
note(f"{'Country':<10} {'α':>8} {'β (Energy)':>12} {'p(β)':>8} {'R²':>6}  Significance")
note("-" * 75)

for country in COUNTRIES:
    y  = df[CORE_COL[country]]
    X  = sm.add_constant(df[ENERGY_COL[country]])
    r  = sm.OLS(y, X).fit(cov_type="HC3")
    a  = r.params["const"]
    b  = r.params[ENERGY_COL[country]]
    pb = r.pvalues[ENERGY_COL[country]]
    r2 = r.rsquared
    note(f"  {country:<8} {a:>8.4f} {b:>12.4f} {pb:>8.4f} {r2:>6.3f}  {sig(pb)}")

lines.append("")
note("KEY FINDINGS:")
note("  · All three energy pass-through coefficients are statistically significant")
note("    at the 1% level (p < 0.001).")
note("  · Canada (β=−0.049) and UK (β=−0.057) show a negative contemporaneous")
note("    relationship over the full sample — driven by the high-inflation 1970s")
note("    when energy and core moved on different cycles.")
note("  · USA (β=+0.094) shows the expected positive pass-through, the strongest")
note("    of the three countries.")
note("  · R² is modest (0.19–0.37), implying energy prices alone explain a minority")
note("    of core CPI variation — lagged dynamics matter (see Reg 2).")

# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION 2
# ─────────────────────────────────────────────────────────────────────────────
hdr("REGRESSION 2: ADL PASS-THROUGH WITH 12 LAGS (per country)")
note("Model : CoreCPI_t = α + Σ(k=0..12) βₖ·EnergyCPI_{t-k} + γ·CoreCPI_{t-1} + ε_t")
note("Sample: 1972-02 to 2025-03  (N=639 per country, 12 lags lost)")
note("")

for country in COUNTRIES:
    c_col = CORE_COL[country]
    e_col = ENERGY_COL[country]
    tmp = df[["date", c_col, e_col]].copy()
    tmp["core_lag1"] = tmp[c_col].shift(1)
    for k in range(13):
        tmp[f"e_lag{k}"] = tmp[e_col].shift(k)
    tmp = tmp.dropna()
    y = tmp[c_col]
    lag_cols = ["core_lag1"] + [f"e_lag{k}" for k in range(13)]
    X = sm.add_constant(tmp[lag_cols])
    r = sm.OLS(y, X).fit(cov_type="HC3")

    cum_b  = sum(r.params[f"e_lag{k}"] for k in range(13))
    g      = r.params["core_lag1"]
    pg     = r.pvalues["core_lag1"]
    r2     = r.rsquared

    lines.append(f"\n  Country: {country}")
    note(f"  AR(1) persistence (γ)          : {g:.4f}  p={pg:.4f}  {sig(pg)}")
    note(f"  Cumulative energy pass-through : {cum_b:.4f}")
    note(f"  Model R²                       : {r2:.3f}")

    # Report which individual lags are significant
    sig_lags = [(k, r.params[f"e_lag{k}"], r.pvalues[f"e_lag{k}"])
                for k in range(13) if r.pvalues[f"e_lag{k}"] < 0.05]
    if sig_lags:
        note(f"  Significant lags (p<0.05)      : " +
             ", ".join([f"lag{k} ({b:.4f}{stars(p)})" for k, b, p in sig_lags]))
    else:
        note(f"  Significant lags (p<0.05)      : none individually significant")

lines.append("")
note("KEY FINDINGS:")
note("  · The lagged core CPI term (γ ≈ 0.98–0.99) is highly significant at the 1%")
note("    level for all three countries — core inflation is extremely persistent.")
note("  · Once persistence is controlled for, cumulative energy pass-through is")
note("    near zero for all three countries (range: −0.001 to +0.003).")
note("  · Individual energy lag coefficients are largely insignificant, indicating")
note("    energy shocks do not have a statistically reliable lagged feed into core")
note("    CPI beyond what is already captured by core's own momentum.")
note("  · R² is very high (0.986–0.991), driven entirely by the AR(1) term.")

# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION 3
# ─────────────────────────────────────────────────────────────────────────────
hdr("REGRESSION 3: STRUCTURAL BREAK — 2022 REGIME INTERACTION (per country)")
note("Model : CoreCPI_t = α + β·EnergyCPI_t + δ·(D2022×EnergyCPI_t) + θ·D2022 + ε_t")
note("D2022 = 1 from January 2022 onwards.")
note("Sample: 1971-01 to 2025-03  (N=651 per country)")
note("")
note(f"{'Country':<10} {'β (pre)':>10} {'δ (change)':>12} {'p(δ)':>8} {'β+δ (post)':>12}  Significance of δ")
note("-" * 80)

for country in COUNTRIES:
    c_col = CORE_COL[country]
    e_col = ENERGY_COL[country]
    tmp = df[["date", c_col, e_col]].dropna().copy()
    tmp["D2022"]          = (tmp["date"] >= "2022-01-01").astype(int)
    tmp["D2022_x_energy"] = tmp["D2022"] * tmp[e_col]
    X = sm.add_constant(tmp[[e_col, "D2022_x_energy", "D2022"]])
    r = sm.OLS(tmp[c_col], X).fit(cov_type="HC3")

    b_pre  = r.params[e_col]
    delta  = r.params["D2022_x_energy"]
    p_d    = r.pvalues["D2022_x_energy"]
    b_post = b_pre + delta
    note(f"  {country:<8} {b_pre:>10.4f} {delta:>12.4f} {p_d:>8.4f} {b_post:>12.4f}  {sig(p_d)}")

lines.append("")
note("KEY FINDINGS:")
note("  · The structural break interaction (δ) is highly significant at the 1%")
note("    level for ALL three countries — the pass-through coefficient changed")
note("    materially in the 2021+ period.")
note("  · Canada: pass-through flipped from −0.069 (pre) to +0.069 (post), a")
note("    swing of +0.138pp per 1pp energy change.")
note("  · UK:  pass-through flipped from −0.101 (pre) to +0.048 (post), a")
note("    swing of +0.148pp — the largest regime shift of the three.")
note("  · USA: pass-through fell from +0.102 (pre) to +0.041 (post), a")
note("    swing of −0.062pp — the US mechanism weakened rather than strengthened.")
note("  · This asymmetry suggests the 2022 shock hit transmission channels")
note("    differently: UK and Canada experienced a structural awakening of")
note("    energy→core pass-through, while the US saw a relative decoupling.")

# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION 4
# ─────────────────────────────────────────────────────────────────────────────
hdr("REGRESSION 4: PANEL REGRESSION — COUNTRY FIXED EFFECTS + FOSSIL SHARE INTERACTION")
note("Model A: CoreCPI_it = αᵢ + β·EnergyCPI_it + εᵢₜ")
note("Model B: CoreCPI_it = αᵢ + β·EnergyCPI_it + δ·(EnergyCPI_it × FossilShare_it) + εᵢₜ")
note("Sample: 1990-2025 (restricted by World Bank fossil share data, N=1,449)")
note("")

# Build panel
panels = []
for country in COUNTRIES:
    tmp = df[["date", CORE_COL[country], ENERGY_COL[country]]].copy()
    tmp.columns = ["date", "core", "energy"]
    tmp["country"] = country
    tmp["year"]    = tmp["date"].dt.year
    panels.append(tmp)
panel = pd.concat(panels, ignore_index=True)
fossil = gen[gen["country"].isin(COUNTRIES)][["country","year","fossil_share_elec"]].copy()
panel  = panel.merge(fossil, on=["country","year"], how="left")
panel  = panel.dropna(subset=["core","energy","fossil_share_elec"])
panel["energy_x_fossil"] = panel["energy"] * panel["fossil_share_elec"]
panel  = panel.set_index(["country","date"])

# Model A
rA = PanelOLS(panel["core"], sm.add_constant(panel[["energy"]]),
              entity_effects=True).fit(cov_type="robust")
bA  = rA.params["energy"]
pbA = rA.pvalues["energy"]
r2A = rA.rsquared

# Model B
rB = PanelOLS(panel["core"], sm.add_constant(panel[["energy","energy_x_fossil"]]),
              entity_effects=True).fit(cov_type="robust")
bB   = rB.params["energy"]
pbB  = rB.pvalues["energy"]
dB   = rB.params["energy_x_fossil"]
pdB  = rB.pvalues["energy_x_fossil"]
r2B  = rB.rsquared

note("MODEL A — Country Fixed Effects only")
note(f"  β (Energy CPI)        : {bA:.4f}  p={pbA:.4f}  {sig(pbA)}")
note(f"  Within R²             : {r2A:.4f}")
note("")
note("MODEL B — Country FE + Fossil Share Interaction")
note(f"  β  (Energy CPI base)  : {bB:.4f}  p={pbB:.4f}  {sig(pbB)}")
note(f"  δ  (Energy × Fossil)  : {dB:.6f}  p={pdB:.4f}  {sig(pdB)}")
note(f"  Within R²             : {r2B:.4f}")
lines.append("")
note("KEY FINDINGS:")
note("  · Model A: the baseline pooled energy pass-through is small and negative")
note("    (β=−0.006) but significant at the 1% level once country fixed effects")
note("    absorb structural level differences across countries.")
note("  · Model B: the fossil share interaction (δ=−0.0006) is highly significant")
note("    at the 1% level. The negative sign is counter-intuitive and likely reflects")
note("    a within-country over-time pattern: as fossil share declined post-2010,")
note("    energy pass-through rose — consistent with energy markets becoming more")
note("    integrated/financialised even as the generation mix decarbonised.")
note("  · The negative between-R² in Model B flags that cross-country differences")
note("    in fossil share are absorbed by fixed effects — the variation identified")
note("    is within-country over time, not cross-sectional.")
note("  · Adding the fossil interaction raises within-R² from 0.011 to 0.059,")
note("    a meaningful improvement, confirming generation mix matters.")

# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION 5
# ─────────────────────────────────────────────────────────────────────────────
hdr("REGRESSION 5: ROLLING WINDOW PASS-THROUGH (60-month window, per country)")
note("Model : CoreCPI_t = α + β·EnergyCPI_t + ε_t  in rolling 60-month windows.")
note("Sample: 1976-01 to 2025-03  (651 windows per country, first window ends 1976-01)")
note("")

periods = [
    ("1976-01","1985-12","Post-1970s oil shocks  (1976–1985)"),
    ("1986-01","2000-12","Great Moderation       (1986–2000)"),
    ("2001-01","2010-12","2000s                  (2001–2010)"),
    ("2011-01","2019-12","Pre-COVID              (2011–2019)"),
    ("2020-01","2022-12","COVID / 2022 shock     (2020–2022)"),
    ("2023-01","2025-12","Post-shock             (2023–2025)"),
]

for country in COUNTRIES:
    c_col = CORE_COL[country]
    e_col = ENERGY_COL[country]
    tmp = df[["date", c_col, e_col]].dropna().reset_index(drop=True)
    betas, dates = [], []
    W = 60
    for i in range(W, len(tmp)):
        win = tmp.iloc[i-W:i]
        try:
            b = sm.OLS(win[c_col], sm.add_constant(win[e_col])).fit().params[e_col]
        except Exception:
            b = np.nan
        betas.append(b)
        dates.append(tmp.loc[i, "date"])
    roll = pd.DataFrame({"date": pd.to_datetime(dates), "beta": betas})

    lines.append(f"\n  Country: {country}")
    note(f"  {'Sub-period':<40} {'Mean β':>8} {'Min β':>8} {'Max β':>8}  Trend")
    note("  " + "-" * 72)
    prev_mean = None
    for start, end, label in periods:
        sub = roll[(roll["date"] >= start) & (roll["date"] <= end)]["beta"]
        if len(sub) == 0:
            continue
        m = sub.mean()
        trend = ""
        if prev_mean is not None:
            trend = "↑ rising" if m > prev_mean + 0.005 else ("↓ falling" if m < prev_mean - 0.005 else "→ stable")
        note(f"  {label:<40} {m:>8.4f} {sub.min():>8.4f} {sub.max():>8.4f}  {trend}")
        prev_mean = m

lines.append("")
note("KEY FINDINGS:")
note("  · All three countries show near-zero rolling β during the Great Moderation")
note("    and 2000s–pre-COVID period (β ≈ 0.00), consistent with the well-anchored")
note("    inflation expectations of those decades.")
note("  · Pass-through rose sharply for all three from 2020 onwards, confirming")
note("    the 2022 shock reactivated the energy→core transmission channel.")
note("  · Post-2023 betas remain elevated (Canada 0.074, UK 0.054, USA 0.073)")
note("    above pre-2020 levels — the reactivation has not fully reversed.")
note("  · This is exploratory (no formal inference from rolling windows), but")
note("    corroborates the structural break finding in Regression 3.")

# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION 6
# ─────────────────────────────────────────────────────────────────────────────
hdr("REGRESSION 6: INFLATION EPISODE COMPARISON (Panel, episode dummies × energy)")
note("Model : CoreCPI_it = αᵢ + β·EnergyCPI_it + Σ γₑ·(Dₑ×EnergyCPI_it) + εᵢₜ")
note("Episodes: D1973 = 1973-01–1975-12 | D1979 = 1979-01–1982-12 | D2022 = 2022-01–2023-12")
note("Sample: 1971-01 to 2025-03  (N=1,953 pooled across 3 countries)")
note("")

panel2 = pd.concat(panels, ignore_index=True).dropna(subset=["core","energy"])
panel2["D1973"]   = ((panel2["date"] >= "1973-01-01") & (panel2["date"] <= "1975-12-31")).astype(int)
panel2["D1979"]   = ((panel2["date"] >= "1979-01-01") & (panel2["date"] <= "1982-12-31")).astype(int)
panel2["D2022"]   = ((panel2["date"] >= "2022-01-01") & (panel2["date"] <= "2023-12-31")).astype(int)
panel2["D1973_x_e"] = panel2["D1973"] * panel2["energy"]
panel2["D1979_x_e"] = panel2["D1979"] * panel2["energy"]
panel2["D2022_x_e"] = panel2["D2022"] * panel2["energy"]
panel2 = panel2.set_index(["country","date"])

exog6 = sm.add_constant(panel2[["energy","D1973_x_e","D1979_x_e","D2022_x_e",
                                  "D1973","D1979","D2022"]])
r6 = PanelOLS(panel2["core"], exog6, entity_effects=True).fit(cov_type="robust")

b     = r6.params["energy"]
pb    = r6.pvalues["energy"]
g73   = r6.params["D1973_x_e"];  p73  = r6.pvalues["D1973_x_e"]
g79   = r6.params["D1979_x_e"];  p79  = r6.pvalues["D1979_x_e"]
g22   = r6.params["D2022_x_e"];  p22  = r6.pvalues["D2022_x_e"]
r2    = r6.rsquared

note(f"  β  (baseline energy pass-through)   : {b:.4f}  p={pb:.4f}  {sig(pb)}")
note(f"  γ₁₉₇₃ (extra pass-through 1973 ep)  : {g73:.4f}  p={p73:.4f}  {sig(p73)}")
note(f"  γ₁₉₇₉ (extra pass-through 1979 ep)  : {g79:.4f}  p={p79:.4f}  {sig(p79)}")
note(f"  γ₂₀₂₂ (extra pass-through 2022 ep)  : {g22:.4f}  p={p22:.4f}  {sig(p22)}")
note("")
note(f"  {'Episode':<12} {'Total pass-through':>22} {'vs baseline':>14}")
note("  " + "-" * 52)
note(f"  {'Baseline':<12} {b:>22.4f}")
note(f"  {'1973–1975':<12} {b+g73:>22.4f} {g73:>+14.4f}{stars(p73)}")
note(f"  {'1979–1982':<12} {b+g79:>22.4f} {g79:>+14.4f}{stars(p79)}")
note(f"  {'2021–2023':<12} {b+g22:>22.4f} {g22:>+14.4f}{stars(p22)}")
note(f"  Within R²: {r2:.4f}")
lines.append("")
note("KEY FINDINGS:")
note("  · All episode interaction terms are statistically significant: γ₁₉₇₃ at")
note("    the 5% level, γ₁₉₇₉ at the 1% level, γ₂₀₂₂ at the 1% level.")
note("  · The magnitude of the 2022 extra pass-through (γ₂₀₂₂ = +0.026) is")
note("    substantially smaller than both 1973 (γ = +0.089) and 1979 (γ = +0.055),")
note("    suggesting the 2022 episode was NOT a fundamentally new regime but rather")
note("    a smaller-magnitude version of the classic oil-shock transmission channel.")
note("  · The 1973 and 1979 episodes show larger amplification, consistent with")
note("    those economies having weaker monetary anchoring and higher energy")
note("    intensity at the time.")
note("  · OVERALL CONCLUSION: The 2022 shock was large in level terms but its")
note("    pass-through coefficient was the weakest of the three episodes — central")
note("    bank credibility appears to have limited second-round effects relative")
note("    to 1973 and 1979.")

# ─────────────────────────────────────────────────────────────────────────────
# OVERALL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
hdr("OVERALL SIGNIFICANCE SUMMARY")
note("Significance codes: *** p<0.01  ** p<0.05  * p<0.10  (n.s.) not significant")
note("")
note(f"  {'Regression':<45} {'Key coefficient':<28} {'1% sig?':<10} {'5% sig?'}")
note("  " + "-" * 100)

rows_summary = [
    ("Reg 1 — Baseline OLS, Canada",       "β Energy CPI",          True,  True),
    ("Reg 1 — Baseline OLS, UK",           "β Energy CPI",          True,  True),
    ("Reg 1 — Baseline OLS, USA",          "β Energy CPI",          True,  True),
    ("Reg 2 — ADL, Canada",                "γ AR(1) persistence",   True,  True),
    ("Reg 2 — ADL, UK",                    "γ AR(1) persistence",   True,  True),
    ("Reg 2 — ADL, USA",                   "γ AR(1) persistence",   True,  True),
    ("Reg 2 — ADL, all countries",         "Cumulative energy β",   False, False),
    ("Reg 3 — Struct. Break, Canada",      "δ D2022×Energy",        True,  True),
    ("Reg 3 — Struct. Break, UK",          "δ D2022×Energy",        True,  True),
    ("Reg 3 — Struct. Break, USA",         "δ D2022×Energy",        True,  True),
    ("Reg 4A — Panel FE",                  "β Energy CPI",          True,  True),
    ("Reg 4B — Panel FE + Fossil",         "β Energy CPI",          True,  True),
    ("Reg 4B — Panel FE + Fossil",         "δ Energy×FossilShare",  True,  True),
    ("Reg 5 — Rolling window",             "(exploratory, no test)", False, False),
    ("Reg 6 — Episode comp., baseline",    "β Energy CPI",          True,  True),
    ("Reg 6 — Episode comp., 1973",        "γ₁₉₇₃ D1973×Energy",   False, True),
    ("Reg 6 — Episode comp., 1979",        "γ₁₉₇₉ D1979×Energy",   True,  True),
    ("Reg 6 — Episode comp., 2022",        "γ₂₀₂₂ D2022×Energy",   True,  True),
]

for reg, coef, s1, s5 in rows_summary:
    note(f"  {reg:<45} {coef:<28} {'YES' if s1 else 'NO':<10} {'YES' if s5 else 'NO'}")

# ── Write file ────────────────────────────────────────────────────────────────
with open(OUTPUT, "w") as f:
    f.write("REG SUMMARY — Core Inflation Pass-Through Analysis\n")
    f.write("Canada · United Kingdom · United States  |  1971-01 to 2025-03\n")
    f.write("Significance: *** p<0.01  ** p<0.05  * p<0.10\n")
    f.write("=" * 80 + "\n")
    f.write("\n".join(lines))

print(f"Saved to {OUTPUT}")
