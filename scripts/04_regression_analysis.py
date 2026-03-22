"""
04_regression_analysis.py
==========================
OLS regression models predicting free speech restriction from affective
polarization components, following advisor feedback to move beyond
PCA/UMAP toward explanatory regression analysis.

Research question:
  Do othering, moralizing, and social aversion predict attitudes toward
  campus free speech, controlling for party identification?

Model structure (per advisor guidance):
  DV:  free_speech_restriction_index  (1–7, higher = more pro-restriction)
  IVs: ap_othering, ap_moral, ap_aversion  (1–5, higher = more polarized)
  Control: party_binary  (0 = Democrat, 1 = Republican)
  Optional: interaction terms (polarization × party)

Outputs:
  data/regression_results.csv   — coefficients table for all models
  Console                       — formatted results for thesis write-up

Requires:  data/polarization_clean.csv  (from 00_data_preparation.py)
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/polarization_clean.csv")

# Partisan subsample only (Democrats + Republicans, exclude Independents)
partisan = df[df["party_3cat"].isin(["Democrat", "Republican"])].copy()
partisan["party_binary"] = (partisan["party_3cat"] == "Republican").astype(float)

dems = partisan[partisan["party_binary"] == 0].copy()
reps = partisan[partisan["party_binary"] == 1].copy()

print("=" * 70)
print("REGRESSION ANALYSIS — AFFECTIVE POLARIZATION & FREE SPEECH")
print("=" * 70)
print(f"\nFull sample N = {len(df)}")
print(f"Partisan subsample N = {len(partisan)}")
print(f"  Democrats:   {len(dems)}")
print(f"  Republicans: {len(reps)}")
print(f"\nDV: free_speech_restriction_index  (1–7, higher = more pro-restriction)")
print(f"IVs: ap_othering, ap_moral, ap_aversion  (1–5, higher = more polarized)")
print(f"Control: party_binary  (0 = Democrat, 1 = Republican)")


# ─── 2. Descriptive correlations ───────────────────────────────────────────────
print("\n" + "─" * 70)
print("PEARSON CORRELATIONS (partisan subsample)")
print("─" * 70)
corr_cols = ["ap_moral", "ap_othering", "ap_aversion",
             "free_speech_restriction_index", "party_binary"]
print(partisan[corr_cols].corr(method="pearson").round(3).to_string())


# ─── 3. Model helper ───────────────────────────────────────────────────────────
all_results = []

def run_model(formula, data, label):
    m = smf.ols(formula, data=data).fit()
    print(f"\n{'─'*70}")
    print(f"MODEL: {label}")
    print(f"  Formula: {formula}")
    print(f"  N = {int(m.nobs)}   R² = {m.rsquared:.4f}   "
          f"Adj. R² = {m.rsquared_adj:.4f}   F = {m.fvalue:.3f}   p = {m.f_pvalue:.4f}")
    print(f"\n  {'Variable':<38} {'B':>7} {'SE':>7} {'t':>7} {'p':>8}   95% CI")
    print(f"  {'─'*72}")
    for v in m.params.index:
        b  = m.params[v]
        se = m.bse[v]
        t  = m.tvalues[v]
        p  = m.pvalues[v]
        lo, hi = m.conf_int().loc[v]
        sig = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
        vshort = (v.replace("ap_othering", "Othering")
                   .replace("ap_moral", "Moralizing")
                   .replace("ap_aversion", "Aversion")
                   .replace("party_binary", "Party (Rep=1)")
                   .replace("Intercept", "Intercept"))
        print(f"  {vshort:<38} {b:>7.3f} {se:>7.3f} {t:>7.3f} {p:>8.4f}   "
              f"[{lo:.3f}, {hi:.3f}] {sig}")
        all_results.append({
            "model": label, "variable": v, "B": round(b, 4),
            "SE": round(se, 4), "t": round(t, 4), "p": round(p, 4),
            "CI_low": round(lo, 4), "CI_high": round(hi, 4),
            "N": int(m.nobs), "R2": round(m.rsquared, 4),
            "adj_R2": round(m.rsquared_adj, 4)
        })
    return m


# ─── 4. Part A: Main models ────────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("PART A — MAIN MODELS  (Dems & Reps, all three batteries)")
print("=" * 70)

m1 = run_model(
    "free_speech_restriction_index ~ ap_othering + ap_moral + ap_aversion + party_binary",
    partisan, "M1 — Combined restriction index")

# Faculty and student batteries separately
if "Q95_s" in df.columns:
    faculty_cols = ["Q95_s","Q96_s","Q97_s","Q98_s","Q99_s","Q101_s","Q102_s","Q92_s","Q100_s"]
    student_cols = ["Q103_s","Q104_s","Q105_s","Q106_s"]
    partisan["fs_faculty"] = partisan[[c for c in faculty_cols if c in partisan.columns]].mean(axis=1)
    partisan["fs_student"] = partisan[[c for c in student_cols if c in partisan.columns]].mean(axis=1)

    m2a = run_model(
        "fs_faculty ~ ap_othering + ap_moral + ap_aversion + party_binary",
        partisan, "M2a — Faculty speech restriction")

    m2b = run_model(
        "fs_student ~ ap_othering + ap_moral + ap_aversion + party_binary",
        partisan, "M2b — Student speech restriction")


# ─── 5. Part B: Interaction models ────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("PART B — INTERACTION MODELS  (polarization × party)")
print("Tests whether the polarization → speech link differs by party")
print("=" * 70)

m3 = run_model(
    "free_speech_restriction_index ~ ap_othering + ap_moral + ap_aversion + party_binary"
    " + ap_othering:party_binary + ap_moral:party_binary + ap_aversion:party_binary",
    partisan, "M3 — Combined index + all three interactions")


# ─── 6. Part C: Within-party models ───────────────────────────────────────────
print("\n\n" + "=" * 70)
print("PART C — WITHIN-PARTY MODELS")
print("=" * 70)

m4d = run_model(
    "free_speech_restriction_index ~ ap_othering + ap_moral + ap_aversion",
    dems, "M4-Dem — Democrats only")

m4r = run_model(
    "free_speech_restriction_index ~ ap_othering + ap_moral + ap_aversion",
    reps, "M4-Rep — Republicans only")


# ─── 7. Summary table ──────────────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("SUMMARY TABLE  (B coefficients; * p<.05  ** p<.01  *** p<.001)")
print("=" * 70)
summary_models = [("M1", m1), ("M3 (+Int)", m3), ("M4-Dem", m4d), ("M4-Rep", m4r)]
key_vars = ["Intercept", "ap_othering", "ap_moral", "ap_aversion", "party_binary"]
header = f"  {'Variable':<22}" + "".join(f"  {k:>12}" for k, _ in summary_models)
print(header)
print("  " + "─" * (22 + 14 * len(summary_models)))
for rv in key_vars:
    row = f"  {rv:<22}"
    for _, m in summary_models:
        if rv in m.params.index:
            b = m.params[rv]; p = m.pvalues[rv]
            sig = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else ""
            row += f"  {b:>8.3f}{sig:<4}"
        else:
            row += f"  {'—':>12}"
    print(row)
print("  " + "─" * (22 + 14 * len(summary_models)))
for stat, attr in [("R²", "rsquared"), ("Adj. R²", "rsquared_adj"), ("N", "nobs")]:
    row = f"  {stat:<22}"
    for _, m in summary_models:
        val = getattr(m, attr)
        row += f"  {val:>12.4f}" if stat != "N" else f"  {int(val):>12}"
    print(row)

# ─── 8. Within-party correlations ─────────────────────────────────────────────
print("\n\n" + "─" * 70)
print("WITHIN-PARTY PEARSON CORRELATIONS  (polarization component → restriction)")
print("─" * 70)
print(f"  {'Component':<20}  {'Dems r':>8}  {'Dems p':>8}  {'Reps r':>8}  {'Reps p':>8}")
print(f"  {'─'*56}")
for iv in ["ap_othering", "ap_moral", "ap_aversion"]:
    d_sub = dems[[iv, "free_speech_restriction_index"]].dropna()
    r_sub = reps[[iv, "free_speech_restriction_index"]].dropna()
    rd, pd_ = stats.pearsonr(d_sub[iv], d_sub["free_speech_restriction_index"])
    rr, pr  = stats.pearsonr(r_sub[iv], r_sub["free_speech_restriction_index"])
    print(f"  {iv:<20}  {rd:>8.3f}  {pd_:>8.4f}  {rr:>8.3f}  {pr:>8.4f}")

# ─── 9. Save results ──────────────────────────────────────────────────────────
pd.DataFrame(all_results).to_csv("data/regression_results.csv", index=False)
print("\n\nSaved → data/regression_results.csv")
print("=" * 70)
