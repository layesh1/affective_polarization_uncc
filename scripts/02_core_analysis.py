"""
02_core_analysis.py
===================
Core inferential analysis: independent-samples t-tests, Cohen's d effect sizes,
and Pearson/Spearman correlations for all affective polarization components.
Requires cleaned data from 00_data_preparation.py.

Outputs:
    Printed statistical summary
    data/statistical_summary.csv
"""

import pandas as pd
import numpy as np
from scipy import stats

# ─── LOAD ─────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/polarization_clean.csv")

dems  = df[df["party_3cat"] == "Democrat"]
reps  = df[df["party_3cat"] == "Republican"]
indep = df[df["party_3cat"] == "Independent"]


# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def cohens_d(g1, g2):
    """Pooled-SD Cohen's d (positive = g1 > g2)."""
    n1, n2 = len(g1), len(g2)
    var1, var2 = g1.var(ddof=1), g2.var(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (g1.mean() - g2.mean()) / pooled_sd

def compare_parties(col, label):
    """Run independent t-test + Cohen's d for Democrats vs Republicans."""
    d_vals = dems[col].dropna()
    r_vals = reps[col].dropna()
    t, p   = stats.ttest_ind(d_vals, r_vals)
    d      = cohens_d(d_vals, r_vals)
    sig    = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
    return {
        "Measure": label,
        "Dem_M": round(d_vals.mean(), 3),
        "Dem_SD": round(d_vals.std(), 3),
        "Dem_N": len(d_vals),
        "Rep_M": round(r_vals.mean(), 3),
        "Rep_SD": round(r_vals.std(), 3),
        "Rep_N": len(r_vals),
        "t": round(t, 3),
        "p": round(p, 4),
        "sig": sig,
        "d": round(d, 3),
    }


# ─── MAIN COMPARISONS ─────────────────────────────────────────────────────────

COMPARISONS = [
    ("ap_moral",                  "Moral Identity"),
    ("ap_othering",               "Othering"),
    ("ap_aversion",               "Social Aversion"),
    ("affective_polarization_index", "AP Composite Index"),
    ("FT_gap",                    "Feeling Thermometer Gap"),
    ("free_speech_support_index", "Free Speech Support"),
    ("distrust_index",            "Out-Party Distrust"),
]

results = []
print("=" * 90)
print(f"{'Measure':<35} {'Dem M':>7} {'Rep M':>7} {'t':>8} {'p':>8} {'sig':>5} {'d':>7}")
print("=" * 90)

for col, label in COMPARISONS:
    if col not in df.columns:
        continue
    r = compare_parties(col, label)
    results.append(r)
    print(f"{label:<35} {r['Dem_M']:>7.3f} {r['Rep_M']:>7.3f} "
          f"{r['t']:>8.3f} {r['p']:>8.4f} {r['sig']:>5} {r['d']:>7.3f}")

print("=" * 90)
print("Note: d > 0 means Democrats score higher than Republicans.\n"
      "Effect size benchmarks: |d| < 0.2 small, 0.5 medium, 0.8 large.")

pd.DataFrame(results).to_csv("data/statistical_summary.csv", index=False)
print("\nSaved: data/statistical_summary.csv")


# ─── PARTISAN SORTING: ideology × party ───────────────────────────────────────

if "ideology" in df.columns and "party_num" in df.columns:
    valid = df[["ideology", "party_num"]].dropna()
    rho, p_rho = stats.spearmanr(valid["party_num"], valid["ideology"])
    print(f"\nPartisan sorting — Spearman r(party, ideology) = {rho:.3f}, p = {p_rho:.4f}")


# ─── INTER-COMPONENT CORRELATIONS ─────────────────────────────────────────────

print("\nInter-component correlations (Pearson r):")
comp_cols = ["ap_moral", "ap_othering", "ap_aversion"]
comp_names = ["Moral", "Othering", "Aversion"]
available = [(n, c) for n, c in zip(comp_names, comp_cols) if c in df.columns]
if len(available) >= 2:
    corr_df = df[[c for _, c in available]].dropna()
    corr_df.columns = [n for n, _ in available]
    corr_mat = corr_df.corr()
    print(corr_mat.round(3))
