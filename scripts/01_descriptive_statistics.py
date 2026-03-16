"""
01_descriptive_statistics.py
============================
Descriptive statistics and reliability analysis for all affective polarization
components. Requires the cleaned data from 00_data_preparation.py.

Outputs:
    visualizations/descriptive_party_distribution.png
    visualizations/descriptive_correlation_heatmap.png
    Printed: means, SDs, Cronbach's alpha for each scale
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ─── LOAD CLEANED DATA ────────────────────────────────────────────────────────
df = pd.read_csv("data/polarization_clean.csv")

PARTY_ORDER  = ["Democrat", "Independent", "Republican"]
PARTY_COLORS = {"Democrat": "#2166ac", "Independent": "#969696", "Republican": "#d6604d"}


# ─── CRONBACH'S ALPHA ─────────────────────────────────────────────────────────

def cronbach_alpha(item_df):
    """Compute Cronbach's alpha from a DataFrame of scale items."""
    item_df = item_df.dropna()
    k = item_df.shape[1]
    var_sum = item_df.var(ddof=1).sum()
    total_var = item_df.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - var_sum / total_var)


print("=" * 60)
print("CRONBACH'S ALPHA — INTERNAL CONSISTENCY")
print("=" * 60)

scales = {
    "Moral Identity (Republican items)": ["moral1R", "moral2R", "moral3R"],
    "Moral Identity (Democrat items)":   ["moral1D", "moral2D", "moral3D"],
    "Othering (Republican items)":       ["other1R", "other2R", "other3R"],
    "Othering (Democrat items)":         ["other1D", "other2D", "other3D"],
    "Aversion (Republican items)":       ["Q135_s", "Q136_s", "Q137_s"],
    "Aversion (Democrat items)":         ["Q138_s", "Q139_s", "Q140_s"],
}

for name, cols in scales.items():
    available = [c for c in cols if c in df.columns]
    if len(available) >= 2:
        alpha = cronbach_alpha(df[available])
        print(f"  {name:45s}  α = {alpha:.3f}")
    else:
        print(f"  {name:45s}  (columns not found)")


# ─── DESCRIPTIVE STATS BY PARTY ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS BY PARTY")
print("=" * 60)

INDEX_VARS = {
    "Moral Identity (own party)":  "ap_moral",
    "Othering (out-party)":        "ap_othering",
    "Social Aversion (out-party)": "ap_aversion",
    "AP Index (composite)":        "affective_polarization_index",
    "Feeling Thermometer Gap":     "FT_gap",
    "Free Speech Restriction":     "free_speech_restriction_index",
    "Distrust Index":              "distrust_index",
}

for label, col in INDEX_VARS.items():
    if col not in df.columns:
        continue
    print(f"\n{label}")
    for party in PARTY_ORDER:
        g = df.loc[df["party_3cat"] == party, col].dropna()
        if len(g) == 0:
            continue
        print(f"  {party:15s}  N={len(g):3d}  M={g.mean():.2f}  SD={g.std():.2f}  "
              f"Mdn={g.median():.2f}  [{g.min():.1f}–{g.max():.1f}]")


# ─── FIGURE 1 OF DESCRIPTIVES: PARTY DISTRIBUTION ────────────────────────────

fig, ax = plt.subplots(figsize=(7, 4))
vc = df["party_3cat"].value_counts()
bars = ax.bar([p for p in PARTY_ORDER if p in vc.index],
              [vc.get(p, 0) for p in PARTY_ORDER if p in vc.index],
              color=[PARTY_COLORS[p] for p in PARTY_ORDER if p in vc.index],
              edgecolor="white", linewidth=0.8)
for bar, p in zip(bars, [p for p in PARTY_ORDER if p in vc.index]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"n={vc.get(p, 0)}", ha="center", va="bottom", fontsize=10)
ax.set_xlabel("Party Identification", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Sample Party Distribution", fontsize=13, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("visualizations/descriptive_party_distribution.png", dpi=300, bbox_inches="tight")
plt.close()
print("\nSaved: visualizations/descriptive_party_distribution.png")


# ─── FIGURE 2 OF DESCRIPTIVES: CORRELATION HEATMAP ───────────────────────────

corr_cols = {k: v for k, v in INDEX_VARS.items() if v in df.columns}
corr_df = df[list(corr_cols.values())].rename(columns={v: k for k, v in corr_cols.items()})
corr_matrix = corr_df.corr(method="pearson")

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5,
            cbar_kws={"shrink": 0.8})
ax.set_title("Pearson Correlations Among Polarization Indices", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("visualizations/descriptive_correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: visualizations/descriptive_correlation_heatmap.png")
