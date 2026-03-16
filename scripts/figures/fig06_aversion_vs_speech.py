"""
fig06_aversion_vs_speech.py
============================
Figure 6: Scatter Plot — Social Aversion vs. Free Speech Restriction by Party

X-axis: Social Aversion index (1–5, higher = more avoidance of out-partisans)
Y-axis: Free Speech Restriction index (1–7, higher = more pro-restriction)
        NOTE: we flip the free_speech_support_index (high = support) to
        restriction direction (high = restrict) so the axis reads intuitively.

Per-party regression lines with 95% confidence bands.
A flat/near-zero slope within each party illustrates that aversion and speech
restriction are largely orthogonal within-party dimensions.

Output:
    visualizations/figure_06_aversion_vs_speech_restriction.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

df = pd.read_csv("data/polarization_clean.csv")

PARTY_ORDER  = ["Democrat", "Independent", "Republican"]
PARTY_COLORS = {"Democrat": "#2166ac", "Independent": "#969696", "Republican": "#d6604d"}

# Convert free speech SUPPORT index to RESTRICTION index (flip so high = restrict)
df["free_speech_restriction_index"] = 8 - df["free_speech_support_index"]

fig, ax = plt.subplots(figsize=(8, 5.5))

np.random.seed(42)

for party in PARTY_ORDER:
    sub = df[df["party_3cat"] == party][
        ["ap_aversion", "free_speech_restriction_index"]
    ].dropna()
    if len(sub) < 5:
        continue

    x_vals = sub["ap_aversion"].values
    y_vals = sub["free_speech_restriction_index"].values
    color  = PARTY_COLORS[party]

    # Scatter
    ax.scatter(x_vals, y_vals, alpha=0.30, s=15, color=color, linewidths=0)

    # Regression line + 95% CI band
    slope, intercept, r, p, se = stats.linregress(x_vals, y_vals)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
    y_line = slope * x_line + intercept

    # CI band via bootstrap
    n = len(x_vals)
    boot_preds = []
    for _ in range(500):
        idx = np.random.choice(n, n, replace=True)
        s2, i2, *_ = stats.linregress(x_vals[idx], y_vals[idx])
        boot_preds.append(s2 * x_line + i2)
    boot_preds = np.array(boot_preds)
    lo = np.percentile(boot_preds, 2.5, axis=0)
    hi = np.percentile(boot_preds, 97.5, axis=0)

    ax.plot(x_line, y_line, color=color, linewidth=2, alpha=0.95,
            label=f"{party}  r={r:.2f}, β={slope:.2f}, p={'<.001' if p<.001 else f'{p:.3f}'}")
    ax.fill_between(x_line, lo, hi, color=color, alpha=0.12)

# Reference line: no association
ax.axhline(df["free_speech_restriction_index"].mean(), color="grey",
           linewidth=0.8, linestyle="--", alpha=0.4, label="Overall mean (restriction)")

ax.set_xlabel("Social Aversion Index (1–5)", fontsize=11)
ax.set_ylabel("Free Speech Restriction Score (1–7)", fontsize=11)
ax.set_title(
    "Figure 6: Social Aversion vs. Free Speech Restriction by Party\n"
    "Shaded bands = 95% CI; flat slopes indicate within-party orthogonality",
    fontsize=12, fontweight="bold",
)
ax.legend(fontsize=9, framealpha=0.85, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("visualizations/figure_06_aversion_vs_speech_restriction.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("Saved: visualizations/figure_06_aversion_vs_speech_restriction.png")

# Print correlation table
print("\nPearson r (Aversion × Speech Restriction) by party:")
for party in PARTY_ORDER:
    sub = df[df["party_3cat"] == party][
        ["ap_aversion", "free_speech_restriction_index"]
    ].dropna()
    if len(sub) < 5:
        continue
    r, p = stats.pearsonr(sub["ap_aversion"], sub["free_speech_restriction_index"])
    print(f"  {party:<14}  r={r:+.3f}  p={p:.4f}  N={len(sub)}")
