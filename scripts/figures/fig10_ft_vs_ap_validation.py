"""
fig10_ft_vs_ap_validation.py
==============================
Figure 10: Feeling Thermometer Gap vs. AP Composite Index (Convergent Validity)

Scatter plot with FT gap (classic measure) on Y and AP composite index
(survey-based measure) on X.  A strong positive correlation validates that
both measures are tapping the same underlying construct.

Colored by party; separate regression lines per party.

Output:
    visualizations/figure_10_ft_vs_ap_validation.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

df = pd.read_csv("data/polarization_clean.csv")

PARTY_ORDER  = ["Democrat", "Republican"]
PARTY_COLORS = {"Democrat": "#2166ac", "Republican": "#d6604d"}

# Only partisans have both measures
plot_df = df[df["party_3cat"].isin(PARTY_ORDER)][
    ["ap_moral", "ap_othering", "ap_aversion",
     "affective_polarization_index", "FT_gap", "party_3cat"]
].dropna()

fig, ax = plt.subplots(figsize=(8, 5.5))

np.random.seed(42)
all_x, all_y = [], []

for party in PARTY_ORDER:
    sub   = plot_df[plot_df["party_3cat"] == party]
    x     = sub["affective_polarization_index"].values
    y     = sub["FT_gap"].values
    color = PARTY_COLORS[party]

    ax.scatter(x, y, color=color, alpha=0.35, s=18, linewidths=0)

    slope, intercept, r, p, _ = stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 200)
    ax.plot(xs, slope * xs + intercept, color=color, linewidth=2.2,
            label=f"{party}:  r = {r:.2f}, p {'< .001' if p<.001 else f'= {p:.3f}'}")
    all_x.extend(x); all_y.extend(y)

# Overall regression
all_x, all_y = np.array(all_x), np.array(all_y)
slope_all, int_all, r_all, p_all, _ = stats.linregress(all_x, all_y)
xs_all = np.linspace(all_x.min(), all_x.max(), 200)
ax.plot(xs_all, slope_all * xs_all + int_all, color="black", linewidth=1.5,
        linestyle="--", alpha=0.6,
        label=f"Overall:  r = {r_all:.2f}, p {'< .001' if p_all<.001 else f'= {p_all:.3f}'}")

ax.set_xlabel("AP Composite Index (1–5, survey scale)", fontsize=11)
ax.set_ylabel("Feeling Thermometer Gap (in-party − out-party, 0–100)", fontsize=11)
ax.set_title("Figure 10: Convergent Validity — AP Index vs. Feeling Thermometer Gap\n"
             "Both measures should track together if tapping the same construct",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9.5, framealpha=0.85, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("visualizations/figure_10_ft_vs_ap_validation.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: visualizations/figure_10_ft_vs_ap_validation.png")
print(f"Overall r = {r_all:.3f}, p = {p_all:.4f}  (convergent validity)")
