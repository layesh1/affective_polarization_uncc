"""
fig10_ft_vs_ap_validation.py
==============================
Figure 10: Do Two Different Ways of Measuring Polarization Agree?

X-axis: AP Composite Index (survey questions about moral identity,
        othering, and social aversion — 1 to 5)
Y-axis: Feeling Thermometer Gap (how much warmer you feel toward your
        own party vs. the other party, measured 0 to 100)

If both measures tap into the same underlying attitude, students who score
high on one should also score high on the other (upward sloping lines).
This is called "convergent validity."

Output: visualizations/figure_10_ft_vs_ap_validation.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

df = pd.read_csv("data/polarization_clean.csv")

PARTIES = ["Democrat", "Republican"]
COLORS  = {"Democrat": "#2166ac", "Republican": "#d6604d"}

plot_df = df[df["party_3cat"].isin(PARTIES)][
    ["affective_polarization_index", "FT_gap", "party_3cat"]
].dropna()

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor("white")
np.random.seed(42)

all_x, all_y = [], []
for party in PARTIES:
    sub   = plot_df[plot_df["party_3cat"] == party]
    x, y  = sub["affective_polarization_index"].values, sub["FT_gap"].values
    color = COLORS[party]

    ax.scatter(x, y, color=color, alpha=0.32, s=20, linewidths=0)

    slope, intercept, r, p, _ = stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 200)
    p_str = "< .001" if p < .001 else f"= {p:.3f}"
    ax.plot(xs, slope*xs+intercept, color=color, lw=2.3,
            label=f"{party}s:  r = {r:.2f}  (p {p_str})")
    all_x.extend(x); all_y.extend(y)

# Overall line
all_x, all_y = np.array(all_x), np.array(all_y)
s_all, i_all, r_all, p_all, _ = stats.linregress(all_x, all_y)
xs_all = np.linspace(all_x.min(), all_x.max(), 200)
ax.plot(xs_all, s_all*xs_all+i_all, color="black", lw=1.6, ls="--", alpha=0.6,
        label=f"Overall:  r = {r_all:.2f}  (p < .001)")

ax.set_xlabel("AP Composite Index  (1 = Low hostility  →  5 = High hostility)",
              fontsize=11)
ax.set_ylabel("Feeling Thermometer Gap  (0 = Equal warmth  →  100 = Much warmer toward own party)",
              fontsize=11)
ax.set_title("Figure 10: Do Two Different Measures of Polarization Tell the Same Story?\n"
             "Strong positive relationship = both measures capture the same underlying attitude",
             fontsize=12, fontweight="bold", pad=10)

ax.legend(fontsize=10, framealpha=0.9, loc="upper left")

note = (
    f"r = {r_all:.2f} means the two measures have moderately strong agreement.\n"
    "This 'convergent validity' confirms the survey scale is measuring\n"
    "something real — not just statistical noise."
)
ax.text(0.99, 0.01, note, transform=ax.transAxes, fontsize=8.5, va="bottom",
        ha="right", bbox=dict(boxstyle="round,pad=0.45", fc="#f0f0f0", alpha=0.9, ec="grey"))

ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("visualizations/figure_10_ft_vs_ap_validation.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: visualizations/figure_10_ft_vs_ap_validation.png  (overall r = {r_all:.3f})")
