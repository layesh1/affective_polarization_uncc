"""
fig06_aversion_vs_speech.py
============================
Figure 6: Are Students Who Avoid the Other Party Also More Pro-Restriction on Speech?

X-axis: Social Aversion score (how much a student avoids out-party members, 1–5)
Y-axis: Speech Restriction score (how pro-restriction on campus speech, 1–7)
Color:  Party (blue=Democrat, grey=Independent, red=Republican)

If aversion and speech restriction were the SAME thing, we'd see steep upward
lines. Near-flat lines show they are largely SEPARATE dimensions.

Output: visualizations/figure_06_aversion_vs_speech_restriction.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

df = pd.read_csv("data/polarization_clean.csv")

PARTIES = ["Democrat", "Independent", "Republican"]
COLORS  = {"Democrat": "#2166ac", "Independent": "#888888", "Republican": "#d6604d"}

# free_speech_restriction_index: high = more pro-restriction
fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor("white")
np.random.seed(42)

for party in PARTIES:
    sub = df[df["party_3cat"] == party][
        ["ap_aversion", "free_speech_restriction_index"]
    ].dropna()
    if len(sub) < 5:
        continue
    x_vals = sub["ap_aversion"].values
    y_vals = sub["free_speech_restriction_index"].values
    color  = COLORS[party]

    ax.scatter(x_vals, y_vals, alpha=0.28, s=18, color=color, linewidths=0)

    slope, intercept, r, p, _ = stats.linregress(x_vals, y_vals)
    xs = np.linspace(x_vals.min(), x_vals.max(), 200)

    # Bootstrap CI band
    boot = []
    n = len(x_vals)
    for _ in range(500):
        idx = np.random.choice(n, n, replace=True)
        s2, i2, *_ = stats.linregress(x_vals[idx], y_vals[idx])
        boot.append(s2 * xs + i2)
    boot = np.array(boot)
    ax.fill_between(xs, np.percentile(boot, 2.5, 0),
                    np.percentile(boot, 97.5, 0), color=color, alpha=0.12)

    p_str = "< .001" if p < .001 else f"= {p:.3f}"
    ax.plot(xs, slope*xs + intercept, color=color, lw=2.2,
            label=f"{party}s:  r = {r:.2f} (p {p_str}),  slope = {slope:.2f}")

ax.axhline(df["free_speech_restriction_index"].mean(), color="grey",
           lw=0.9, ls="--", alpha=0.45, label="Overall average (restriction)")

ax.set_xlabel("Social Aversion Score\n(1 = Comfortable with out-partisans  →  5 = Strongly avoids them)",
              fontsize=10.5)
ax.set_ylabel("Speech Restriction Score\n(1 = Strongly pro-free-speech  →  7 = Strongly pro-restriction)",
              fontsize=10.5)
ax.set_title("Figure 6: Do Students Who Avoid the Other Party Also Want to Restrict Speech?\n"
             "Each dot = one student. Flat lines = these are separate attitudes",
             fontsize=12, fontweight="bold", pad=10)

ax.legend(fontsize=9, framealpha=0.9, loc="upper left")

note = (
    "Key finding: The regression lines are nearly flat within each party.\n"
    "This means that avoiding the other party and wanting to restrict speech\n"
    "are largely unrelated attitudes — they measure different things."
)
ax.text(0.01, 0.01, note, transform=ax.transAxes, fontsize=8.5, va="bottom",
        bbox=dict(boxstyle="round,pad=0.45", fc="#f0f0f0", alpha=0.9, ec="grey"))

ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("visualizations/figure_06_aversion_vs_speech_restriction.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: visualizations/figure_06_aversion_vs_speech_restriction.png")
