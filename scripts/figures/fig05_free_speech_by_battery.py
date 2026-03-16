"""
fig05_free_speech_by_battery.py
================================
Figure 5: Free Speech Restriction Scores by Party and Battery

Shows mean restriction scores separately for the Faculty battery and the
Student battery.  Higher bar = more pro-restriction.

Battery classification (based on question wording in survey):
  Faculty battery  — items about faculty expression rights:
      Q92  (freedom item, reversed → restriction), Q101, Q102
  Student battery — items about student/campus speech:
      Q95, Q96, Q97, Q98, Q99  (restriction items, direct)
      Q100, Q103, Q104, Q105, Q106  (freedom items, reversed → restriction)

Restriction direction: higher score (1–7) = more supportive of restricting speech.
For freedom items stored as _s (support direction), flip back: 8 − Q??_s

Output:
    visualizations/figure_05_free_speech_by_battery.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

df = pd.read_csv("data/polarization_clean.csv")

PARTY_ORDER  = ["Democrat", "Independent", "Republican"]
PARTY_COLORS = {"Democrat": "#2166ac", "Independent": "#969696", "Republican": "#d6604d"}

# ─── BUILD RESTRICTION SCORES (higher = more restrictive) ─────────────────────
# Items stored as _s are already in SUPPORT direction (high = supports free speech).
# Flip them back to RESTRICTION direction with 8 − x.

# Faculty battery
faculty_items = {
    "q92_r":  8 - df["Q92_s"],    # freedom item → flip to restriction
    "q101_r": df["Q101_s"],        # restriction item (already restriction direction)
    "q102_r": df["Q102_s"],        # restriction item
}
df["faculty_restriction"] = pd.DataFrame(faculty_items).mean(axis=1)

# Student battery
student_items = {
    "q95_r":   df["Q95_s"],
    "q96_r":   df["Q96_s"],
    "q97_r":   df["Q97_s"],
    "q98_r":   df["Q98_s"],
    "q99_r":   df["Q99_s"],
    "q100_r":  8 - df["Q100_s"],   # freedom → flip
    "q103_r":  8 - df["Q103_s"],
    "q104_r":  8 - df["Q104_s"],
    "q105_r":  8 - df["Q105_s"],
    "q106_r":  8 - df["Q106_s"],
}
df["student_restriction"] = pd.DataFrame(student_items).mean(axis=1)


# ─── COMPUTE STATS ────────────────────────────────────────────────────────────
def party_stats(col):
    out = {}
    for party in PARTY_ORDER:
        g = df.loc[df["party_3cat"] == party, col].dropna()
        if len(g) < 3:
            continue
        n   = len(g)
        m   = g.mean()
        se  = g.std(ddof=1) / np.sqrt(n)
        ci  = se * stats.t.ppf(0.975, df=n - 1)
        out[party] = {"mean": m, "ci": ci, "n": n}
    return out

faculty_stats = party_stats("faculty_restriction")
student_stats = party_stats("student_restriction")


# ─── PLOT ─────────────────────────────────────────────────────────────────────
BATTERIES = ["Faculty Battery", "Student Battery"]
battery_data = [faculty_stats, student_stats]

x        = np.arange(len(BATTERIES))
n_parties = len(PARTY_ORDER)
bar_w    = 0.22
offsets  = np.linspace(-(n_parties - 1) * bar_w / 2,
                        (n_parties - 1) * bar_w / 2, n_parties)

fig, ax = plt.subplots(figsize=(8, 5.5))

for i, (party, offset) in enumerate(zip(PARTY_ORDER, offsets)):
    means = [bd.get(party, {}).get("mean", np.nan) for bd in battery_data]
    cis   = [bd.get(party, {}).get("ci",   np.nan) for bd in battery_data]
    bars  = ax.bar(x + offset, means, bar_w,
                   color=PARTY_COLORS[party], alpha=0.88, label=party,
                   edgecolor="white", linewidth=0.7)
    ax.errorbar(x + offset, means, yerr=cis,
                fmt="none", color="black", capsize=4, linewidth=1.2, capthick=1.2)
    # Annotate mean value above each bar
    for xi, (m, ci) in zip(x + offset, zip(means, cis)):
        if not np.isnan(m):
            ax.text(xi, m + ci + 0.06, f"{m:.2f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                    color=PARTY_COLORS[party])

# Scale midpoint reference
ax.axhline(4, color="grey", linewidth=0.8, linestyle=":", alpha=0.6)
ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 1.5, 4.05,
        "midpoint", fontsize=8, color="grey", va="bottom")

ax.set_xticks(x)
ax.set_xticklabels(BATTERIES, fontsize=12)
ax.set_ylabel("Mean Restriction Score (1–7 scale)", fontsize=11)
ax.set_ylim(1, 7.5)
ax.set_title(
    "Figure 5: Free Speech Restriction Scores by Party and Battery\n"
    "Higher = more support for restricting speech; error bars = 95% CI",
    fontsize=12, fontweight="bold",
)
legend_patches = [mpatches.Patch(color=PARTY_COLORS[p], alpha=0.88, label=p) for p in PARTY_ORDER]
ax.legend(handles=legend_patches, fontsize=10, framealpha=0.85)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("visualizations/figure_05_free_speech_by_battery.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: visualizations/figure_05_free_speech_by_battery.png")

# Print summary
for battery_label, bstats in zip(BATTERIES, battery_data):
    print(f"\n{battery_label}:")
    for party, s in bstats.items():
        print(f"  {party:<14}  M={s['mean']:.3f}  ±{s['ci']:.3f}  N={s['n']}")
