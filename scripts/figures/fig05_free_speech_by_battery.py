"""
fig05_free_speech_by_battery.py
================================
Figure 5: Who Supports More Speech Restrictions? By Party and Question Set

Higher bar = more agreement with restricting speech.

Faculty battery (Q92, Q101, Q102):  questions about faculty expression rights
Student battery (Q95–Q99, Q100, Q103–Q106): questions about student/campus speech

Note: In this UNCC sample, Republicans score higher on both batteries.
This is a specific finding for this campus context — it may reflect that
Republican students on a liberal-leaning campus favor restricting speech
they perceive as politically biased (e.g., faculty political expression).

Output: visualizations/figure_05_free_speech_by_battery.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

df = pd.read_csv("data/polarization_clean.csv")

PARTIES = ["Democrat", "Independent", "Republican"]
COLORS  = {"Democrat": "#2166ac", "Independent": "#888888", "Republican": "#d6604d"}

# ── Build restriction scores per battery ──────────────────────────────────────
# All _s columns are already in restriction direction (high = more restrictive).
# Faculty: Q92_s (reversed freedom item), Q101_s, Q102_s
# Student: Q95_s–Q99_s (restriction items), Q100_s, Q103_s–Q106_s (reversed freedom)

df["faculty_restriction"] = df[["Q92_s", "Q101_s", "Q102_s"]].mean(axis=1)
df["student_restriction"] = df[["Q95_s", "Q96_s", "Q97_s", "Q98_s", "Q99_s",
                                  "Q100_s", "Q103_s", "Q104_s", "Q105_s", "Q106_s"]].mean(axis=1)

BATTERIES = {
    "Faculty Battery\n(Q about faculty\nexpression rights)": "faculty_restriction",
    "Student Battery\n(Q about campus\nspeech norms)":       "student_restriction",
}

def party_stats(col):
    out = {}
    for p in PARTIES:
        g   = df.loc[df["party_3cat"] == p, col].dropna()
        if len(g) < 3:
            continue
        n   = len(g)
        m   = g.mean()
        ci  = stats.t.ppf(0.975, df=n-1) * g.std(ddof=1) / np.sqrt(n)
        out[p] = {"mean": m, "ci": ci, "n": n}
    return out

bstats = {lbl: party_stats(col) for lbl, col in BATTERIES.items()}

x      = np.arange(len(BATTERIES))
bar_w  = 0.24
offs   = [-bar_w, 0, bar_w]

fig, ax = plt.subplots(figsize=(9.5, 6))
fig.patch.set_facecolor("white")

for party, offset in zip(PARTIES, offs):
    means = [bstats[lbl].get(party, {}).get("mean", np.nan) for lbl in BATTERIES]
    cis   = [bstats[lbl].get(party, {}).get("ci",   np.nan) for lbl in BATTERIES]
    bars  = ax.bar(x + offset, means, bar_w, color=COLORS[party], alpha=0.88,
                   label=f"{party}s", edgecolor="white")
    ax.errorbar(x + offset, means, yerr=cis,
                fmt="none", color="black", capsize=4, lw=1.3, capthick=1.3)
    for xi, (m, ci) in zip(x + offset, zip(means, cis)):
        if not np.isnan(m):
            ax.text(xi, m + ci + 0.07, f"{m:.2f}",
                    ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold", color=COLORS[party])

ax.axhline(4, color="grey", lw=0.8, ls=":", alpha=0.5)
ax.text(len(x) - 0.42, 4.06, "Scale midpoint (4 = Neutral)", fontsize=8.5, color="grey")

ax.set_xticks(x)
ax.set_xticklabels(list(BATTERIES.keys()), fontsize=11.5)
ax.set_ylabel("Average Restriction Score\n(1 = Strongly against restricting speech  →  7 = Strongly for restricting speech)",
              fontsize=10)
ax.set_ylim(1, 7.5)
ax.set_title("Figure 5: Who Supports More Speech Restrictions?\n"
             "Average agreement with restricting faculty or student speech, by party  (error bars = 95% CI)",
             fontsize=12, fontweight="bold", pad=10)

legend_patches = [mpatches.Patch(color=COLORS[p], alpha=0.88, label=f"{p}s") for p in PARTIES]
ax.legend(handles=legend_patches, fontsize=11, framealpha=0.9, loc="upper left")

note = (
    "Note: Higher bar = more support for speech restrictions.\n"
    "In this UNCC sample, Republicans score higher on both batteries.\n"
    "Faculty battery: 3 items about faculty expression rights\n"
    "Student battery: 10 items about campus speech norms"
)
ax.text(0.01, 0.01, note, transform=ax.transAxes, fontsize=8.5, va="bottom",
        bbox=dict(boxstyle="round,pad=0.45", fc="#f0f0f0", alpha=0.9, ec="grey"))

ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("visualizations/figure_05_free_speech_by_battery.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: visualizations/figure_05_free_speech_by_battery.png")
for lbl, bs in bstats.items():
    print(f"\n{lbl.split(chr(10))[0]}:")
    for p, s in bs.items():
        print(f"  {p:<14}  M={s['mean']:.3f}  ±{s['ci']:.3f}  N={s['n']}")
