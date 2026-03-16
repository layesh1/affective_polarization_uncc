"""
fig02_pca_biplot.py
====================
Figure 2: PCA Biplot — What patterns explain differences in polarization?

PCA finds the directions of maximum variation in the 9-item AP battery.
PC1 = the "how polarized overall" axis; PC2 = what type of polarization.

Output: visualizations/figure_02_pca_biplot.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("data/polarization_clean.csv")

is_dem = df["party_3cat"] == "Democrat"
is_rep = df["party_3cat"] == "Republican"

ITEM_LABELS = [
    "Moral (1)", "Moral (2)", "Moral (3)",
    "Othering (1)", "Othering (2)", "Othering (3)",
    "Aversion (1)", "Aversion (2)", "Aversion (3)",
]

CLUSTER_COLORS = {
    "Moral Identity":  "#1a9850",
    "Othering":        "#756bb1",
    "Social Aversion": "#e6550d",
}
ITEM_CLUSTER_IDX = {0: "Moral Identity", 1: "Moral Identity", 2: "Moral Identity",
                    3: "Othering",        4: "Othering",        5: "Othering",
                    6: "Social Aversion", 7: "Social Aversion", 8: "Social Aversion"}

features = pd.DataFrame({
    "moral_1":    np.where(is_dem, df.get("moral1D"), np.where(is_rep, df.get("moral1R"), np.nan)),
    "moral_2":    np.where(is_dem, df.get("moral2D"), np.where(is_rep, df.get("moral2R"), np.nan)),
    "moral_3":    np.where(is_dem, df.get("moral3D"), np.where(is_rep, df.get("moral3R"), np.nan)),
    "other_1":    np.where(is_dem, df.get("other1D"), np.where(is_rep, df.get("other1R"), np.nan)),
    "other_2":    np.where(is_dem, df.get("other2D"), np.where(is_rep, df.get("other2R"), np.nan)),
    "other_3":    np.where(is_dem, df.get("other3D"), np.where(is_rep, df.get("other3R"), np.nan)),
    "aversion_1": np.where(is_dem, df.get("Q138_s"),  np.where(is_rep, df.get("Q135_s"), np.nan)),
    "aversion_2": np.where(is_dem, df.get("Q139_s"),  np.where(is_rep, df.get("Q136_s"), np.nan)),
    "aversion_3": np.where(is_dem, df.get("Q140_s"),  np.where(is_rep, df.get("Q137_s"), np.nan)),
})

mask       = features.notna().all(axis=1) & df["party_combined"].notna()
X          = features[mask].values
party_cats = df.loc[mask, "party_3cat"].values

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca      = PCA(n_components=2)
scores   = pca.fit_transform(X_scaled)
loadings = pca.components_.T
pct_var  = pca.explained_variance_ratio_ * 100

scale = max(np.abs(scores[:, 0]).max(), np.abs(scores[:, 1]).max()) * 0.42

PARTY_COLORS = {"Democrat": "#2166ac", "Independent": "#888888", "Republican": "#d6604d"}
fig, ax = plt.subplots(figsize=(9, 7.5))
fig.patch.set_facecolor("white")

# Respondent dots
for party, color in PARTY_COLORS.items():
    idx = party_cats == party
    ax.scatter(scores[idx, 0], scores[idx, 1],
               c=color, alpha=0.30, s=14, linewidths=0, label=party, zorder=2)

# Loading arrows + labels
for i, (label, lx, ly) in enumerate(zip(ITEM_LABELS, loadings[:, 0], loadings[:, 1])):
    cluster = ITEM_CLUSTER_IDX[i]
    color   = CLUSTER_COLORS[cluster]
    ax.annotate("", xy=(lx*scale, ly*scale), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0), zorder=5)
    nudge = 1.22
    ax.text(lx*scale*nudge, ly*scale*nudge, label,
            fontsize=8.5, color=color, fontweight="bold", ha="center", va="center", zorder=6)

ax.axhline(0, color="grey", lw=0.6, ls="--", alpha=0.4)
ax.axvline(0, color="grey", lw=0.6, ls="--", alpha=0.4)

# Legends
party_patches   = [mpatches.Patch(color=c, label=p) for p, c in PARTY_COLORS.items()]
cluster_patches = [mpatches.Patch(color=c, label=n) for n, c in CLUSTER_COLORS.items()]
l1 = ax.legend(handles=party_patches, title="Student Party", loc="upper right",
               fontsize=9, framealpha=0.9)
ax.add_artist(l1)
ax.legend(handles=cluster_patches, title="Survey Item Type", loc="lower right",
          fontsize=9, framealpha=0.9)

ax.set_xlabel(f"Dimension 1: Overall Polarization Level  ({pct_var[0]:.1f}% of variation explained)",
              fontsize=10.5)
ax.set_ylabel(f"Dimension 2: Type of Polarization  ({pct_var[1]:.1f}% of variation explained)",
              fontsize=10.5)
ax.set_title("Figure 2: What Drives Political Distance?\n"
             "Arrows show which survey questions pull in the same direction",
             fontsize=13, fontweight="bold", pad=10)

explanation = (
    "How to read this chart:\n"
    "• Each dot = one student; dots near each other = similar answers\n"
    "• Arrows = survey questions (color = which group of questions)\n"
    "• Arrows pointing the same direction = those questions are answered similarly\n"
    "• Left = less polarized overall;  Right = more polarized overall"
)
ax.text(0.01, 0.01, explanation, transform=ax.transAxes,
        fontsize=8.5, va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="#f0f0f0", alpha=0.9, ec="grey"))

ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("visualizations/figure_02_pca_biplot.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: visualizations/figure_02_pca_biplot.png  (PC1={pct_var[0]:.1f}%, PC2={pct_var[1]:.1f}%)")
