"""
fig01_umap_projection.py
=========================
Figure 1: UMAP 2-D Projection of Affective Polarization Items by Party

Each respondent is represented as a dot.  UMAP compresses 9 survey questions
into 2 dimensions so we can see which students are attitudinally similar.
Dots are colored from blue (Strong Democrat) to red (Strong Republican).

Requires: pip install umap-learn

Output: visualizations/figure_01_umap_projection.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

try:
    import umap
except ImportError:
    raise ImportError("Install umap-learn:  pip3 install umap-learn")

df = pd.read_csv("data/polarization_clean.csv")

is_dem = df["party_3cat"] == "Democrat"
is_rep = df["party_3cat"] == "Republican"

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

mask        = features.notna().all(axis=1) & df["party_combined"].notna()
X           = features[mask].values
party_vals  = df.loc[mask, "party_combined"].values
party_cats  = df.loc[mask, "party_3cat"].values

np.random.seed(42)
reducer   = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(X)

CMAP = "RdBu_r"
norm = mcolors.Normalize(vmin=1, vmax=5)
true_indep = party_cats == "Independent"

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor("white")

sc = ax.scatter(
    embedding[~true_indep, 0], embedding[~true_indep, 1],
    c=party_vals[~true_indep], cmap=CMAP, norm=norm,
    s=22, alpha=0.72, linewidths=0, zorder=3,
)
ax.scatter(
    embedding[true_indep, 0], embedding[true_indep, 1],
    color="#888888", s=22, alpha=0.50, linewidths=0, zorder=2,
)

# Colorbar
cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
cbar.set_ticks([1, 2, 3, 4, 5])
cbar.set_ticklabels(["Strong\nDemocrat", "Somewhat\nDemocrat", "Independent",
                     "Somewhat\nRepublican", "Strong\nRepublican"])
cbar.ax.tick_params(labelsize=8.5)
cbar.set_label("Party Identification", fontsize=10, labelpad=8)

# Cluster labels
for party, color, val_range in [("Democrat\ncluster", "#2166ac", (1, 2.5)),
                                  ("Republican\ncluster", "#d6604d", (3.5, 5))]:
    mask_c = (party_vals >= val_range[0]) & (party_vals <= val_range[1])
    if mask_c.sum() > 10:
        cx = embedding[mask_c, 0].mean()
        cy = embedding[mask_c, 1].mean()
        ax.annotate(party, xy=(cx, cy), fontsize=10, color=color,
                    fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85, ec=color, lw=1.5))

# Legend for independents
ax.legend(handles=[Line2D([0],[0], marker="o", color="w", markerfacecolor="#888888",
                           markersize=9, label="Independents (no party lean)")],
          loc="lower right", fontsize=9.5, framealpha=0.9)

# Title & labels
ax.set_title("Figure 1: Attitude Map — Where Do Students Cluster by Party?",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Horizontal Position (attitude dimension 1)", fontsize=10.5)
ax.set_ylabel("Vertical Position (attitude dimension 2)", fontsize=10.5)

# Explanation box
explanation = (
    "How to read this chart:\n"
    "Each dot = one student. Dots closer together = more similar\n"
    "survey answers. Color = party identity (blue → Democrat,\n"
    "red → Republican). Grey = Independent."
)
ax.text(0.01, 0.01, explanation, transform=ax.transAxes,
        fontsize=8.5, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.5", fc="#f0f0f0", alpha=0.9, ec="grey"))

ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("visualizations/figure_01_umap_projection.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: visualizations/figure_01_umap_projection.png")
