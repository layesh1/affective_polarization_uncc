"""
fig01_umap_projection.py
=========================
Figure 1: UMAP 2-D Projection of Affective Polarization Items by Party

Each respondent is represented as a point in 2-D UMAP space computed from
their 9 affective-polarization items (3 moral identity + 3 othering + 3 social
aversion), using their own-party version of each item.  Points are colored on a
continuous gradient from Strong Democrat (blue) to Strong Republican (red), with
True Independents in grey.

Install:  pip install umap-learn

Output:
    visualizations/figure_01_umap_projection.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.lines import Line2D

try:
    import umap
except ImportError:
    raise ImportError(
        "umap-learn is required. Install with:  pip install umap-learn"
    )

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
df = pd.read_csv("data/polarization_clean.csv")

# ─── BUILD FEATURE MATRIX ─────────────────────────────────────────────────────
# For each respondent: 9 features = [3 moral, 3 othering, 3 aversion]
# using the own-party version of each item.

# Moral items: Democrats answered moral1D–moral3D; Republicans moral1R–moral3R
# Othering:    Democrats other1D–other3D;   Republicans other1R–other3R
# Aversion:    Democrats Q138_s–Q140_s;     Republicans Q135_s–Q137_s

is_dem = df["party_3cat"] == "Democrat"
is_rep = df["party_3cat"] == "Republican"

# 9-column feature array (NaN where items don't apply)
features = pd.DataFrame({
    "moral_1": np.where(is_dem, df.get("moral1D"), np.where(is_rep, df.get("moral1R"), np.nan)),
    "moral_2": np.where(is_dem, df.get("moral2D"), np.where(is_rep, df.get("moral2R"), np.nan)),
    "moral_3": np.where(is_dem, df.get("moral3D"), np.where(is_rep, df.get("moral3R"), np.nan)),
    "other_1": np.where(is_dem, df.get("other1D"), np.where(is_rep, df.get("other1R"), np.nan)),
    "other_2": np.where(is_dem, df.get("other2D"), np.where(is_rep, df.get("other2R"), np.nan)),
    "other_3": np.where(is_dem, df.get("other3D"), np.where(is_rep, df.get("other3R"), np.nan)),
    "aversion_1": np.where(is_dem, df.get("Q138_s"), np.where(is_rep, df.get("Q135_s"), np.nan)),
    "aversion_2": np.where(is_dem, df.get("Q139_s"), np.where(is_rep, df.get("Q136_s"), np.nan)),
    "aversion_3": np.where(is_dem, df.get("Q140_s"), np.where(is_rep, df.get("Q137_s"), np.nan)),
})

# Keep rows with complete data and a known party
mask = features.notna().all(axis=1) & df["party_combined"].notna()
X = features[mask].values
party_vals = df.loc[mask, "party_combined"].values      # numeric 1–5 gradient
party_cats = df.loc[mask, "party_3cat"].values

# ─── RUN UMAP ─────────────────────────────────────────────────────────────────
np.random.seed(42)
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(X)

# ─── COLOR MAPPING ────────────────────────────────────────────────────────────
# Continuous gradient: 1=Strong Dem (blue) → 3=Independent (grey) → 5=Strong Rep (red)
# Use a diverging colormap centered at 3 (Independent)

CMAP_NAME = "RdBu_r"  # blue at low end (Democrats), red at high end (Republicans)
cmap = plt.get_cmap(CMAP_NAME)
norm = mcolors.Normalize(vmin=1, vmax=5)
colors = cmap(norm(party_vals))

# Independents (party_combined == 3, 1.5 leaners handled by gradient)
true_indep = party_cats == "Independent"

# ─── PLOT ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6.5))

sc = ax.scatter(
    embedding[~true_indep, 0], embedding[~true_indep, 1],
    c=party_vals[~true_indep], cmap=CMAP_NAME, norm=norm,
    s=18, alpha=0.70, linewidths=0,
)
ax.scatter(
    embedding[true_indep, 0], embedding[true_indep, 1],
    color="#969696", s=18, alpha=0.55, linewidths=0, zorder=2, label="True Independent",
)

# Colorbar
cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
cbar.set_ticks([1, 2, 3, 4, 5])
cbar.set_ticklabels(["Strong\nDem", "Somewhat\nDem", "Independent", "Somewhat\nRep", "Strong\nRep"])
cbar.ax.tick_params(labelsize=8)
cbar.set_label("Party Identification", fontsize=10)

# Legend for independents
legend_elements = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#969696",
                          markersize=8, label="True Independent (no lean)")]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.8)

ax.set_xlabel("UMAP Dimension 1", fontsize=11)
ax.set_ylabel("UMAP Dimension 2", fontsize=11)
ax.set_title(
    "Figure 1: UMAP Projection of Affective Polarization Items by Party\n"
    "(n=9 items: moral identity + othering + social aversion)",
    fontsize=12, fontweight="bold", pad=10,
)

# Directional annotation
n_dem = np.sum(party_vals < 2.5)
n_rep = np.sum(party_vals > 3.5)
if n_dem > 0 and n_rep > 0:
    dem_cx = embedding[party_vals < 2.5, 0].mean()
    dem_cy = embedding[party_vals < 2.5, 1].mean()
    rep_cx = embedding[party_vals > 3.5, 0].mean()
    rep_cy = embedding[party_vals > 3.5, 1].mean()
    ax.annotate("Democrat\ncluster", xy=(dem_cx, dem_cy),
                fontsize=9, color="#2166ac", fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="#2166ac"))
    ax.annotate("Republican\ncluster", xy=(rep_cx, rep_cy),
                fontsize=9, color="#d6604d", fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="#d6604d"))

ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("visualizations/figure_01_umap_projection.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: visualizations/figure_01_umap_projection.png")
