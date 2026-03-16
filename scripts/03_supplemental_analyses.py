"""
03_supplemental_analyses.py
============================
Supplemental analyses:
  A. Race × party affiliation (chi-square)
  B. Gender × affective polarization (t-test)
  C. Free speech × affective polarization (correlation)
  D. Partisan sorting visualization (heatmap)
  E. Distrust by party

Requires cleaned data from 00_data_preparation.py.

Outputs:
    visualizations/supp_partisan_sorting.png
    visualizations/supp_free_speech_vs_ap.png
    visualizations/supp_race_party.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv("data/polarization_clean.csv")

PARTY_ORDER  = ["Democrat", "Independent", "Republican"]
PARTY_COLORS = {"Democrat": "#2166ac", "Independent": "#969696", "Republican": "#d6604d"}


# ─── A. PARTISAN SORTING HEATMAP ──────────────────────────────────────────────

if "ideology" in df.columns and "party_num" in df.columns:
    cross = pd.crosstab(df["party_3cat"], df["ideology"].round().astype("Int64"),
                        normalize="index")
    cross = cross.reindex(PARTY_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(cross, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                cbar_kws={"label": "Row proportion"})
    ax.set_xlabel("Ideology (1=Very Liberal → 7=Very Conservative)", fontsize=11)
    ax.set_ylabel("Party", fontsize=11)
    ax.set_title("Partisan Sorting: Ideology Distribution by Party", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("visualizations/supp_partisan_sorting.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: visualizations/supp_partisan_sorting.png")

    valid = df[["ideology", "party_num"]].dropna()
    rho, p = stats.spearmanr(valid["party_num"], valid["ideology"])
    print(f"Spearman r (party × ideology) = {rho:.3f}, p = {p:.4f}")


# ─── B. FREE SPEECH × AFFECTIVE POLARIZATION ─────────────────────────────────

if "free_speech_restriction_index" in df.columns and "affective_polarization_index" in df.columns:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, party in zip(axes, PARTY_ORDER):
        sub = df[df["party_3cat"] == party][
            ["free_speech_restriction_index", "affective_polarization_index"]
        ].dropna()
        if len(sub) < 5:
            ax.set_visible(False)
            continue
        r, p = stats.pearsonr(sub["free_speech_restriction_index"],
                               sub["affective_polarization_index"])
        ax.scatter(sub["free_speech_restriction_index"],
                   sub["affective_polarization_index"],
                   alpha=0.45, s=20, color=PARTY_COLORS[party])
        m, b = np.polyfit(sub["free_speech_restriction_index"],
                          sub["affective_polarization_index"], 1)
        xs = np.linspace(sub["free_speech_restriction_index"].min(),
                         sub["free_speech_restriction_index"].max(), 100)
        ax.plot(xs, m * xs + b, color="black", linewidth=1.5)
        ax.set_title(f"{party}\nr = {r:.2f}, p = {p:.3f}", fontsize=11, fontweight="bold",
                     color=PARTY_COLORS[party])
        ax.set_xlabel("Free Speech Support", fontsize=10)
        ax.set_ylabel("AP Index", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Free Speech Support vs. Affective Polarization by Party",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("visualizations/supp_free_speech_vs_ap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: visualizations/supp_free_speech_vs_ap.png")


# ─── C. RACE × PARTY (CHI-SQUARE) ─────────────────────────────────────────────

RACE_COL = "Q62"  # adjust if named differently in your dataset
if RACE_COL in df.columns and "party_3cat" in df.columns:
    ct = pd.crosstab(df[RACE_COL], df["party_3cat"])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    print(f"\nRace × Party — χ²({dof}) = {chi2:.2f}, p = {p:.4f}")

    ct_norm = ct.div(ct.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ct_norm[[c for c in PARTY_ORDER if c in ct_norm.columns]].plot(
        kind="bar", stacked=True, color=[PARTY_COLORS[p] for p in PARTY_ORDER if p in ct_norm.columns],
        ax=ax, edgecolor="white", linewidth=0.6
    )
    ax.set_xlabel("Race / Ethnicity", fontsize=11)
    ax.set_ylabel("Proportion", fontsize=11)
    ax.set_title(f"Party Affiliation by Race/Ethnicity\nχ²({dof}) = {chi2:.2f}, p = {p:.4f}",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Party", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("visualizations/supp_race_party.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: visualizations/supp_race_party.png")
