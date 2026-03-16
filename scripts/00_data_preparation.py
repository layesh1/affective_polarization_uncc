"""
00_data_preparation.py
======================
Master data preparation script for the Affective Polarization study.
Loads the raw Qualtrics export, applies all scale mappings and reverse coding,
and saves a cleaned analysis-ready DataFrame.

Run this script first before any analysis or figure script.

Data file expected at:
    data/POLS Lab Student Omnibus Fall 25- text.csv

Output:
    data/polarization_clean.csv  (gitignored — stays local)
"""

import pandas as pd
import numpy as np

# ─── 1. LOAD DATA ─────────────────────────────────────────────────────────────

DATA_PATH = "data/POLS Lab Student Omnibus Fall 25- text.csv"

# Qualtrics exports have a question-text row at row index 1 — skip it
df = pd.read_csv(DATA_PATH, low_memory=False, encoding="utf-8-sig", skiprows=[1])

print(f"Loaded {len(df)} rows × {len(df.columns)} columns")


# ─── 2. PARTY IDENTIFICATION ──────────────────────────────────────────────────
# party: 5-point scale (text labels in the -text export)
#   1 = Strongly Democrat
#   2 = Somewhat Democrat
#   3 = Not sure / neither one / other
#   4 = Somewhat Republican
#   5 = Strongly Republican
#
# partylean: direction lean for true independents (party == 3)
#   1 = Probably Democrat
#   2 = Neither / not sure
#   3 = Probably Republican

PARTY_TEXT_MAP = {
    "Strongly Democrat": 1,
    "Somewhat Democrat": 2,
    "Not sure/neither one/other": 3,
    "Not sure/neither/other": 3,
    "Somewhat Republican": 4,
    "Strongly Republican": 5,
}
PARTYLEAN_TEXT_MAP = {
    "Probably Democrat": 1,
    "Neither / not sure": 2,
    "Neither/not sure": 2,
    "Probably Republican": 3,
}

# Map text → numeric if the column is still text
if df["party"].dtype == object:
    df["party_num"] = df["party"].map(PARTY_TEXT_MAP)
else:
    df["party_num"] = df["party"].astype(float)

if df["partylean"].dtype == object:
    df["partylean_num"] = df["partylean"].map(PARTYLEAN_TEXT_MAP)
else:
    df["partylean_num"] = df["partylean"].astype(float)

# Combine: lean-Dems → 1.5, lean-Reps → 4.5
df["party_combined"] = df["party_num"].copy()
df.loc[(df["party_num"] == 3) & (df["partylean_num"] == 1), "party_combined"] = 1.5
df.loc[(df["party_num"] == 3) & (df["partylean_num"] == 3), "party_combined"] = 4.5

# 3-category label for analyses
df["party_3cat"] = pd.cut(
    df["party_combined"],
    bins=[0, 2.5, 3.5, 6],
    labels=["Democrat", "Independent", "Republican"],
)

# 7-category label for UMAP gradient (Strong Dem → Strong Rep)
PARTY_7_MAP = {1: "Strong Dem", 1.5: "Lean Dem", 2: "Somewhat Dem",
               3: "Independent", 4.5: "Lean Rep", 4: "Somewhat Rep",
               5: "Strong Rep"}
df["party_7cat"] = df["party_combined"].map(PARTY_7_MAP)


# ─── 3. SCALE MAPPINGS ────────────────────────────────────────────────────────

def reverse_5(series):
    """Reverse a 1–5 Likert scale: 1→5, 2→4, 3→3, 4→2, 5→1."""
    return 6 - series

def reverse_7(series):
    """Reverse a 1–7 Likert scale: 1→7, 2→6, 3→5, 4→4, 5→3, 6→2, 7→1."""
    return 8 - series

def map_5point(series):
    """Convert text responses from a 5-point None-to-Great-Deal scale to 1–5."""
    mapping = {
        "None at all": 1,
        "A little": 2,
        "A moderate amount": 3,
        "A lot": 4,
        "A great deal": 5,
    }
    return series.map(mapping) if series.dtype == object else series.astype(float)

def map_5agree(series):
    """Convert text responses from Strongly Disagree–Strongly Agree (5-pt) to 1–5."""
    mapping = {
        "Strongly disagree": 1,
        "Somewhat disagree": 2,
        "Neither agree nor disagree": 3,
        "Somewhat agree": 4,
        "Strongly agree": 5,
    }
    return series.map(mapping) if series.dtype == object else series.astype(float)

def map_7agree(series):
    """Convert text responses from Strongly Disagree–Strongly Agree (7-pt) to 1–7."""
    mapping = {
        "Strongly disagree": 1,
        "Disagree": 2,
        "Somewhat disagree": 3,
        "Neither agree nor disagree": 4,
        "Somewhat agree": 5,
        "Agree": 6,
        "Strongly agree": 7,
    }
    return series.map(mapping) if series.dtype == object else series.astype(float)


# ─── 4. MORAL IDENTITY ITEMS (1–5, no reversal needed) ───────────────────────
# Items ask how much party identity is connected to core moral beliefs.
# Higher = stronger moral-identity tie.
# Republican versions: moral1R, moral2R, moral3R
# Democrat versions:   moral1D, moral2D, moral3D

MORAL_R_COLS = ["moral1R", "moral2R", "moral3R"]
MORAL_D_COLS = ["moral1D", "moral2D", "moral3D"]

for col in MORAL_R_COLS + MORAL_D_COLS:
    df[col] = map_5point(df[col])

df["moral_index_R"] = df[MORAL_R_COLS].mean(axis=1)
df["moral_index_D"] = df[MORAL_D_COLS].mean(axis=1)


# ─── 5. OTHERING ITEMS (1–5, no reversal needed) ─────────────────────────────
# Items ask how different / alien the out-party seems.
# Higher = stronger othering.
# Republican-toward-Democrat versions: other1R, other2R, other3R
# Democrat-toward-Republican versions: other1D, other2D, other3D

OTHER_R_COLS = ["other1R", "other2R", "other3R"]
OTHER_D_COLS = ["other1D", "other2D", "other3D"]

for col in OTHER_R_COLS + OTHER_D_COLS:
    df[col] = map_5point(df[col])

df["othering_index_R"] = df[OTHER_R_COLS].mean(axis=1)
df["othering_index_D"] = df[OTHER_D_COLS].mean(axis=1)


# ─── 6. SOCIAL AVERSION ITEMS (1–5, Q137 & Q140 reversed) ───────────────────
# Republican aversion toward Democrats:
#   Q135: "I would not want to be friends with a Democrat."          (DIRECT)
#   Q136: "I would want to stop spending time with a Democrat friend."(DIRECT)
#   Q137: "There are people I like who are Democrats."               (REVERSED: liking → aversion)
#
# Democrat aversion toward Republicans:
#   Q138: "I would not want to be friends with a Republican."        (DIRECT)
#   Q139: "I would want to stop spending time with a Republican."    (DIRECT)
#   Q140: "There are people I like who are Republicans."             (REVERSED: liking → aversion)

for col in ["Q135", "Q136", "Q137", "Q138", "Q139", "Q140"]:
    df[col] = map_5agree(df[col])

df["Q135_s"] = df["Q135"]                  # direct
df["Q136_s"] = df["Q136"]                  # direct
df["Q137_s"] = reverse_5(df["Q137"])       # REVERSED
df["Q138_s"] = df["Q138"]                  # direct
df["Q139_s"] = df["Q139"]                  # direct
df["Q140_s"] = reverse_5(df["Q140"])       # REVERSED

df["aversion_index_R"] = df[["Q135_s", "Q136_s", "Q137_s"]].mean(axis=1)
df["aversion_index_D"] = df[["Q138_s", "Q139_s", "Q140_s"]].mean(axis=1)


# ─── 7. COMBINED AFFECTIVE POLARIZATION INDEX ─────────────────────────────────
# Per-respondent composite: uses own-party version of each component.
# Dems: moral_D + othering_D + aversion_D
# Reps: moral_R + othering_R + aversion_R
# Independents: NaN (excluded from partisan comparisons)

is_dem = df["party_3cat"] == "Democrat"
is_rep = df["party_3cat"] == "Republican"

df["ap_moral"]    = np.where(is_dem, df["moral_index_D"],
                    np.where(is_rep, df["moral_index_R"], np.nan))
df["ap_othering"] = np.where(is_dem, df["othering_index_D"],
                    np.where(is_rep, df["othering_index_R"], np.nan))
df["ap_aversion"] = np.where(is_dem, df["aversion_index_D"],
                    np.where(is_rep, df["aversion_index_R"], np.nan))

df["affective_polarization_index"] = df[["ap_moral", "ap_othering", "ap_aversion"]].mean(axis=1)


# ─── 8. FEELING THERMOMETERS (0–100) ─────────────────────────────────────────
# Q148: warmth toward Republicans (0=very cold, 100=very warm)
# Q149: warmth toward Democrats

df["FT_Republicans"] = pd.to_numeric(df["Q148"], errors="coerce")
df["FT_Democrats"]   = pd.to_numeric(df["Q149"], errors="coerce")

df["FT_inparty"]  = np.where(is_dem, df["FT_Democrats"],
                    np.where(is_rep, df["FT_Republicans"], np.nan))
df["FT_outparty"] = np.where(is_dem, df["FT_Republicans"],
                    np.where(is_rep, df["FT_Democrats"], np.nan))
df["FT_gap"] = df["FT_inparty"] - df["FT_outparty"]


# ─── 9. FREE SPEECH ITEMS (1–7) ───────────────────────────────────────────────
# 1 = Strongly Agree, 7 = Strongly Disagree
# After reverse-coding, higher score = greater SUPPORT for free speech.
#
# Pro-FREEDOM questions (REVERSED so high=support freedom):
#   Q92, Q100, Q103, Q104, Q105, Q106  →  8 - original
#
# Pro-RESTRICTION questions (kept as-is: high=oppose restriction = support freedom):
#   Q95, Q96, Q97, Q98, Q99, Q101, Q102

FREEDOM_COLS      = ["Q92", "Q100", "Q103", "Q104", "Q105", "Q106"]
RESTRICTION_COLS  = ["Q95", "Q96", "Q97", "Q98", "Q99", "Q101", "Q102"]

for col in FREEDOM_COLS + RESTRICTION_COLS:
    df[col] = map_7agree(df[col])

for col in FREEDOM_COLS:
    df[f"{col}_s"] = reverse_7(df[col])    # REVERSED

for col in RESTRICTION_COLS:
    df[f"{col}_s"] = df[col]               # kept as-is

FS_SCALED = [f"{c}_s" for c in FREEDOM_COLS + RESTRICTION_COLS]
df["free_speech_support_index"] = df[FS_SCALED].mean(axis=1)


# ─── 10. TRUST / DISTRUST ITEMS (1–7) ────────────────────────────────────────
# After coding, higher score = greater OUT-PARTY DISTRUST.
#
# Pro-TRUST questions (REVERSED so high=distrust):
#   Q110, Q112, Q113, Q114, Q115, Q116, Q117, Q118, Q121  →  8 - original
#
# Pro-DISTRUST questions (kept as-is: high=distrust):
#   Q119, Q120, Q122

TRUST_COLS   = ["Q110", "Q112", "Q113", "Q114", "Q115", "Q116", "Q117", "Q118", "Q121"]
DISTRUST_COLS = ["Q119", "Q120", "Q122"]

for col in TRUST_COLS + DISTRUST_COLS:
    df[col] = map_7agree(df[col])

for col in TRUST_COLS:
    df[f"{col}_s"] = reverse_7(df[col])    # REVERSED (trust → distrust direction)

for col in DISTRUST_COLS:
    df[f"{col}_s"] = df[col]               # kept as-is

DISTRUST_SCALED = [f"{c}_s" for c in TRUST_COLS + DISTRUST_COLS]
df["distrust_index"] = df[DISTRUST_SCALED].mean(axis=1)


# ─── 11. DEMOGRAPHICS ────────────────────────────────────────────────────────
# Ideology: 7-point scale (1=Very liberal … 7=Very conservative)
# Race: Q62
# Gender: as recorded in survey

df["ideology"] = pd.to_numeric(df.get("ideology", pd.Series(dtype=float)), errors="coerce")


# ─── 12. SAVE CLEANED DATA ────────────────────────────────────────────────────

OUTPUT_COLS = [
    # Party
    "party_num", "partylean_num", "party_combined", "party_3cat", "party_7cat",
    # Moral identity (raw + index)
    *MORAL_R_COLS, *MORAL_D_COLS, "moral_index_R", "moral_index_D",
    # Othering (raw + index)
    *OTHER_R_COLS, *OTHER_D_COLS, "othering_index_R", "othering_index_D",
    # Aversion (raw scaled + index)
    "Q135_s", "Q136_s", "Q137_s", "Q138_s", "Q139_s", "Q140_s",
    "aversion_index_R", "aversion_index_D",
    # Per-respondent AP components
    "ap_moral", "ap_othering", "ap_aversion", "affective_polarization_index",
    # Feeling thermometers
    "FT_Republicans", "FT_Democrats", "FT_inparty", "FT_outparty", "FT_gap",
    # Free speech
    *FS_SCALED, "free_speech_support_index",
    # Distrust
    *DISTRUST_SCALED, "distrust_index",
    # Demographics
    "ideology",
]

# Keep only columns that exist in the DataFrame
OUTPUT_COLS = [c for c in OUTPUT_COLS if c in df.columns]
df[OUTPUT_COLS].to_csv("data/polarization_clean.csv", index=False)
print(f"Saved cleaned data → data/polarization_clean.csv  ({len(df)} rows)")
print(f"\nParty breakdown:\n{df['party_3cat'].value_counts()}")
