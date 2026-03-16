# Variable Codebook

Data file: `POLS Lab Student Omnibus Fall 25- text.csv`
Cleaned output: `data/polarization_clean.csv`

---

## Party Identification

| Variable | Type | Values | Notes |
|----------|------|--------|-------|
| `party_num` | Numeric | 1–5 | Converted from text labels |
| `partylean_num` | Numeric | 1–3 | 1=Lean Dem, 2=Neither, 3=Lean Rep |
| `party_combined` | Numeric | 1, 1.5, 2, 3, 4, 4.5, 5 | Leaners folded in at 1.5/4.5 |
| `party_3cat` | Categorical | Democrat / Independent / Republican | Cut at 2.5 and 3.5 |
| `party_7cat` | Categorical | Strong Dem → Strong Rep | 7-level label for gradient plots |

---

## Moral Identity Items (1–5 scale)

> Response options: None at all (1) → A little (2) → A moderate amount (3) → A lot (4) → A great deal (5)
> **No reverse coding applied.**

| Variable | Party | Question |
|----------|-------|----------|
| `moral1R` | Republican | Party identity connected to core moral beliefs |
| `moral2R` | Republican | Party reflects beliefs about right and wrong |
| `moral3R` | Republican | Party identity rooted in moral principles |
| `moral1D` | Democrat | Party identity connected to core moral beliefs |
| `moral2D` | Democrat | Party reflects beliefs about right and wrong |
| `moral3D` | Democrat | Party identity rooted in moral principles |
| `moral_index_R` | Republican | Mean of moral1R–moral3R |
| `moral_index_D` | Democrat | Mean of moral1D–moral3D |
| `ap_moral` | Per respondent | Uses D index for Dems, R index for Reps |

---

## Othering Items (1–5 scale)

> Response options: None at all (1) → A little (2) → A moderate amount (3) → A lot (4) → A great deal (5)
> **No reverse coding applied.**

| Variable | Party | Question |
|----------|-------|----------|
| `other1R` | Republican | Out-party (Dems) is very different |
| `other2R` | Republican | Out-party lives in a different world |
| `other3R` | Republican | Cannot understand out-party actions |
| `other1D` | Democrat | Out-party (Reps) is very different |
| `other2D` | Democrat | Out-party lives in a different world |
| `other3D` | Democrat | Cannot understand out-party actions |
| `othering_index_R` | Republican | Mean of other1R–other3R |
| `othering_index_D` | Democrat | Mean of other1D–other3D |
| `ap_othering` | Per respondent | Uses D index for Dems, R index for Reps |

---

## Social Aversion Items (1–5 scale)

> Response options: Strongly Disagree (1) → Strongly Agree (5)
> **Q137 and Q140 are REVERSE CODED** (formula: `6 − original`).

| Variable | Party | Question | Direction |
|----------|-------|----------|-----------|
| `Q135` | Republican | "Would not want to be friends with a Democrat" | Direct |
| `Q135_s` | Republican | Scaled version (same as Q135) | — |
| `Q136` | Republican | "Would stop spending time with a Democratic friend" | Direct |
| `Q136_s` | Republican | Scaled version | — |
| `Q137` | Republican | "There are people I like who are Democrats" | **REVERSED** |
| `Q137_s` | Republican | `6 − Q137` | — |
| `Q138` | Democrat | "Would not want to be friends with a Republican" | Direct |
| `Q138_s` | Democrat | Scaled version | — |
| `Q139` | Democrat | "Would stop spending time with a Republican friend" | Direct |
| `Q139_s` | Democrat | Scaled version | — |
| `Q140` | Democrat | "There are people I like who are Republicans" | **REVERSED** |
| `Q140_s` | Democrat | `6 − Q140` | — |
| `aversion_index_R` | Republican | Mean of Q135_s, Q136_s, Q137_s | — |
| `aversion_index_D` | Democrat | Mean of Q138_s, Q139_s, Q140_s | — |
| `ap_aversion` | Per respondent | Uses D index for Dems, R index for Reps | — |

---

## Combined Affective Polarization Index

| Variable | Range | Formula |
|----------|-------|---------|
| `affective_polarization_index` | 1–5 | Mean of `ap_moral`, `ap_othering`, `ap_aversion` |

Higher scores = greater affective polarization.
Independents receive NaN and are excluded from partisan comparisons.

---

## Feeling Thermometers (0–100)

| Variable | Description |
|----------|-------------|
| `FT_Republicans` (Q148) | Warmth toward Republicans |
| `FT_Democrats` (Q149) | Warmth toward Democrats |
| `FT_inparty` | Own-party warmth (Q149 for Dems; Q148 for Reps) |
| `FT_outparty` | Out-party warmth (Q148 for Dems; Q149 for Reps) |
| `FT_gap` | `FT_inparty − FT_outparty` (higher = more polarized) |

---

## Free Speech Index (1–7 → rescaled)

After coding, higher score = **greater support for free speech**.

### Pro-freedom items (REVERSED: `8 − original`)

| Column | Topic |
|--------|-------|
| Q92 | Government should not restrict speech |
| Q100 | Controversial speakers should be allowed on campus |
| Q103 | People should be free to express unpopular views |
| Q104 | Free expression is essential to democracy |
| Q105 | Even offensive speech should be protected |
| Q106 | Censorship causes more harm than the speech itself |

### Pro-restriction items (kept as-is)

| Column | Topic |
|--------|-------|
| Q95 | Hate speech should be regulated |
| Q96 | Speech that causes harm should be limited |
| Q97 | Universities should restrict offensive content |
| Q98 | Social media platforms should remove harmful speech |
| Q99 | Some speech deserves no protection |
| Q101 | Faculty speech should be subject to professional norms |
| Q102 | Students reporting offensive speech is appropriate |

| Variable | Formula |
|----------|---------|
| `free_speech_support_index` | Mean of all 13 scaled items |

---

## Distrust Index (1–7 → rescaled)

After coding, higher score = **greater out-party distrust**.

### Pro-trust items (REVERSED: `8 − original`)

Q110, Q112, Q113, Q114, Q115, Q116, Q117, Q118, Q121

### Pro-distrust items (kept as-is)

| Column | Topic |
|--------|-------|
| Q119 | Opposite party cannot be trusted |
| Q120 | Political divisions make trust difficult |
| Q122 | Feel frustrated/angry with opposite party |

| Variable | Formula |
|----------|---------|
| `distrust_index` | Mean of all 12 scaled items |

---

## Demographics

| Variable | Description |
|----------|-------------|
| `ideology` | 7-point scale: 1=Very Liberal → 7=Very Conservative |
| `Q62` | Race / Ethnicity (categorical) |
| Gender | As recorded in survey |
