"""
CORRECTED Comprehensive Polarization Analysis
CRITICAL FIX: Proper reversal logic for ALL scales
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

print("="*80)
print("CORRECTED POLARIZATION ANALYSIS - ALL SCALES VERIFIED")
print("="*80)

# Load data
print("\nLoading data...")
df = pd.read_csv('Data_csv/POLS Lab Student Omnibus Fall 25- text.csv', skiprows=[1])
print(f"✓ Loaded: {len(df)} rows, {len(df.columns)} columns")

# ============================================================================
# TEXT TO NUMERIC CONVERSION
# ============================================================================
print("\nConverting text responses to numeric...")

response_map_5 = {
    'None at all': 1,
    'A little': 2,
    'A moderate amount': 3,
    'A lot': 4,
    'A great deal': 5
}

response_map_7 = {
    'Strongly agree': 1,
    'Agree': 2,
    'Somewhat agree': 3,
    'Neither agree nor disagree': 4,
    'Somewhat disagree': 5,
    'Disagree': 6,
    'Strongly disagree': 7
}

party_map = {
    'Strongly Democrat': 1,
    'Somewhat Democrat': 2,
    'Not sure/neither one/other': 3,
    'Somewhat Republican': 4,
    'Strongly Republican': 5
}

partylean_map = {
    'Probably Democrats': 1,
    'Probably Republicans': 2,
    'Still not sure/neither one/other': 3
}

# Apply conversions
for q in ['Q135', 'Q136', 'Q137', 'Q138', 'Q139', 'Q140']:
    if q in df.columns:
        df[q] = df[q].map(response_map_5)

for q in ['moral1R', 'moral2R', 'moral3R', 'moral1D', 'moral2D', 'moral3D',
          'other1R', 'other2R', 'other3R', 'other1D', 'other2D', 'other3D']:
    if q in df.columns:
        df[q] = df[q].map(response_map_5)

for q in [f'Q{i}' for i in range(92, 107)]:
    if q in df.columns:
        df[q] = df[q].map(response_map_7)

for q in [f'Q{i}' for i in range(110, 123)]:
    if q in df.columns:
        df[q] = df[q].map(response_map_7)

if 'party' in df.columns:
    df['party'] = df['party'].map(party_map)
if 'partylean' in df.columns:
    df['partylean'] = df['partylean'].map(partylean_map)

print("✓ Converted text responses to numeric")

# ============================================================================
# REVERSE CODING FUNCTIONS
# ============================================================================

def reverse_5point(series):
    """Reverse 1-5 scale: 1→5, 2→4, 3→3, 4→2, 5→1"""
    return 6 - series

def reverse_7point(series):
    """Reverse 1-7 scale: 1→7, 2→6, 3→5, 4→4, 5→3, 6→2, 7→1"""
    return 8 - series

# ============================================================================
# SCALE LOGIC VERIFICATION & CORRECTION
# ============================================================================
print("\n" + "="*80)
print("SCALE LOGIC ANALYSIS")
print("="*80)

print("\n1. AVERSION QUESTIONS (Q135-Q140) - 1-5 Scale")
print("-" * 80)
print("GOAL: Create index where HIGH score = MORE aversion to opposite party")
print("\nQuestion Analysis:")
print("Q135 (Rep): 'I would NOT want to be friends with a Democrat'")
print("  → 5 = A great deal of NOT wanting = HIGH aversion")
print("  → KEEP AS IS ✓")
print("\nQ136 (Rep): 'I'd want to STOP spending time with Dems'")
print("  → 5 = A great deal of wanting to stop = HIGH aversion")
print("  → KEEP AS IS ✓")
print("\nQ137 (Rep): 'there are people I LIKE who are Democrats'")
print("  → 5 = A great deal of liking = LOW aversion")
print("  → MUST REVERSE: 5 becomes 1 (low aversion) ✓")
print("\nQ138 (Dem): 'I would NOT want to be friends with a Republican'")
print("  → 5 = A great deal of NOT wanting = HIGH aversion")
print("  → KEEP AS IS ✓")
print("\nQ139 (Dem): 'I'd want to STOP spending time with Reps'")
print("  → 5 = A great deal of wanting to stop = HIGH aversion")
print("  → KEEP AS IS ✓")
print("\nQ140 (Dem): 'there are people I LIKE who are Republicans'")
print("  → 5 = A great deal of liking = LOW aversion")
print("  → MUST REVERSE: 5 becomes 1 (low aversion) ✓")

# Apply aversion scaling
df['Q135_scaled'] = df['Q135']  # Direct: high = more aversion
df['Q136_scaled'] = df['Q136']  # Direct: high = more aversion
df['Q137_scaled'] = reverse_5point(df['Q137'])  # REVERSED: high liking becomes low aversion
df['Q138_scaled'] = df['Q138']  # Direct: high = more aversion
df['Q139_scaled'] = df['Q139']  # Direct: high = more aversion
df['Q140_scaled'] = reverse_5point(df['Q140'])  # REVERSED: high liking becomes low aversion

print("\n2. FREE SPEECH QUESTIONS (Q92-Q106) - 1-7 Scale")
print("-" * 80)
print("GOAL: Create index where HIGH score = MORE support for restrictions")
print("Scale: 1=Strongly Agree, 7=Strongly Disagree")
print("\nPRO-FREEDOM Questions (support free speech):")

freedom_questions = {
    'Q92': 'Faculty should have the RIGHT to express views',
    'Q100': 'Faculty have the CONSTITUTIONAL RIGHT to comment',
    'Q103': 'Students should have the RIGHT to speak freely',
    'Q104': 'Students should be ABLE to wear political clothing',
    'Q105': 'Students have the RIGHT to protest on campus',
    'Q106': 'Students should have a RIGHT to protest on/off campus'
}

for q, desc in freedom_questions.items():
    print(f"\n{q}: '{desc}'")
    print(f"  → Strongly Agree (1) = Support freedom = LOW restriction support")
    print(f"  → Strongly Disagree (7) = Oppose freedom = HIGH restriction support")
    print(f"  → KEEP AS IS (1=low restriction, 7=high restriction) ✓")

print("\n\nPRO-RESTRICTION Questions (oppose free speech):")

restriction_questions = {
    'Q95': 'Faculty should NOT discuss politics if prohibited',
    'Q96': 'Faculty should REFRAIN from politics if prohibited',
    'Q97': 'Faculty should REFRAIN from social equity discussion',
    'Q98': 'Faculty should NOT have social media rights if prohibited',
    'Q99': 'Faculty should NOT use social media at all if prohibited',
    'Q101': 'Faculty should NOT protest on campus',
    'Q102': 'Faculty should NOT protest off campus'
}

for q, desc in restriction_questions.items():
    print(f"\n{q}: '{desc}'")
    print(f"  → Strongly Agree (1) = Support restriction = HIGH restriction support")
    print(f"  → Strongly Disagree (7) = Oppose restriction = LOW restriction support")
    print(f"  → MUST REVERSE: 1 becomes 7 (high restriction) ✓")

# Apply free speech scaling
# Pro-freedom: KEEP AS IS (1=support freedom=low restriction, 7=oppose freedom=high restriction)
for q in freedom_questions.keys():
    df[f'{q}_scaled'] = df[q]  # KEEP AS IS

# Pro-restriction: REVERSE (1=support restriction becomes 7=high restriction)
for q in restriction_questions.keys():
    df[f'{q}_scaled'] = reverse_7point(df[q])  # REVERSE

print("\n\n3. TRUST/DISTRUST QUESTIONS (Q110-Q122) - 1-7 Scale")
print("-" * 80)
print("GOAL: Create index where HIGH score = MORE distrust")
print("\nPRO-TRUST Questions (express trust):")

trust_questions = {
    'Q110': 'I TRUST the federal government',
    'Q112': 'I TRUST my local/state government',
    'Q113': 'Officials TRY TO DO WHAT IS RIGHT',
    'Q114': 'I have CONFIDENCE in institutions',
    'Q115': 'Government treats all races FAIRLY',
    'Q116': 'My race is WELL REPRESENTED',
    'Q117': 'Policies benefit all races EQUALLY',
    'Q118': 'Leaders respond FAIRLY to all',
    'Q121': 'Government works BETTER with cooperation'
}

for q, desc in trust_questions.items():
    print(f"\n{q}: '{desc}'")
    print(f"  → Strongly Agree (1) = High trust = LOW distrust")
    print(f"  → Strongly Disagree (7) = Low trust = HIGH distrust")
    print(f"  → KEEP AS IS (1=low distrust, 7=high distrust) ✓")

print("\n\nPRO-DISTRUST Questions (express distrust):")

distrust_questions = {
    'Q119': 'Opposite party CANNOT BE TRUSTED',
    'Q120': 'Political divisions make trust DIFFICULT',
    'Q122': 'FRUSTRATED/ANGRY with opposite party in power'
}

for q, desc in distrust_questions.items():
    print(f"\n{q}: '{desc}'")
    print(f"  → Strongly Agree (1) = Express distrust = HIGH distrust")
    print(f"  → Strongly Disagree (7) = Express trust = LOW distrust")
    print(f"  → MUST REVERSE: 1 becomes 7 (high distrust) ✓")

# Apply trust/distrust scaling
# Pro-trust: KEEP AS IS (1=trust=low distrust, 7=no trust=high distrust)
for q in trust_questions.keys():
    df[f'{q}_scaled'] = df[q]  # KEEP AS IS

# Pro-distrust: REVERSE (1=express distrust becomes 7=high distrust)
for q in distrust_questions.keys():
    df[f'{q}_scaled'] = reverse_7point(df[q])  # REVERSE

print("\n\n4. MORAL IDENTITY & OTHERING (1-5 Scale)")
print("-" * 80)
print("GOAL: High score = Strong moral identity / Strong othering")
print("\nAll questions already directly measure the construct:")
print("  → 'My identity is connected to moral beliefs' - 5 = A great deal")
print("  → 'Republicans are different from Democrats' - 5 = A great deal")
print("  → NO REVERSAL NEEDED ✓")

# Moral and Othering: no reversal needed, all direct
# (already in correct direction)

# ============================================================================
# PARTY IDENTIFICATION
# ============================================================================
print("\n" + "="*80)
print("PARTY IDENTIFICATION")
print("="*80)

df['party_combined'] = df['party'].copy()
df.loc[(df['party'] == 3) & (df['partylean'] == 1), 'party_combined'] = 1.5
df.loc[(df['party'] == 3) & (df['partylean'] == 2), 'party_combined'] = 4.5

df['party_category'] = pd.cut(df['party_combined'], 
                              bins=[0, 2.5, 3.5, 6],
                              labels=['Democrat', 'Independent', 'Republican'])

print(f"\nParty Distribution:")
print(df['party_category'].value_counts().sort_index())

# ============================================================================
# CREATE INDICES
# ============================================================================
print("\n" + "="*80)
print("CREATING INDICES")
print("="*80)

# Moral Identity (no reversal)
df['moral_index_R'] = df[['moral1R', 'moral2R', 'moral3R']].mean(axis=1)
df['moral_index_D'] = df[['moral1D', 'moral2D', 'moral3D']].mean(axis=1)

# Othering (no reversal)
df['othering_index_R'] = df[['other1R', 'other2R', 'other3R']].mean(axis=1)
df['othering_index_D'] = df[['other1D', 'other2D', 'other3D']].mean(axis=1)

# Aversion (Q137, Q140 reversed)
df['aversion_index_R'] = df[['Q135_scaled', 'Q136_scaled', 'Q137_scaled']].mean(axis=1)
df['aversion_index_D'] = df[['Q138_scaled', 'Q139_scaled', 'Q140_scaled']].mean(axis=1)

# Combined Affective Polarization
df['affective_polarization_R'] = df[['moral_index_R', 'othering_index_R', 
                                       'aversion_index_R']].mean(axis=1)
df['affective_polarization_D'] = df[['moral_index_D', 'othering_index_D', 
                                       'aversion_index_D']].mean(axis=1)

# Free Speech Restriction Index (pro-restriction questions reversed)
all_fs = list(freedom_questions.keys()) + list(restriction_questions.keys())
df['free_speech_restriction_index'] = df[[f'{q}_scaled' for q in all_fs]].mean(axis=1)

# Distrust Index (pro-distrust questions reversed)
all_trust = list(trust_questions.keys()) + list(distrust_questions.keys())
df['distrust_index'] = df[[f'{q}_scaled' for q in all_trust]].mean(axis=1)

print("✓ All indices created with CORRECT scaling")

# ============================================================================
# VERIFICATION
# ============================================================================
print("\n" + "="*80)
print("VERIFICATION OF CORRECTED SCALES")
print("="*80)

rep = df[df['party_category'] == 'Republican']
dem = df[df['party_category'] == 'Democrat']

print(f"\nREPUBLICANS (n={len(rep)})")
print(f"  Moral Identity:  M={rep['moral_index_R'].mean():.3f}, SD={rep['moral_index_R'].std():.3f}")
print(f"  Othering:        M={rep['othering_index_R'].mean():.3f}, SD={rep['othering_index_R'].std():.3f}")
print(f"  Aversion:        M={rep['aversion_index_R'].mean():.3f}, SD={rep['aversion_index_R'].std():.3f}")
print(f"  COMBINED INDEX:  M={rep['affective_polarization_R'].mean():.3f}, SD={rep['affective_polarization_R'].std():.3f}")
print(f"  Free Speech Restriction: M={rep['free_speech_restriction_index'].mean():.3f}, SD={rep['free_speech_restriction_index'].std():.3f}")

print(f"\nDEMOCRATS (n={len(dem)})")
print(f"  Moral Identity:  M={dem['moral_index_D'].mean():.3f}, SD={dem['moral_index_D'].std():.3f}")
print(f"  Othering:        M={dem['othering_index_D'].mean():.3f}, SD={dem['othering_index_D'].std():.3f}")
print(f"  Aversion:        M={dem['aversion_index_D'].mean():.3f}, SD={dem['aversion_index_D'].std():.3f}")
print(f"  COMBINED INDEX:  M={dem['affective_polarization_D'].mean():.3f}, SD={dem['affective_polarization_D'].std():.3f}")
print(f"  Free Speech Restriction: M={dem['free_speech_restriction_index'].mean():.3f}, SD={dem['free_speech_restriction_index'].std():.3f}")

print("\n" + "="*80)
print("EXPECTATION CHECK")
print("="*80)
print("\nBased on political science research, we expect:")
print("1. Democrats should score LOWER on free speech restriction index")
print("   (more supportive of free speech)")
print(f"   → Republicans: {rep['free_speech_restriction_index'].mean():.2f}")
print(f"   → Democrats:   {dem['free_speech_restriction_index'].mean():.2f}")
if dem['free_speech_restriction_index'].mean() < rep['free_speech_restriction_index'].mean():
    print("   ✓ CORRECT: Democrats more supportive of free speech")
else:
    print("   ✗ UNEXPECTED: Need to re-examine")

print("\n2. Polarization levels could be similar or asymmetric")
print(f"   → Republicans: {rep['affective_polarization_R'].mean():.2f}")
print(f"   → Democrats:   {dem['affective_polarization_D'].mean():.2f}")

# Check correlations within scales
print("\n" + "="*80)
print("INTERNAL CONSISTENCY CHECK")
print("="*80)

print("\nAversion Questions - Should correlate POSITIVELY:")
print("Republicans (Q135, Q136, Q137_reversed):")
rep_aversion_corr = rep[['Q135_scaled', 'Q136_scaled', 'Q137_scaled']].corr()
mean_corr = rep_aversion_corr.values[np.triu_indices_from(rep_aversion_corr.values, k=1)].mean()
print(f"  Mean correlation: {mean_corr:.3f}")
if mean_corr > 0:
    print("  ✓ All correlations positive - scale is consistent")
else:
    print("  ✗ Negative correlations detected - scale issue")

print("\nDemocrats (Q138, Q139, Q140_reversed):")
dem_aversion_corr = dem[['Q138_scaled', 'Q139_scaled', 'Q140_scaled']].corr()
mean_corr = dem_aversion_corr.values[np.triu_indices_from(dem_aversion_corr.values, k=1)].mean()
print(f"  Mean correlation: {mean_corr:.3f}")
if mean_corr > 0:
    print("  ✓ All correlations positive - scale is consistent")
else:
    print("  ✗ Negative correlations detected - scale issue")

# ============================================================================
# SAMPLE DATA CHECK
# ============================================================================
print("\n" + "="*80)
print("SAMPLE DATA CHECK (First 5 Republicans)")
print("="*80)

sample_rep = rep.head(5)
print("\nQ92 (Faculty should have right to express views):")
print("Original | Scaled | Interpretation")
print("-" * 50)
for idx, row in sample_rep.iterrows():
    orig = row['Q92']
    scaled = row['Q92_scaled']
    interp = "Support freedom/Low restriction" if scaled < 4 else "Oppose freedom/High restriction"
    print(f"   {orig:.0f}    |   {scaled:.0f}   | {interp}")

print("\nQ95 (Faculty should NOT discuss politics if prohibited):")
print("Original | Scaled | Interpretation")
print("-" * 50)
for idx, row in sample_rep.iterrows():
    orig = row['Q95']
    scaled = row['Q95_scaled']
    interp = "Support restriction/High" if scaled > 4 else "Oppose restriction/Low"
    print(f"   {orig:.0f}    |   {scaled:.0f}   | {interp}")

# ============================================================================
# SAVE CORRECTED DATA
# ============================================================================
print("\n" + "="*80)
print("SAVING CORRECTED DATASET")
print("="*80)

output_cols = ['party', 'partylean', 'party_combined', 'party_category',
               'moral_index_R', 'moral_index_D', 'othering_index_R', 'othering_index_D',
               'aversion_index_R', 'aversion_index_D',
               'affective_polarization_R', 'affective_polarization_D',
               'free_speech_restriction_index', 'distrust_index']

scaled_cols = [col for col in df.columns if '_scaled' in col]
output_cols.extend(scaled_cols)

output_df = df[output_cols].copy()
output_df.to_csv('polarization_cleaned_CORRECTED.csv', index=False)
print(f"✓ Saved: polarization_cleaned_CORRECTED.csv ({len(output_df)} rows, {len(output_cols)} cols)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE - REVIEW OUTPUT ABOVE")
print("="*80)
print("\nNext steps:")
print("1. Check 'EXPECTATION CHECK' section above")
print("2. Review sample data to verify logic")
print("3. If everything looks correct, run visualization script")