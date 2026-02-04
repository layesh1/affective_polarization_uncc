"""
AFFECTIVE POLARIZATION ANALYSIS
Examining emotional polarization, out-party hostility, moral identity, and social distance
Author: Lena

Key constructs:
- Othering: Perceiving out-party as fundamentally different
- Moralizing: Linking party identity to moral principles
- Aversion: Unwillingness to befriend out-party members
- Feeling Thermometers: Warmth toward out-party vs in-party
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu
import warnings
import os
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

print("="*80)
print("AFFECTIVE POLARIZATION ANALYSIS")
print("Measuring Emotional Distance Between Democrats and Republicans")
print("="*80)

# Load data
data_file = 'POLS Lab Student Omnibus Fall 25- text.csv'
if not os.path.exists(data_file):
    data_file = 'POLS_Lab_Student_Omnibus_Fall_25-_text.csv'
if not os.path.exists(data_file):
    print("ERROR: Cannot find data file")
    exit()

df = pd.read_csv(data_file, low_memory=False, encoding='utf-8-sig')
df = df[~df['StartDate'].str.contains('ImportId', na=False)]

# Create party binary
df['party_binary'] = df['party'].map({
    'Strongly Democrat': 'Democrat',
    'Somewhat Democrat': 'Democrat',
    'Not sure/neither one/other': 'Independent',
    'Somewhat Republican': 'Republican',
    'Strongly Republican': 'Republican'
})

print(f"\nSample: {len(df)} responses")
print(f"Democrats: {(df['party_binary']=='Democrat').sum()}")
print(f"Republicans: {(df['party_binary']=='Republican').sum()}")
print(f"Independents: {(df['party_binary']=='Independent').sum()}")

# ================== 1. MORAL IDENTITY BY PARTY ==================
print("\n" + "="*80)
print("[1] MORAL IDENTITY: Party as Moral Foundation")
print("="*80)

# Combine Democrat and Republican moral questions
moral_vars = {
    'moral1': ('moral1D', 'moral1R', 'Party identity connected to core moral beliefs'),
    'moral2': ('moral2D', 'moral2R', 'Party reflects beliefs about right and wrong'),
    'moral3': ('moral3D', 'moral3R', 'Party identity rooted in moral principles')
}

moral_map = {'None at all': 1, 'A little': 2, 'A moderate amount': 3, 'A lot': 4, 'A great deal': 5}

for key, (dem_var, rep_var, description) in moral_vars.items():
    df[key] = df[dem_var].fillna(df[rep_var]).map(moral_map)

# Calculate moral identity index (average of 3 items)
df['moral_identity_index'] = df[['moral1', 'moral2', 'moral3']].mean(axis=1)

print("\nMORAL IDENTITY INDEX (1=None at all, 5=A great deal)")
for party in ['Democrat', 'Republican']:
    party_df = df[df['party_binary'] == party]
    moral_score = party_df['moral_identity_index'].dropna()
    if len(moral_score) > 0:
        print(f"\n{party}s:")
        print(f"  Mean: {moral_score.mean():.2f}")
        print(f"  Median: {moral_score.median():.2f}")
        print(f"  SD: {moral_score.std():.2f}")
        print(f"  % High (4-5): {((moral_score >= 4).sum() / len(moral_score) * 100):.1f}%")

# T-test comparing Democrats vs Republicans
dem_moral = df[df['party_binary']=='Democrat']['moral_identity_index'].dropna()
rep_moral = df[df['party_binary']=='Republican']['moral_identity_index'].dropna()

if len(dem_moral) > 0 and len(rep_moral) > 0:
    t_stat, p_val = ttest_ind(dem_moral, rep_moral)
    print(f"\nT-test: t={t_stat:.3f}, p={p_val:.4f}")
    print(f"Conclusion: {'Significant' if p_val < 0.05 else 'No significant'} difference in moral identity strength")

# ================== 2. OTHERING: Out-Party Perceived as Different ==================
print("\n" + "="*80)
print("[2] OTHERING: Perceiving Out-Party as Fundamentally Different")
print("="*80)

other_vars = {
    'other1': ('other1D', 'other1R', 'Out-party is very different'),
    'other2': ('other2D', 'other2R', 'Out-party lives in different world'),
    'other3': ('other3D', 'other3R', 'Cannot understand out-party actions')
}

other_map = {'None at all': 1, 'A little': 2, 'A moderate amount': 3, 'A lot': 4, 'A great deal': 5}

for key, (dem_var, rep_var, description) in other_vars.items():
    df[key] = df[dem_var].fillna(df[rep_var]).map(other_map)

df['othering_index'] = df[['other1', 'other2', 'other3']].mean(axis=1)

print("\nOTHERING INDEX (1=None at all, 5=A great deal)")
for party in ['Democrat', 'Republican']:
    party_df = df[df['party_binary'] == party]
    other_score = party_df['othering_index'].dropna()
    if len(other_score) > 0:
        print(f"\n{party}s viewing out-party:")
        print(f"  Mean: {other_score.mean():.2f}")
        print(f"  Median: {other_score.median():.2f}")
        print(f"  % High (4-5): {((other_score >= 4).sum() / len(other_score) * 100):.1f}%")

dem_other = df[df['party_binary']=='Democrat']['othering_index'].dropna()
rep_other = df[df['party_binary']=='Republican']['othering_index'].dropna()

if len(dem_other) > 0 and len(rep_other) > 0:
    t_stat, p_val = ttest_ind(dem_other, rep_other)
    print(f"\nT-test: t={t_stat:.3f}, p={p_val:.4f}")

# ================== 3. AVERSION: Social Distance from Out-Party ==================
print("\n" + "="*80)
print("[3] AVERSION: Unwillingness to Befriend Out-Party Members")
print("="*80)

# Map aversion variables (Q135-Q140)
# Note: Q137/Q140 are REVERSE coded (positive statement)
aversion_map = {'None at all': 1, 'A little': 2, 'A moderate amount': 3, 'A lot': 4, 'A great deal': 5}

# Democrats' aversion to Republicans
df['aversion_friend_D'] = df['Q138'].map(aversion_map)  # Wouldn't want R friend
df['aversion_stop_D'] = df['Q139'].map(aversion_map)    # Stop spending time with R
df['aversion_like_D'] = 6 - df['Q140'].map(aversion_map)  # REVERSE: Like some Rs

# Republicans' aversion to Democrats
df['aversion_friend_R'] = df['Q135'].map(aversion_map)  # Wouldn't want D friend
df['aversion_stop_R'] = df['Q136'].map(aversion_map)    # Stop spending time with D
df['aversion_like_R'] = 6 - df['Q137'].map(aversion_map)  # REVERSE: Like some Ds

# Create unified aversion index
df['aversion_index_D'] = df[['aversion_friend_D', 'aversion_stop_D', 'aversion_like_D']].mean(axis=1)
df['aversion_index_R'] = df[['aversion_friend_R', 'aversion_stop_R', 'aversion_like_R']].mean(axis=1)

print("\nSOCIAL DISTANCE / AVERSION INDEX (1=Low, 5=High)")

dem_aversion = df[df['party_binary']=='Democrat']['aversion_index_D'].dropna()
rep_aversion = df[df['party_binary']=='Republican']['aversion_index_R'].dropna()

if len(dem_aversion) > 0:
    print(f"\nDemocrats' aversion to Republicans:")
    print(f"  Mean: {dem_aversion.mean():.2f}")
    print(f"  % High aversion (4-5): {((dem_aversion >= 4).sum() / len(dem_aversion) * 100):.1f}%")

if len(rep_aversion) > 0:
    print(f"\nRepublicans' aversion to Democrats:")
    print(f"  Mean: {rep_aversion.mean():.2f}")
    print(f"  % High aversion (4-5): {((rep_aversion >= 4).sum() / len(rep_aversion) * 100):.1f}%")

if len(dem_aversion) > 0 and len(rep_aversion) > 0:
    t_stat, p_val = ttest_ind(dem_aversion, rep_aversion)
    print(f"\nT-test: t={t_stat:.3f}, p={p_val:.4f}")
    print(f"Conclusion: {'Significant' if p_val < 0.05 else 'No'} asymmetry in aversion")

# ================== 4. FEELING THERMOMETERS ==================
print("\n" + "="*80)
print("[4] FEELING THERMOMETERS: Warmth Toward In-Party vs Out-Party")
print("="*80)

# Convert to numeric (Q148 = feelings toward Republicans, Q149 = toward Democrats)
df['FT_Republicans'] = pd.to_numeric(df['Q148'], errors='coerce')
df['FT_Democrats'] = pd.to_numeric(df['Q149'], errors='coerce')

# Create in-party and out-party thermometers
df['FT_inparty'] = np.where(df['party_binary']=='Democrat', df['FT_Democrats'], 
                    np.where(df['party_binary']=='Republican', df['FT_Republicans'], np.nan))

df['FT_outparty'] = np.where(df['party_binary']=='Democrat', df['FT_Republicans'],
                     np.where(df['party_binary']=='Republican', df['FT_Democrats'], np.nan))

# Affective polarization = In-party warmth - Out-party warmth
df['affective_polarization'] = df['FT_inparty'] - df['FT_outparty']

print("\nFEELING THERMOMETER SCORES (0-100, higher = warmer)")

for party in ['Democrat', 'Republican']:
    party_df = df[df['party_binary'] == party]
    
    inparty_ft = party_df['FT_inparty'].dropna()
    outparty_ft = party_df['FT_outparty'].dropna()
    pol_score = party_df['affective_polarization'].dropna()
    
    if len(inparty_ft) > 0 and len(outparty_ft) > 0:
        print(f"\n{party}s:")
        print(f"  Warmth toward own party: {inparty_ft.mean():.1f} (SD={inparty_ft.std():.1f})")
        print(f"  Warmth toward out-party: {outparty_ft.mean():.1f} (SD={outparty_ft.std():.1f})")
        print(f"  Affective Polarization Gap: {pol_score.mean():.1f} points")
        print(f"  % with out-party warmth <50: {((outparty_ft < 50).sum() / len(outparty_ft) * 100):.1f}%")

# Overall polarization comparison
dem_pol = df[df['party_binary']=='Democrat']['affective_polarization'].dropna()
rep_pol = df[df['party_binary']=='Republican']['affective_polarization'].dropna()

if len(dem_pol) > 0 and len(rep_pol) > 0:
    print(f"\nAFFECTIVE POLARIZATION COMPARISON:")
    print(f"  Democrats' gap: {dem_pol.mean():.1f} points (SD={dem_pol.std():.1f})")
    print(f"  Republicans' gap: {rep_pol.mean():.1f} points (SD={rep_pol.std():.1f})")
    
    t_stat, p_val = ttest_ind(dem_pol, rep_pol)
    print(f"\n  T-test: t={t_stat:.3f}, p={p_val:.4f}")
    print(f"  Conclusion: {'Asymmetric' if p_val < 0.05 else 'Symmetric'} polarization")

# ================== 5. TRUST IN OUT-PARTY MEMBERS (Q119) ==================
print("\n" + "="*80)
print("[5] OUT-PARTY TRUST: 'Opposite party cannot be trusted in government'")
print("="*80)

trust_map = {'Strongly agree': 7, 'Agree': 6, 'Somewhat agree': 5, 
             'Neither agree nor disagree': 4, 'Somewhat disagree': 3, 'Disagree': 2, 'Strongly disagree': 1}

df['outparty_distrust'] = df['Q119'].map(trust_map)

print("\nOUT-PARTY DISTRUST (1=Strongly disagree, 7=Strongly agree)")
for party in ['Democrat', 'Republican']:
    party_df = df[df['party_binary'] == party]
    distrust = party_df['outparty_distrust'].dropna()
    if len(distrust) > 0:
        print(f"\n{party}s:")
        print(f"  Mean: {distrust.mean():.2f}")
        print(f"  % Agree (5-7): {((distrust >= 5).sum() / len(distrust) * 100):.1f}%")

# ================== 6. CORRELATIONS BETWEEN AFFECTIVE POLARIZATION MEASURES ==================
print("\n" + "="*80)
print("[6] CORRELATIONS: How Affective Polarization Measures Relate")
print("="*80)

# Create correlation matrix
affective_vars = ['moral_identity_index', 'othering_index', 'affective_polarization', 'outparty_distrust']
affective_df = df[df['party_binary'].isin(['Democrat', 'Republican'])][affective_vars].dropna()

if len(affective_df) > 20:
    corr_matrix = affective_df.corr()
    print("\nCorrelation Matrix:")
    print(corr_matrix.round(3))
    
    # Key insights
    if 'moral_identity_index' in corr_matrix.columns and 'affective_polarization' in corr_matrix.columns:
        r = corr_matrix.loc['moral_identity_index', 'affective_polarization']
        print(f"\n🔍 Moral Identity × Affective Polarization: r = {r:.3f}")
        print(f"   {'Strong' if abs(r) > 0.3 else 'Weak'} relationship")

# ================== 7. SAVE CLEAN DATASET ==================
print("\n" + "="*80)
print("[7] SAVING AFFECTIVE POLARIZATION DATASET")
print("="*80)

export_cols = ['ResponseId', 'party', 'party_binary', 
               'moral_identity_index', 'moral1', 'moral2', 'moral3',
               'othering_index', 'other1', 'other2', 'other3',
               'aversion_index_D', 'aversion_index_R',
               'FT_Democrats', 'FT_Republicans', 'FT_inparty', 'FT_outparty',
               'affective_polarization', 'outparty_distrust']

export_cols = [col for col in export_cols if col in df.columns]
df[export_cols].to_csv('affective_polarization_data.csv', index=False)
print(f"✓ Saved: affective_polarization_data.csv ({len(export_cols)} variables)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)