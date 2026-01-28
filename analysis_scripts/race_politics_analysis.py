"""
RACE × POLITICAL AFFILIATION ANALYSIS
Examining how race intersects with party ID, ideology, policy attitudes, and trust
Author: Lena

Research questions:
- How does party affiliation vary by race?
- Do racial groups differ in trust in government?
- How do racial policy attitudes (Q115-Q118) vary by party?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
import warnings
import os
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

print("="*80)
print("RACE × POLITICAL AFFILIATION ANALYSIS")
print("="*80)

# Load data
data_file = 'POLS Lab Student Omnibus Fall 25- text.csv'
if not os.path.exists(data_file):
    data_file = 'POLS_Lab_Student_Omnibus_Fall_25-_text.csv'

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

# Map ideology
ideology_map = {
    'Very liberal': 1, 'Liberal': 2, 'Somewhat liberal': 3, 'Moderate': 4,
    'Somewhat conservative': 5, 'Conservative': 6, 'Very conservative': 7
}
df['ideology_numeric'] = df['ideology'].map(ideology_map)

# Clean race variable (Q62)
df['race'] = df['Q62']

print(f"Total sample: {len(df)}")
print(f"With race data: {df['race'].notna().sum()}")

# ================== 1. RACE DISTRIBUTION ==================
print("\n" + "="*80)
print("[1] RACIAL COMPOSITION OF SAMPLE")
print("="*80)

race_counts = df['race'].value_counts()
print("\nRace/Ethnicity:")
for race, count in race_counts.head(10).items():
    print(f"  {race}: {count} ({count/len(df)*100:.1f}%)")

# Create simplified race categories
def categorize_race(race_str):
    if pd.isna(race_str):
        return 'Missing'
    race_str = str(race_str).lower()
    if 'white' in race_str:
        return 'White'
    elif 'black' in race_str or 'african american' in race_str:
        return 'Black/African American'
    elif 'hispanic' in race_str or 'latino' in race_str:
        return 'Hispanic/Latino'
    elif 'asian' in race_str:
        return 'Asian'
    elif 'middle eastern' in race_str:
        return 'Middle Eastern/North African'
    elif ',' in race_str:
        return 'Multiracial'
    else:
        return 'Other'

df['race_category'] = df['race'].apply(categorize_race)

print("\nSimplified Race Categories:")
for race, count in df['race_category'].value_counts().items():
    print(f"  {race}: {count} ({count/len(df)*100:.1f}%)")

# ================== 2. PARTY AFFILIATION BY RACE ==================
print("\n" + "="*80)
print("[2] PARTY AFFILIATION BY RACE")
print("="*80)

# Cross-tabulation
race_party_crosstab = pd.crosstab(df['race_category'], df['party_binary'], normalize='index') * 100

print("\nParty Affiliation by Race (% within each racial group):")
print(race_party_crosstab.round(1))

# Chi-square test
chi2, p_val, dof, expected = chi2_contingency(pd.crosstab(df['race_category'], df['party_binary']))
print(f"\nChi-square test: χ²={chi2:.2f}, p={p_val:.4f}")
print(f"Conclusion: {'Significant' if p_val < 0.05 else 'No significant'} relationship between race and party")

# ================== 3. IDEOLOGY BY RACE ==================
print("\n" + "="*80)
print("[3] IDEOLOGICAL SELF-PLACEMENT BY RACE")
print("="*80)

print("\nMean Ideology by Race (1=Very Liberal, 7=Very Conservative):")
for race in ['White', 'Black/African American', 'Hispanic/Latino', 'Asian']:
    race_df = df[df['race_category'] == race]
    ideology = race_df['ideology_numeric'].dropna()
    if len(ideology) > 10:
        print(f"\n{race}:")
        print(f"  Mean: {ideology.mean():.2f}")
        print(f"  Median: {ideology.median():.1f}")
        print(f"  SD: {ideology.std():.2f}")
        print(f"  n = {len(ideology)}")

# ANOVA test
race_groups = []
race_labels = []
for race in ['White', 'Black/African American', 'Hispanic/Latino', 'Asian']:
    ideology = df[df['race_category'] == race]['ideology_numeric'].dropna()
    if len(ideology) > 5:
        race_groups.append(ideology)
        race_labels.append(race)

if len(race_groups) >= 2:
    f_stat, p_val = stats.f_oneway(*race_groups)
    print(f"\nANOVA test: F={f_stat:.3f}, p={p_val:.4f}")
    print(f"Conclusion: {'Significant' if p_val < 0.05 else 'No significant'} ideological differences across races")

# ================== 4. TRUST IN GOVERNMENT BY RACE ==================
print("\n" + "="*80)
print("[4] TRUST IN GOVERNMENT BY RACE")
print("="*80)

# Map trust variables (1=Strongly agree, 7=Strongly disagree)
# We'll reverse code so higher = more trust
trust_map = {'Strongly agree': 7, 'Agree': 6, 'Somewhat agree': 5,
             'Neither agree nor disagree': 4, 'Somewhat disagree': 3, 'Disagree': 2, 'Strongly disagree': 1}

# Q110: Trust federal government
df['trust_federal'] = df['Q110'].map(trust_map)
# Q112: Trust local/state government  
df['trust_local'] = df['Q112'].map(trust_map)

print("\nTrust in Federal Government by Race (1=Low, 7=High):")
for race in ['White', 'Black/African American', 'Hispanic/Latino']:
    race_df = df[df['race_category'] == race]
    trust = race_df['trust_federal'].dropna()
    if len(trust) > 10:
        print(f"\n{race}:")
        print(f"  Mean: {trust.mean():.2f}")
        print(f"  % High trust (5-7): {((trust >= 5).sum() / len(trust) * 100):.1f}%")

# ================== 5. RACIAL FAIRNESS PERCEPTIONS ==================
print("\n" + "="*80)
print("[5] PERCEPTIONS OF RACIAL FAIRNESS IN GOVERNMENT")
print("="*80)

# Q115: Government treats all racial groups fairly
# Q116: My racial/ethnic background well represented in government
# Q117: Public policies benefit all racial groups equally
# Q118: Government responds fairly to all communities on racial issues

fairness_vars = {
    'Q115': 'Government treats all races fairly',
    'Q116': 'My race well represented in government',
    'Q117': 'Policies benefit all races equally',
    'Q118': 'Government responds fairly to racial issues'
}

for var, description in fairness_vars.items():
    df[f'{var}_numeric'] = df[var].map(trust_map)

# Create racial fairness index
df['racial_fairness_index'] = df[['Q115_numeric', 'Q116_numeric', 'Q117_numeric', 'Q118_numeric']].mean(axis=1)

print("\nRACIAL FAIRNESS INDEX by Race (1=Strongly disagree, 7=Strongly agree):")
for race in ['White', 'Black/African American', 'Hispanic/Latino']:
    race_df = df[df['race_category'] == race]
    fairness = race_df['racial_fairness_index'].dropna()
    if len(fairness) > 10:
        print(f"\n{race}:")
        print(f"  Mean: {fairness.mean():.2f}")
        print(f"  SD: {fairness.std():.2f}")
        print(f"  % Agree gov't is fair (5-7): {((fairness >= 5).sum() / len(fairness) * 100):.1f}%")

# ANOVA comparing races
race_fairness_groups = []
for race in ['White', 'Black/African American', 'Hispanic/Latino']:
    fairness = df[df['race_category'] == race]['racial_fairness_index'].dropna()
    if len(fairness) > 5:
        race_fairness_groups.append(fairness)

if len(race_fairness_groups) >= 2:
    f_stat, p_val = stats.f_oneway(*race_fairness_groups)
    print(f"\nANOVA: F={f_stat:.3f}, p={p_val:.4f}")
    print(f"Conclusion: {'Significant' if p_val < 0.05 else 'No significant'} differences in racial fairness perceptions")

# ================== 6. RACE × PARTY INTERACTION ON RACIAL FAIRNESS ==================
print("\n" + "="*80)
print("[6] RACE × PARTY: Perceptions of Racial Fairness")
print("="*80)

print("\nRACIAL FAIRNESS INDEX by Race and Party:")
for race in ['White', 'Black/African American', 'Hispanic/Latino']:
    print(f"\n{race}:")
    for party in ['Democrat', 'Republican']:
        subset = df[(df['race_category']==race) & (df['party_binary']==party)]
        fairness = subset['racial_fairness_index'].dropna()
        if len(fairness) > 5:
            print(f"  {party}s: Mean={fairness.mean():.2f} (n={len(fairness)})")

# ================== 7. CLIMATE CHANGE BELIEFS BY RACE & PARTY ==================
print("\n" + "="*80)
print("[7] CLIMATE CHANGE BELIEFS: Race × Party Interaction")
print("="*80)

# Q167: Is climate change caused by human activities?
print("\nBelief that climate change is caused by human activities:")

climate_crosstab = pd.crosstab([df['race_category'], df['party_binary']], 
                               df['Q167'], normalize='index') * 100

# Focus on "Caused mostly by human activities"
human_caused = climate_crosstab['Caused mostly by human activities'] if 'Caused mostly by human activities' in climate_crosstab.columns else None

if human_caused is not None:
    print("\n% Believing climate change is human-caused:")
    for (race, party), pct in human_caused.items():
        if race in ['White', 'Black/African American', 'Hispanic/Latino']:
            print(f"  {race} {party}s: {pct:.1f}%")

# ================== 8. SAVE DATASET ==================
print("\n" + "="*80)
print("[8] SAVING RACE × POLITICS DATASET")
print("="*80)

export_cols = ['ResponseId', 'party', 'party_binary', 'ideology', 'ideology_numeric',
               'race', 'race_category', 'trust_federal', 'trust_local',
               'racial_fairness_index', 'Q115_numeric', 'Q116_numeric', 'Q117_numeric', 'Q118_numeric',
               'Q167', 'Q62', 'Q63']

export_cols = [col for col in export_cols if col in df.columns]
df[export_cols].to_csv('race_politics_data.csv', index=False)
print(f"✓ Saved: race_politics_data.csv ({len(export_cols)} variables)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)