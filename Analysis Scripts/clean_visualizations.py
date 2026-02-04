"""
CLEAN VISUALIZATIONS FOR POLARIZATION STUDY
Simple, interpretable charts with clear takeaways
Author: Lena
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

# Clean, minimal style
sns.set_style("white")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

print("Creating clean visualizations...")

# Load data
data_file = 'POLS Lab Student Omnibus Fall 25- text.csv'
if not os.path.exists(data_file):
    data_file = 'POLS_Lab_Student_Omnibus_Fall_25-_text.csv'

df = pd.read_csv(data_file, low_memory=False, encoding='utf-8-sig')
df = df[~df['StartDate'].str.contains('ImportId', na=False)]
df = df[~df['party'].str.contains('Many of the questions', na=False)]

# Setup variables
df['party_binary'] = df['party'].map({
    'Strongly Democrat': 'Democrat', 'Somewhat Democrat': 'Democrat',
    'Not sure/neither one/other': 'Independent',
    'Somewhat Republican': 'Republican', 'Strongly Republican': 'Republican'
})

# Feeling thermometers
df['FT_Republicans'] = pd.to_numeric(df['Q148'], errors='coerce')
df['FT_Democrats'] = pd.to_numeric(df['Q149'], errors='coerce')
df['FT_inparty'] = np.where(df['party_binary']=='Democrat', df['FT_Democrats'],
                    np.where(df['party_binary']=='Republican', df['FT_Republicans'], np.nan))
df['FT_outparty'] = np.where(df['party_binary']=='Democrat', df['FT_Republicans'],
                     np.where(df['party_binary']=='Republican', df['FT_Democrats'], np.nan))
df['affective_polarization'] = df['FT_inparty'] - df['FT_outparty']

# Moral identity
moral_map = {'None at all': 1, 'A little': 2, 'A moderate amount': 3, 'A lot': 4, 'A great deal': 5}
df['moral1'] = df['moral1D'].fillna(df['moral1R']).map(moral_map)
df['moral2'] = df['moral2D'].fillna(df['moral2R']).map(moral_map)
df['moral3'] = df['moral3D'].fillna(df['moral3R']).map(moral_map)
df['moral_identity_index'] = df[['moral1', 'moral2', 'moral3']].mean(axis=1)

# ================== FIGURE 1: SIMPLE FEELING THERMOMETER COMPARISON ==================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

categories = ['Democrats\nrating\nDemocrats', 'Democrats\nrating\nRepublicans', 
              'Republicans\nrating\nRepublicans', 'Republicans\nrating\nDemocrats']

# Calculate means
dem_rate_dem = df[df['party_binary']=='Democrat']['FT_Democrats'].mean()
dem_rate_rep = df[df['party_binary']=='Democrat']['FT_Republicans'].mean()
rep_rate_rep = df[df['party_binary']=='Republican']['FT_Republicans'].mean()
rep_rate_dem = df[df['party_binary']=='Republican']['FT_Democrats'].mean()

values = [dem_rate_dem, dem_rate_rep, rep_rate_rep, rep_rate_dem]
colors = ['#0015BC', '#87CEEB', '#E81B23', '#FFA07A']

bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

# Add horizontal line at 50 (neutral)
ax.axhline(50, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Neutral (50)')

ax.set_ylabel('Warmth Rating (0-100)', fontsize=13, fontweight='bold')
ax.set_title('How Warm Do Students Feel Toward Each Party?\nHigher = More Positive Feelings', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(0, 100)
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add interpretation box
textstr = f'KEY FINDING:\n• Democrats: {dem_rate_dem-dem_rate_rep:.1f} point gap (own party warmer)\n• Republicans: {rep_rate_rep-rep_rate_dem:.1f} point gap (own party warmer)\n• Democrats show TWICE the polarization'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig('VIZ_01_feeling_thermometers_simple.png', dpi=300, bbox_inches='tight')
print("✓ Saved: VIZ_01_feeling_thermometers_simple.png")
plt.close()

# ================== FIGURE 2: POLARIZATION GAP ==================
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

dem_gap = df[df['party_binary']=='Democrat']['affective_polarization'].dropna()
rep_gap = df[df['party_binary']=='Republican']['affective_polarization'].dropna()

data_to_plot = [dem_gap, rep_gap]
labels = [f'Democrats\n(n={len(dem_gap)})', f'Republicans\n(n={len(rep_gap)})']

bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.6,
                showmeans=True, meanline=True,
                boxprops=dict(linewidth=2),
                medianprops=dict(color='black', linewidth=3),
                meanprops=dict(color='red', linewidth=2, linestyle='--'))

# Color boxes
bp['boxes'][0].set_facecolor('#0015BC')
bp['boxes'][0].set_alpha(0.6)
bp['boxes'][1].set_facecolor('#E81B23')
bp['boxes'][1].set_alpha(0.6)

ax.axhline(0, color='black', linestyle='-', linewidth=2, alpha=0.3, label='No polarization')
ax.set_ylabel('Affective Polarization Score\n(Positive = Like own party more)', 
              fontsize=12, fontweight='bold')
ax.set_title('The Polarization Gap: How Much More Do You Like Your Own Party?', 
             fontsize=13, fontweight='bold', pad=20)
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)

# Add mean annotations
ax.text(1, dem_gap.mean() + 3, f'Mean: {dem_gap.mean():.1f}', 
        ha='center', fontweight='bold', fontsize=11, color='#0015BC')
ax.text(2, rep_gap.mean() + 3, f'Mean: {rep_gap.mean():.1f}', 
        ha='center', fontweight='bold', fontsize=11, color='#E81B23')

plt.tight_layout()
plt.savefig('VIZ_02_polarization_gap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: VIZ_02_polarization_gap.png")
plt.close()

# ================== FIGURE 3: MORAL IDENTITY ==================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Individual moral items
moral_labels = ['Party connected to\ncore moral beliefs', 
                'Party reflects beliefs\nabout right vs wrong',
                'Party rooted in\nmoral principles']

dem_moral = [df[df['party_binary']=='Democrat'][var].mean() for var in ['moral1', 'moral2', 'moral3']]
rep_moral = [df[df['party_binary']=='Republican'][var].mean() for var in ['moral1', 'moral2', 'moral3']]

x = np.arange(len(moral_labels))
width = 0.35

bars1 = ax.bar(x - width/2, dem_moral, width, label='Democrats', 
               color='#0015BC', alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, rep_moral, width, label='Republicans', 
               color='#E81B23', alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

ax.set_ylabel('Agreement Level\n(1=None at all, 5=A great deal)', fontsize=12, fontweight='bold')
ax.set_title('Is Your Party Identity About Morality?\nBoth Groups Say YES', 
             fontsize=13, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(moral_labels)
ax.legend(fontsize=11)
ax.set_ylim(0, 5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('VIZ_03_moral_identity.png', dpi=300, bbox_inches='tight')
print("✓ Saved: VIZ_03_moral_identity.png")
plt.close()

# ================== FIGURE 4: RACE × PARTY ==================
# Categorize race
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
    else:
        return 'Other'

df['race_category'] = df['Q62'].apply(categorize_race)

# Create crosstab
race_party_data = pd.crosstab(df['race_category'], df['party_binary'], normalize='index') * 100
race_party_data = race_party_data.loc[['White', 'Black/African American', 'Hispanic/Latino', 'Asian']]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

race_party_data.plot(kind='bar', ax=ax, color=['#0015BC', '#808080', '#E81B23'], 
                     alpha=0.7, edgecolor='black', linewidth=1.5, width=0.8)

ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Race/Ethnicity', fontsize=12, fontweight='bold')
ax.set_title('Party Affiliation by Race\nDemocrats Dominate Across All Groups', 
             fontsize=13, fontweight='bold', pad=20)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.legend(title='Party', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)

# Add percentage labels
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f%%', fontsize=9)

plt.tight_layout()
plt.savefig('VIZ_04_race_party.png', dpi=300, bbox_inches='tight')
print("✓ Saved: VIZ_04_race_party.png")
plt.close()

# ================== FIGURE 5: TRUST IN GOVERNMENT BY PARTY ==================
trust_map = {'Strongly agree': 7, 'Agree': 6, 'Somewhat agree': 5,
             'Neither agree nor disagree': 4, 'Somewhat disagree': 3, 
             'Disagree': 2, 'Strongly disagree': 1}

df['trust_federal'] = df['Q110'].map(trust_map)
df['trust_local'] = df['Q112'].map(trust_map)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

trust_categories = ['Trust Federal\nGovernment', 'Trust Local/State\nGovernment']
dem_trust = [df[df['party_binary']=='Democrat']['trust_federal'].mean(),
             df[df['party_binary']=='Democrat']['trust_local'].mean()]
rep_trust = [df[df['party_binary']=='Republican']['trust_federal'].mean(),
             df[df['party_binary']=='Republican']['trust_local'].mean()]

x = np.arange(len(trust_categories))
width = 0.35

bars1 = ax.bar(x - width/2, dem_trust, width, label='Democrats', 
               color='#0015BC', alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, rep_trust, width, label='Republicans', 
               color='#E81B23', alpha=0.7, edgecolor='black', linewidth=1.5)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

ax.axhline(4, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Neutral (4)')
ax.set_ylabel('Trust Level\n(1=Strongly disagree, 7=Strongly agree)', fontsize=12, fontweight='bold')
ax.set_title('Trust in Government: Both Parties Are Skeptical', 
             fontsize=13, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(trust_categories)
ax.legend(fontsize=11)
ax.set_ylim(0, 7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('VIZ_05_trust_government.png', dpi=300, bbox_inches='tight')
print("✓ Saved: VIZ_05_trust_government.png")
plt.close()

print("\n" + "="*80)
print("✅ ALL CLEAN VISUALIZATIONS COMPLETE!")
print("="*80)
print("\nGenerated 5 publication-ready figures:")
print("  1. VIZ_01_feeling_thermometers_simple.png - Easy bar chart")
print("  2. VIZ_02_polarization_gap.png - Boxplot comparison")
print("  3. VIZ_03_moral_identity.png - Moral foundations by party")
print("  4. VIZ_04_race_party.png - Party affiliation by race")
print("  5. VIZ_05_trust_government.png - Government trust levels")
print("="*80)