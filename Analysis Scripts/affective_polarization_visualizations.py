"""
AFFECTIVE POLARIZATION VISUALIZATIONS
Advanced charts showing emotional polarization, social distance, and out-party hostility
Author: Lena
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
sns.set_palette("husl")

# Load data
data_file = 'POLS Lab Student Omnibus Fall 25- text.csv'
if not os.path.exists(data_file):
    data_file = 'POLS_Lab_Student_Omnibus_Fall_25-_text.csv'

df = pd.read_csv(data_file, low_memory=False, encoding='utf-8-sig')
df = df[~df['StartDate'].str.contains('ImportId', na=False)]

# Setup party binary
df['party_binary'] = df['party'].map({
    'Strongly Democrat': 'Democrat',
    'Somewhat Democrat': 'Democrat',
    'Not sure/neither one/other': 'Independent',
    'Somewhat Republican': 'Republican',
    'Strongly Republican': 'Republican'
})

# Prepare affective polarization measures
moral_map = {'None at all': 1, 'A little': 2, 'A moderate amount': 3, 'A lot': 4, 'A great deal': 5}

# Moral identity
df['moral1'] = df['moral1D'].fillna(df['moral1R']).map(moral_map)
df['moral2'] = df['moral2D'].fillna(df['moral2R']).map(moral_map)
df['moral3'] = df['moral3D'].fillna(df['moral3R']).map(moral_map)
df['moral_identity_index'] = df[['moral1', 'moral2', 'moral3']].mean(axis=1)

# Othering
df['other1'] = df['other1D'].fillna(df['other1R']).map(moral_map)
df['other2'] = df['other2D'].fillna(df['other2R']).map(moral_map)
df['other3'] = df['other3D'].fillna(df['other3R']).map(moral_map)
df['othering_index'] = df[['other1', 'other2', 'other3']].mean(axis=1)

# Feeling thermometers
df['FT_Republicans'] = pd.to_numeric(df['Q148'], errors='coerce')
df['FT_Democrats'] = pd.to_numeric(df['Q149'], errors='coerce')

df['FT_inparty'] = np.where(df['party_binary']=='Democrat', df['FT_Democrats'],
                    np.where(df['party_binary']=='Republican', df['FT_Republicans'], np.nan))
df['FT_outparty'] = np.where(df['party_binary']=='Democrat', df['FT_Republicans'],
                     np.where(df['party_binary']=='Republican', df['FT_Democrats'], np.nan))
df['affective_polarization'] = df['FT_inparty'] - df['FT_outparty']

# Out-party distrust
trust_map = {'Strongly agree': 7, 'Agree': 6, 'Somewhat agree': 5,
             'Neither agree nor disagree': 4, 'Somewhat disagree': 3, 'Disagree': 2, 'Strongly disagree': 1}
df['outparty_distrust'] = df['Q119'].map(trust_map)

print("Creating affective polarization visualizations...")

# ================== FIGURE 1: FEELING THERMOMETERS ==================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel A: In-party vs Out-party warmth
for party, color in [('Democrat', '#0015BC'), ('Republican', '#E81B23')]:
    party_df = df[df['party_binary'] == party]
    
    inparty = party_df['FT_inparty'].dropna()
    outparty = party_df['FT_outparty'].dropna()
    
    positions = [0, 1] if party == 'Democrat' else [2.5, 3.5]
    axes[0].boxplot([inparty, outparty], positions=positions, widths=0.6,
                    patch_artist=True, 
                    boxprops=dict(facecolor=color, alpha=0.6),
                    medianprops=dict(color='black', linewidth=2))

axes[0].set_xticks([0.5, 3])
axes[0].set_xticklabels(['Democrats', 'Republicans'])
axes[0].set_ylabel('Feeling Thermometer (0-100)', fontweight='bold')
axes[0].set_title('In-Party vs Out-Party Warmth\n(Darker = In-Party, Lighter = Out-Party)', 
                  fontweight='bold', fontsize=12)
axes[0].axhline(50, color='gray', linestyle='--', alpha=0.5, label='Neutral')
axes[0].grid(axis='y', alpha=0.3)
axes[0].legend()

# Panel B: Affective polarization distribution
dem_pol = df[df['party_binary']=='Democrat']['affective_polarization'].dropna()
rep_pol = df[df['party_binary']=='Republican']['affective_polarization'].dropna()

axes[1].hist(dem_pol, bins=20, alpha=0.6, color='#0015BC', label=f'Democrats (M={dem_pol.mean():.1f})')
axes[1].hist(rep_pol, bins=20, alpha=0.6, color='#E81B23', label=f'Republicans (M={rep_pol.mean():.1f})')
axes[1].axvline(0, color='black', linestyle='--', linewidth=2, label='No polarization')
axes[1].set_xlabel('Affective Polarization\n(In-Party Warmth - Out-Party Warmth)', fontweight='bold')
axes[1].set_ylabel('Frequency', fontweight='bold')
axes[1].set_title('Distribution of Affective Polarization Scores', fontweight='bold', fontsize=12)
axes[1].legend()
axes[1].grid(alpha=0.3)

# Panel C: Scatter plot
party_colors = {'Democrat': '#0015BC', 'Republican': '#E81B23'}
for party in ['Democrat', 'Republican']:
    party_df = df[df['party_binary'] == party]
    axes[2].scatter(party_df['FT_outparty'], party_df['FT_inparty'], 
                    alpha=0.4, s=30, color=party_colors[party], label=party)

axes[2].plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Equal warmth line')
axes[2].set_xlabel('Out-Party Warmth', fontweight='bold')
axes[2].set_ylabel('In-Party Warmth', fontweight='bold')
axes[2].set_title('In-Party vs Out-Party Warmth Scatter', fontweight='bold', fontsize=12)
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('07_feeling_thermometers.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 07_feeling_thermometers.png")
plt.close()

# ================== FIGURE 2: MORAL IDENTITY & OTHERING ==================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: Moral identity by party
moral_data = []
party_labels = []
colors_moral = []

for party, color in [('Democrat', '#0015BC'), ('Republican', '#E81B23')]:
    moral_scores = df[df['party_binary']==party]['moral_identity_index'].dropna()
    if len(moral_scores) > 0:
        moral_data.append(moral_scores)
        party_labels.append(f'{party}\n(n={len(moral_scores)})')
        colors_moral.append(color)

bp1 = axes[0,0].boxplot(moral_data, labels=party_labels, patch_artist=True, showmeans=True)
for patch, color in zip(bp1['boxes'], colors_moral):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

axes[0,0].set_ylabel('Moral Identity Index\n(1=None at all, 5=A great deal)', fontweight='bold')
axes[0,0].set_title('Moral Identity: Party as Moral Foundation', fontweight='bold', fontsize=12)
axes[0,0].grid(axis='y', alpha=0.3)

# Panel B: Othering index by party
other_data = []
for party, color in [('Democrat', '#0015BC'), ('Republican', '#E81B23')]:
    other_scores = df[df['party_binary']==party]['othering_index'].dropna()
    if len(other_scores) > 0:
        other_data.append(other_scores)

bp2 = axes[0,1].boxplot(other_data, labels=party_labels, patch_artist=True, showmeans=True)
for patch, color in zip(bp2['boxes'], colors_moral):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

axes[0,1].set_ylabel('Othering Index\n(1=None at all, 5=A great deal)', fontweight='bold')
axes[0,1].set_title('Othering: Out-Party Perceived as Different', fontweight='bold', fontsize=12)
axes[0,1].grid(axis='y', alpha=0.3)

# Panel C: Out-party distrust
distrust_data = []
for party in ['Democrat', 'Republican']:
    distrust = df[df['party_binary']==party]['outparty_distrust'].dropna()
    if len(distrust) > 0:
        distrust_data.append(distrust)

bp3 = axes[1,0].boxplot(distrust_data, labels=party_labels, patch_artist=True, showmeans=True)
for patch, color in zip(bp3['boxes'], colors_moral):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

axes[1,0].set_ylabel('Out-Party Distrust\n(1=Strongly disagree, 7=Strongly agree)', fontweight='bold')
axes[1,0].set_title('"Out-Party Cannot Be Trusted in Government"', fontweight='bold', fontsize=12)
axes[1,0].grid(axis='y', alpha=0.3)

# Panel D: Correlation between measures
affective_df = df[df['party_binary'].isin(['Democrat', 'Republican'])][
    ['moral_identity_index', 'othering_index', 'affective_polarization', 'outparty_distrust']
].dropna()

if len(affective_df) > 20:
    corr = affective_df.corr()
    im = axes[1,1].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    
    axes[1,1].set_xticks(range(len(corr.columns)))
    axes[1,1].set_yticks(range(len(corr.columns)))
    axes[1,1].set_xticklabels(['Moral\nIdentity', 'Othering', 'Affective\nPolarization', 'Out-Party\nDistrust'], 
                              rotation=45, ha='right')
    axes[1,1].set_yticklabels(['Moral Identity', 'Othering', 'Affective Polarization', 'Out-Party Distrust'])
    axes[1,1].set_title('Correlations Between Polarization Measures', fontweight='bold', fontsize=12)
    
    # Add correlation values
    for i in range(len(corr)):
        for j in range(len(corr)):
            text = axes[1,1].text(j, i, f'{corr.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=axes[1,1], label='Correlation')

plt.tight_layout()
plt.savefig('08_moral_othering_indices.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 08_moral_othering_indices.png")
plt.close()

# ================== FIGURE 3: ASYMMETRIC POLARIZATION ==================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Mean scores comparison
measures = ['Moral\nIdentity', 'Othering', 'Affective\nPolarization', 'Out-Party\nDistrust']
dem_means = [
    df[df['party_binary']=='Democrat']['moral_identity_index'].mean(),
    df[df['party_binary']=='Democrat']['othering_index'].mean(),
    df[df['party_binary']=='Democrat']['affective_polarization'].mean() / 20,  # Scale to 1-5
    df[df['party_binary']=='Democrat']['outparty_distrust'].mean()
]
rep_means = [
    df[df['party_binary']=='Republican']['moral_identity_index'].mean(),
    df[df['party_binary']=='Republican']['othering_index'].mean(),
    df[df['party_binary']=='Republican']['affective_polarization'].mean() / 20,  # Scale to 1-5
    df[df['party_binary']=='Republican']['outparty_distrust'].mean()
]

x = np.arange(len(measures))
width = 0.35

axes[0].bar(x - width/2, dem_means, width, label='Democrats', color='#0015BC', alpha=0.7)
axes[0].bar(x + width/2, rep_means, width, label='Republicans', color='#E81B23', alpha=0.7)

axes[0].set_ylabel('Mean Score', fontweight='bold')
axes[0].set_title('Affective Polarization: Democrats vs Republicans', fontweight='bold', fontsize=12)
axes[0].set_xticks(x)
axes[0].set_xticklabels(measures)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Panel B: Distribution of affective polarization scores
axes[1].hist(dem_pol, bins=30, alpha=0.5, color='#0015BC', label='Democrats', density=True)
axes[1].hist(rep_pol, bins=30, alpha=0.5, color='#E81B23', label='Republicans', density=True)
axes[1].axvline(dem_pol.mean(), color='#0015BC', linestyle='--', linewidth=2)
axes[1].axvline(rep_pol.mean(), color='#E81B23', linestyle='--', linewidth=2)
axes[1].set_xlabel('Affective Polarization Score', fontweight='bold')
axes[1].set_ylabel('Density', fontweight='bold')
axes[1].set_title('Distribution of Affective Polarization', fontweight='bold', fontsize=12)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('09_asymmetric_polarization.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 09_asymmetric_polarization.png")
plt.close()

print("\n" + "="*80)
print("All visualizations complete!")
print("  • 07_feeling_thermometers.png")
print("  • 08_moral_othering_indices.png")
print("  • 09_asymmetric_polarization.png")
print("="*80)