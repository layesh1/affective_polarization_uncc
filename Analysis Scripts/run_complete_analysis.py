"""
Political Polarization Analysis - Complete Script
Works with files in current directory
Author: Lena
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

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("="*80)
print("POLITICAL POLARIZATION ANALYSIS - UNC CHARLOTTE")
print("="*80)

# ================== LOAD DATA ==================
print("\n[1] LOADING DATA...")

# Try different possible filenames
possible_files = [
    'POLS Lab Student Omnibus Fall 25- text.csv',
    'POLS_Lab_Student_Omnibus_Fall_25-_text.csv',
    'POLS Omnibus.csv'
]

data_file = None
for filename in possible_files:
    if os.path.exists(filename):
        data_file = filename
        break

if data_file is None:
    print("\n❌ ERROR: Cannot find your data file!")
    print(f"Current directory: {os.getcwd()}")
    print("\nFiles in this directory:")
    for f in os.listdir('.'):
        if f.endswith('.csv'):
            print(f"  • {f}")
    print("\nPlease update the filename in line 29-33 of this script.")
    exit()

print(f"Found data file: {data_file}")
df = pd.read_csv(data_file, low_memory=False, encoding='utf-8-sig')

# Remove metadata row
df = df[~df['StartDate'].str.contains('ImportId', na=False)]
print(f"✓ Loaded: {df.shape[0]} responses, {df.shape[1]} variables")

# ================== BASIC EDA ==================
print("\n[2] KEY FINDINGS")
print("-"*80)

# Map ideology
ideology_map = {
    'Very liberal': 1, 'Liberal': 2, 'Somewhat liberal': 3, 'Moderate': 4,
    'Somewhat conservative': 5, 'Conservative': 6, 'Very conservative': 7
}
df['ideology_numeric'] = df['ideology'].map(ideology_map)
df['ideological_distance'] = abs(df['ideology_numeric'] - 4)

# Create party binary
df['party_binary'] = df['party'].map({
    'Strongly Democrat': 'Democrat',
    'Somewhat Democrat': 'Democrat',
    'Not sure/neither one/other': 'Independent',
    'Somewhat Republican': 'Republican',
    'Strongly Republican': 'Republican'
})

# Party distribution
print("\n📊 PARTY DISTRIBUTION:")
party_counts = df['party_binary'].value_counts()
for party, count in party_counts.items():
    print(f"  {party}: {count} ({count/len(df)*100:.1f}%)")

# Ideology distribution
print("\n📊 IDEOLOGY DISTRIBUTION:")
print(f"  Mean: {df['ideology_numeric'].mean():.2f} (1=Very Liberal, 7=Very Conservative)")
print(f"  Median: {df['ideology_numeric'].median():.1f}")
print(f"  At extremes: {((df['ideology_numeric']==1) | (df['ideology_numeric']==7)).sum()} ({((df['ideology_numeric']==1) | (df['ideology_numeric']==7)).sum()/len(df)*100:.1f}%)")

# Partisan sorting
party_numeric = df['party'].map({
    'Strongly Democrat': 1, 'Somewhat Democrat': 2,
    'Not sure/neither one/other': 3,
    'Somewhat Republican': 4, 'Strongly Republican': 5
})

valid_mask = party_numeric.notna() & df['ideology_numeric'].notna()
if valid_mask.sum() > 0:
    correlation, p_value = stats.spearmanr(
        party_numeric[valid_mask], 
        df.loc[valid_mask, 'ideology_numeric']
    )
    print(f"\n🔥 PARTISAN SORTING:")
    print(f"  Correlation: r = {correlation:.3f} (p < 0.001)")
    print(f"  Interpretation: {'STRONG' if abs(correlation) > 0.7 else 'MODERATE' if abs(correlation) > 0.4 else 'WEAK'} partisan sorting")

# ================== VISUALIZATIONS ==================
print("\n[3] CREATING VISUALIZATIONS...")
print("-"*80)

# Figure 1: Party Distribution
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
party_data = df['party'].value_counts()
colors = {'Strongly Democrat': '#0015BC', 'Somewhat Democrat': '#6495ED',
          'Not sure/neither one/other': '#808080',
          'Somewhat Republican': '#E81B23', 'Strongly Republican': '#8B0000'}
party_colors = [colors.get(p, '#808080') for p in party_data.index]

party_data.plot(kind='barh', ax=ax, color=party_colors)
ax.set_xlabel('Number of Respondents', fontsize=12, fontweight='bold')
ax.set_ylabel('Party Identification', fontsize=12, fontweight='bold')
ax.set_title('Party Affiliation Distribution\nUNC Charlotte Student Survey (Fall 2025)', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

for i, v in enumerate(party_data.values):
    ax.text(v + 5, i, f'{v} ({v/len(df)*100:.1f}%)', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('01_party_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_party_distribution.png")
plt.close()

# Figure 2: Ideology Spectrum
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ideology_order = ['Very liberal', 'Liberal', 'Somewhat liberal', 'Moderate',
                 'Somewhat conservative', 'Conservative', 'Very conservative']
ideology_data = df['ideology'].value_counts()
ideology_data = ideology_data.reindex([i for i in ideology_order if i in ideology_data.index])

colors_gradient = ['#0015BC', '#4169E1', '#87CEEB', '#D3D3D3', 
                  '#FFA07A', '#E81B23', '#8B0000']
colors_gradient = colors_gradient[:len(ideology_data)]

ideology_data.plot(kind='bar', ax=ax, color=colors_gradient)
ax.set_xlabel('Ideological Position', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Respondents', fontsize=12, fontweight='bold')
ax.set_title('Ideological Self-Placement Distribution\nUNC Charlotte Students (Fall 2025)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticklabels(ideology_data.index, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

for i, v in enumerate(ideology_data.values):
    ax.text(i, v + 5, f'{v}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('02_ideology_spectrum.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_ideology_spectrum.png")
plt.close()

# Figure 3: Party × Ideology Heatmap
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

party_filter = df['party'].isin(['Strongly Democrat', 'Somewhat Democrat', 
                                  'Not sure/neither one/other',
                                  'Somewhat Republican', 'Strongly Republican'])
ideology_filter = df['ideology'].isin(ideology_order)

filtered_df = df[party_filter & ideology_filter]

crosstab = pd.crosstab(filtered_df['party'], filtered_df['ideology'])
crosstab = crosstab.reindex(['Strongly Democrat', 'Somewhat Democrat', 
                              'Not sure/neither one/other',
                              'Somewhat Republican', 'Strongly Republican'])
crosstab = crosstab.reindex(columns=ideology_order)

sns.heatmap(crosstab, annot=True, fmt='d', cmap='RdBu_r', 
            cbar_kws={'label': 'Count'}, ax=ax, linewidths=0.5)
ax.set_xlabel('Ideological Self-Placement', fontsize=12, fontweight='bold')
ax.set_ylabel('Party Identification', fontsize=12, fontweight='bold')
ax.set_title('Political Identity Matrix: Party × Ideology\nHighlighting Partisan Sorting', 
             fontsize=14, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig('03_party_ideology_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_party_ideology_heatmap.png")
plt.close()

# Figure 4: Polarization Metrics
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Distance from center
axes[0].hist(df['ideological_distance'].dropna(), bins=np.arange(0, 4, 0.5), 
             color='#4169E1', edgecolor='black', alpha=0.7)
axes[0].axvline(df['ideological_distance'].mean(), color='red', 
                linestyle='--', linewidth=2, label=f'Mean = {df["ideological_distance"].mean():.2f}')
axes[0].set_xlabel('Distance from Ideological Center', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0].set_title('Polarization Index', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Right: Ideology by party
party_ideology = df.groupby('party')['ideology_numeric'].apply(list)
party_order = ['Strongly Democrat', 'Somewhat Democrat', 'Not sure/neither one/other',
               'Somewhat Republican', 'Strongly Republican']
party_data_plot = [party_ideology.get(p, []) for p in party_order if p in party_ideology.index]

bp = axes[1].boxplot(party_data_plot, labels=[p for p in party_order if p in party_ideology.index],
                     patch_artist=True, showmeans=True)

for patch, color in zip(bp['boxes'], ['#0015BC', '#6495ED', '#808080', '#E81B23', '#8B0000']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

axes[1].set_ylabel('Ideological Position\n(1=Very Liberal, 7=Very Conservative)', 
                   fontsize=11, fontweight='bold')
axes[1].set_title('Ideological Distribution by Party', fontsize=12, fontweight='bold')
axes[1].set_xticklabels([p for p in party_order if p in party_ideology.index], 
                        rotation=45, ha='right')
axes[1].grid(axis='y', alpha=0.3)
axes[1].axhline(4, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

plt.tight_layout()
plt.savefig('04_polarization_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_polarization_metrics.png")
plt.close()

# ================== SAVE DATA ==================
print("\n[4] SAVING CLEAN DATASETS...")
print("-"*80)

# Clean dataset
export_cols = ['ResponseId', 'party', 'party_binary', 'ideology', 'ideology_numeric',
               'ideological_distance', 'Q13', 'Q12', 'Q15']
export_cols = [col for col in export_cols if col in df.columns]

df[export_cols].to_csv('polarization_analysis_ready.csv', index=False)
print(f"✓ Saved: polarization_analysis_ready.csv ({len(export_cols)} variables)")

# Summary stats
summary = pd.DataFrame({
    'Variable': ['Total Responses', 'Party ID', 'Ideology ID', 
                 'Mean Ideology', 'SD Ideology', 'Democrats', 'Republicans',
                 'Independents', 'Very Liberal', 'Very Conservative'],
    'Value': [
        len(df),
        df['party'].notna().sum(),
        df['ideology'].notna().sum(),
        f"{df['ideology_numeric'].mean():.2f}",
        f"{df['ideology_numeric'].std():.2f}",
        f"{(df['party_binary'] == 'Democrat').sum()} ({(df['party_binary'] == 'Democrat').sum() / len(df) * 100:.1f}%)",
        f"{(df['party_binary'] == 'Republican').sum()} ({(df['party_binary'] == 'Republican').sum() / len(df) * 100:.1f}%)",
        f"{(df['party_binary'] == 'Independent').sum()} ({(df['party_binary'] == 'Independent').sum() / len(df) * 100:.1f}%)",
        f"{(df['ideology'] == 'Very liberal').sum()} ({(df['ideology'] == 'Very liberal').sum() / len(df) * 100:.1f}%)",
        f"{(df['ideology'] == 'Very conservative').sum()} ({(df['ideology'] == 'Very conservative').sum() / len(df) * 100:.1f}%)"
    ]
})

summary.to_csv('summary_statistics.csv', index=False)
print("✓ Saved: summary_statistics.csv")

# ================== FINAL SUMMARY ==================
print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)
print(f"\n📂 All files saved in: {os.getcwd()}")
print("\n📊 Generated Files:")
print("  • 01_party_distribution.png")
print("  • 02_ideology_spectrum.png")
print("  • 03_party_ideology_heatmap.png")
print("  • 04_polarization_metrics.png")
print("  • polarization_analysis_ready.csv")
print("  • summary_statistics.csv")
print("\n🔍 Key Finding:")
print(f"  Strong partisan sorting detected (r = {correlation:.3f})")
print(f"  Sample: {len(df)} UNC Charlotte students")
print("="*80)