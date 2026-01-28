"""
RACE, GENDER, AND AFFECTIVE POLARIZATION
Cross-correlational analysis examining how demographics predict polarization
Author: Lena

Research Questions:
1. Does affective polarization vary by race/ethnicity?
2. Does affective polarization vary by gender?
3. Are there Race × Party or Gender × Party interactions?
4. Which demographic groups show the highest polarization?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, f_oneway, pearsonr
import warnings
import os
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

print("="*90)
print(" " * 20 + "DEMOGRAPHICS × AFFECTIVE POLARIZATION")
print("="*90)

# ================== LOAD AND PREPARE DATA ==================
data_file = 'POLS Lab Student Omnibus Fall 25- text.csv'
if not os.path.exists(data_file):
    data_file = 'POLS_Lab_Student_Omnibus_Fall_25-_text.csv'

df = pd.read_csv(data_file, low_memory=False, encoding='utf-8-sig')
df = df[~df['StartDate'].str.contains('ImportId', na=False)]
df = df[~df['party'].str.contains('Many of the questions', na=False)]

# Party binary
df['party_binary'] = df['party'].map({
    'Strongly Democrat': 'Democrat', 'Somewhat Democrat': 'Democrat',
    'Not sure/neither one/other': 'Independent',
    'Somewhat Republican': 'Republican', 'Strongly Republican': 'Republican'
})

# Affective polarization measures
df['FT_Republicans'] = pd.to_numeric(df['Q148'], errors='coerce')
df['FT_Democrats'] = pd.to_numeric(df['Q149'], errors='coerce')

df['FT_inparty'] = np.where(df['party_binary']=='Democrat', df['FT_Democrats'],
                    np.where(df['party_binary']=='Republican', df['FT_Republicans'], np.nan))
df['FT_outparty'] = np.where(df['party_binary']=='Democrat', df['FT_Republicans'],
                     np.where(df['party_binary']=='Republican', df['FT_Democrats'], np.nan))
df['affective_polarization'] = df['FT_inparty'] - df['FT_outparty']

# Moral identity & othering
moral_map = {'None at all': 1, 'A little': 2, 'A moderate amount': 3, 'A lot': 4, 'A great deal': 5}
df['moral1'] = df['moral1D'].fillna(df['moral1R']).map(moral_map)
df['moral2'] = df['moral2D'].fillna(df['moral2R']).map(moral_map)
df['moral3'] = df['moral3D'].fillna(df['moral3R']).map(moral_map)
df['moral_identity_index'] = df[['moral1', 'moral2', 'moral3']].mean(axis=1)

df['other1'] = df['other1D'].fillna(df['other1R']).map(moral_map)
df['other2'] = df['other2D'].fillna(df['other2R']).map(moral_map)
df['other3'] = df['other3D'].fillna(df['other3R']).map(moral_map)
df['othering_index'] = df[['other1', 'other2', 'other3']].mean(axis=1)

# Race categories
def categorize_race(race_str):
    if pd.isna(race_str):
        return 'Missing'
    race_str = str(race_str).lower()
    if 'white' in race_str and ',' not in race_str:
        return 'White'
    elif 'black' in race_str or 'african american' in race_str:
        return 'Black/African American'
    elif 'hispanic' in race_str or 'latino' in race_str:
        return 'Hispanic/Latino'
    elif 'asian' in race_str:
        return 'Asian'
    elif ',' in race_str:
        return 'Multiracial'
    else:
        return 'Other'

df['race_category'] = df['Q62'].apply(categorize_race)

# Gender (Q63)
df['gender'] = df['Q63']

print(f"\nSample: {len(df)} total responses")
print(f"With affective polarization data: {df['affective_polarization'].notna().sum()}")
print(f"With race data: {df['race_category'].notna().sum()}")
print(f"With gender data: {df['gender'].notna().sum()}")

# ================== ANALYSIS 1: AFFECTIVE POLARIZATION BY RACE ==================
print("\n" + "="*90)
print("[1] AFFECTIVE POLARIZATION BY RACE/ETHNICITY")
print("="*90)

race_polarization = df[df['party_binary'].isin(['Democrat', 'Republican'])].groupby('race_category')['affective_polarization'].agg(['mean', 'std', 'count']).round(2)

print("\nMean Affective Polarization by Race:")
print(race_polarization)

# Focus on main groups
main_races = ['White', 'Black/African American', 'Hispanic/Latino', 'Asian']
race_groups = []
race_labels = []

for race in main_races:
    race_data = df[(df['race_category']==race) & (df['party_binary'].isin(['Democrat', 'Republican']))]['affective_polarization'].dropna()
    if len(race_data) > 10:
        race_groups.append(race_data)
        race_labels.append(f"{race}\n(n={len(race_data)})")
        print(f"\n{race}:")
        print(f"  Mean: {race_data.mean():.2f}")
        print(f"  SD: {race_data.std():.2f}")
        print(f"  Median: {race_data.median():.2f}")
        print(f"  Range: {race_data.min():.0f} to {race_data.max():.0f}")

# ANOVA test
if len(race_groups) >= 2:
    f_stat, p_val = f_oneway(*race_groups)
    print(f"\n📈 ONE-WAY ANOVA:")
    print(f"  F({len(race_groups)-1}, {sum([len(g) for g in race_groups])-len(race_groups)}) = {f_stat:.3f}, p = {p_val:.4f}")
    print(f"  Conclusion: {'SIGNIFICANT' if p_val < 0.05 else 'NO SIGNIFICANT'} differences in polarization across races")
    
    if p_val < 0.05:
        print(f"\n  Post-hoc observation:")
        means = [(race_labels[i].split('\\n')[0], race_groups[i].mean()) for i in range(len(race_groups))]
        means.sort(key=lambda x: x[1], reverse=True)
        print(f"    Highest: {means[0][0]} ({means[0][1]:.2f})")
        print(f"    Lowest: {means[-1][0]} ({means[-1][1]:.2f})")

# ================== ANALYSIS 2: AFFECTIVE POLARIZATION BY GENDER ==================
print("\n" + "="*90)
print("[2] AFFECTIVE POLARIZATION BY GENDER")
print("="*90)

gender_polarization = df[df['party_binary'].isin(['Democrat', 'Republican'])].groupby('gender')['affective_polarization'].agg(['mean', 'std', 'count']).round(2)

print("\nMean Affective Polarization by Gender:")
print(gender_polarization)

# T-test for Men vs Women
men_pol = df[(df['gender']=='Man') & (df['party_binary'].isin(['Democrat', 'Republican']))]['affective_polarization'].dropna()
women_pol = df[(df['gender']=='Woman') & (df['party_binary'].isin(['Democrat', 'Republican']))]['affective_polarization'].dropna()

if len(men_pol) > 10 and len(women_pol) > 10:
    print(f"\nMen:")
    print(f"  Mean: {men_pol.mean():.2f}")
    print(f"  SD: {men_pol.std():.2f}")
    print(f"  n = {len(men_pol)}")
    
    print(f"\nWomen:")
    print(f"  Mean: {women_pol.mean():.2f}")
    print(f"  SD: {women_pol.std():.2f}")
    print(f"  n = {len(women_pol)}")
    
    t_stat, p_val = ttest_ind(men_pol, women_pol)
    cohen_d = (men_pol.mean() - women_pol.mean()) / np.sqrt((men_pol.std()**2 + women_pol.std()**2) / 2)
    
    print(f"\n📈 INDEPENDENT SAMPLES T-TEST:")
    print(f"  t({len(men_pol)+len(women_pol)-2}) = {t_stat:.3f}, p = {p_val:.4f}")
    print(f"  Cohen's d = {cohen_d:.3f} ({'Small' if abs(cohen_d) < 0.5 else 'Medium' if abs(cohen_d) < 0.8 else 'Large'} effect)")
    print(f"  Conclusion: {'SIGNIFICANT' if p_val < 0.05 else 'NO SIGNIFICANT'} gender difference")

# ================== ANALYSIS 3: RACE × PARTY INTERACTION ==================
print("\n" + "="*90)
print("[3] RACE × PARTY INTERACTION ON POLARIZATION")
print("="*90)

print("\nMean Affective Polarization by Race and Party:")
print("(Does polarization differ by race within each party?)")

for race in main_races:
    race_df = df[df['race_category']==race]
    
    dem_pol = race_df[race_df['party_binary']=='Democrat']['affective_polarization'].dropna()
    rep_pol = race_df[race_df['party_binary']=='Republican']['affective_polarization'].dropna()
    
    if len(dem_pol) >= 5 and len(rep_pol) >= 5:
        print(f"\n{race}:")
        print(f"  Democrats: M={dem_pol.mean():.2f}, SD={dem_pol.std():.2f}, n={len(dem_pol)}")
        print(f"  Republicans: M={rep_pol.mean():.2f}, SD={rep_pol.std():.2f}, n={len(rep_pol)}")
        print(f"  Gap: {abs(dem_pol.mean() - rep_pol.mean()):.2f} points")
        
        if len(dem_pol) >= 10 and len(rep_pol) >= 10:
            t_stat, p_val = ttest_ind(dem_pol, rep_pol)
            print(f"  Within-race difference: {'Significant' if p_val < 0.05 else 'Not significant'} (p={p_val:.3f})")

# ================== ANALYSIS 4: GENDER × PARTY INTERACTION ==================
print("\n" + "="*90)
print("[4] GENDER × PARTY INTERACTION ON POLARIZATION")
print("="*90)

print("\nMean Affective Polarization by Gender and Party:")

for gender in ['Man', 'Woman']:
    gender_df = df[df['gender']==gender]
    
    dem_pol = gender_df[gender_df['party_binary']=='Democrat']['affective_polarization'].dropna()
    rep_pol = gender_df[gender_df['party_binary']=='Republican']['affective_polarization'].dropna()
    
    if len(dem_pol) >= 10 and len(rep_pol) >= 10:
        print(f"\n{gender}:")
        print(f"  Democrats: M={dem_pol.mean():.2f}, SD={dem_pol.std():.2f}, n={len(dem_pol)}")
        print(f"  Republicans: M={rep_pol.mean():.2f}, SD={rep_pol.std():.2f}, n={len(rep_pol)}")
        print(f"  Within-gender gap: {abs(dem_pol.mean() - rep_pol.mean()):.2f} points")
        
        t_stat, p_val = ttest_ind(dem_pol, rep_pol)
        print(f"  Difference: {'Significant' if p_val < 0.05 else 'Not significant'} (p={p_val:.3f})")

# ================== ANALYSIS 5: CORRELATIONS WITH OTHER POLARIZATION MEASURES ==================
print("\n" + "="*90)
print("[5] DO RACE/GENDER PREDICT OTHER POLARIZATION DIMENSIONS?")
print("="*90)

# Create numeric codes for categorical variables
df['race_numeric'] = df['race_category'].map({
    'White': 1, 'Black/African American': 2, 'Hispanic/Latino': 3, 
    'Asian': 4, 'Multiracial': 5, 'Other': 6
})

df['gender_numeric'] = df['gender'].map({'Man': 1, 'Woman': 2, 'Non-binary / third gender': 3})

# Correlations
polarization_vars = ['affective_polarization', 'moral_identity_index', 'othering_index']
demographic_vars = ['race_numeric', 'gender_numeric']

print("\nCorrelations (Spearman's rho):")
print("\n" + " "*20 + "Affective Pol.   Moral Identity   Othering")

for dem_var, dem_label in [('race_numeric', 'Race'), ('gender_numeric', 'Gender')]:
    correlations = []
    for pol_var in polarization_vars:
        valid_df = df[df[dem_var].notna() & df[pol_var].notna()]
        if len(valid_df) > 20:
            r, p = stats.spearmanr(valid_df[dem_var], valid_df[pol_var])
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            correlations.append(f"{r:>6.3f} {sig:<3}")
        else:
            correlations.append("   N/A    ")
    
    print(f"{dem_label:<20}" + "    ".join(correlations))

# ================== VISUALIZATIONS ==================
print("\n" + "="*90)
print("[6] CREATING VISUALIZATIONS")
print("="*90)

# Figure 1: Polarization by Race
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel A: Boxplot by race
if len(race_groups) >= 2:
    bp = axes[0].boxplot(race_groups, labels=race_labels, patch_artist=True, 
                         showmeans=True, widths=0.6)
    
    colors = ['#E8F4F8', '#B8E0F0', '#88CCE8', '#5EBBE0']
    for patch, color in zip(bp['boxes'], colors[:len(race_groups)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    axes[0].set_ylabel('Affective Polarization Score', fontsize=12, fontweight='bold')
    axes[0].set_title('Affective Polarization by Race/Ethnicity', fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

# Panel B: Boxplot by gender
if len(men_pol) > 10 and len(women_pol) > 10:
    bp2 = axes[1].boxplot([men_pol, women_pol], 
                          labels=[f'Men\n(n={len(men_pol)})', f'Women\n(n={len(women_pol)})'],
                          patch_artist=True, showmeans=True, widths=0.6)
    
    bp2['boxes'][0].set_facecolor('#6495ED')
    bp2['boxes'][0].set_alpha(0.7)
    bp2['boxes'][1].set_facecolor('#FF69B4')
    bp2['boxes'][1].set_alpha(0.7)
    
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    axes[1].set_ylabel('Affective Polarization Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Affective Polarization by Gender', fontsize=13, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('VIZ_06_demographics_polarization.png', dpi=300, bbox_inches='tight')
print("✓ Saved: VIZ_06_demographics_polarization.png")
plt.close()

# Figure 2: Race × Party Heatmap
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Create pivot table
heatmap_data = df[df['party_binary'].isin(['Democrat', 'Republican'])].pivot_table(
    values='affective_polarization',
    index='race_category',
    columns='party_binary',
    aggfunc='mean'
)

# Filter to main races
heatmap_data = heatmap_data.loc[main_races]

sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlBu_r', center=30,
            cbar_kws={'label': 'Affective Polarization'}, linewidths=2, ax=ax)

ax.set_xlabel('Party', fontsize=12, fontweight='bold')
ax.set_ylabel('Race/Ethnicity', fontsize=12, fontweight='bold')
ax.set_title('Affective Polarization: Race × Party Interaction\nHigher Values = More Polarized', 
             fontsize=13, fontweight='bold', pad=20)
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig('VIZ_07_race_party_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: VIZ_07_race_party_heatmap.png")
plt.close()

# ================== SAVE DATASET ==================
print("\n" + "="*90)
print("[7] SAVING CROSS-CORRELATION DATASET")
print("="*90)

export_cols = ['ResponseId', 'party', 'party_binary', 
               'affective_polarization', 'moral_identity_index', 'othering_index',
               'race_category', 'gender', 
               'FT_Democrats', 'FT_Republicans', 'FT_inparty', 'FT_outparty']

export_cols = [col for col in export_cols if col in df.columns]
df[export_cols].to_csv('demographics_polarization_data.csv', index=False)
print(f"✓ Saved: demographics_polarization_data.csv ({len(export_cols)} variables)")

print("\n" + "="*90)
print("✅ CROSS-CORRELATION ANALYSIS COMPLETE")
print("="*90)
print("\n🔍 KEY FINDINGS SUMMARY:")
print("  1. Race effects on polarization: Check ANOVA results above")
print("  2. Gender effects: Check t-test results above")
print("  3. Race × Party interaction: See heatmap visualization")
print("  4. Correlations with other measures: See table above")
print("\n📊 Generated:")
print("  • VIZ_06_demographics_polarization.png (Race & Gender boxplots)")
print("  • VIZ_07_race_party_heatmap.png (Interaction heatmap)")
print("  • demographics_polarization_data.csv (Clean dataset)")
print("="*90)