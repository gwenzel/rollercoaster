"""
Visualize rating distribution to RFDB mapping statistics
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load mapping (prefer enhanced version if available)
import os
if os.path.exists('rating_to_rfdb_mapping_enhanced.csv'):
    mapping_df = pd.read_csv('rating_to_rfdb_mapping_enhanced.csv')
elif os.path.exists('rating_distribution_to_rfdb_mapping.csv'):
    mapping_df = pd.read_csv('rating_distribution_to_rfdb_mapping.csv')
else:
    print("Error: No mapping file found!")
    sys.exit(1)

print("=" * 70)
print("RATING TO RFDB MAPPING STATISTICS")
print("=" * 70)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Rating Distribution to RFDB Mapping Analysis', fontsize=16, fontweight='bold')

# 1. Match type distribution
ax1 = axes[0, 0]
match_counts = mapping_df['match_type'].value_counts()
colors = ['#2ecc71', '#3498db']
ax1.pie(match_counts, labels=[f'{label.capitalize()}\n({count} coasters)' 
                               for label, count in match_counts.items()],
        autopct='%1.1f%%', colors=colors, startangle=90)
ax1.set_title('Match Quality Distribution', fontsize=14, fontweight='bold')

# 2. Similarity distribution
ax2 = axes[0, 1]
ax2.hist(mapping_df['similarity'], bins=30, color='#3498db', alpha=0.7, edgecolor='black')
ax2.axvline(mapping_df['similarity'].mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {mapping_df["similarity"].mean():.1f}%')
ax2.axvline(95, color='green', linestyle='--', linewidth=2, label='Perfect threshold: 95%')
ax2.set_xlabel('Similarity Score (%)', fontsize=12)
ax2.set_ylabel('Number of Coasters', fontsize=12)
ax2.set_title('Name Similarity Distribution', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 3. CSV count distribution
ax3 = axes[1, 0]
csv_counts = mapping_df['csv_count'].value_counts().sort_index()
ax3.bar(csv_counts.index, csv_counts.values, color='#e74c3c', alpha=0.7, edgecolor='black')
ax3.set_xlabel('Number of CSV Files per Coaster', fontsize=12)
ax3.set_ylabel('Number of Coasters', fontsize=12)
ax3.set_title('Accelerometer Data Availability', fontsize=14, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# Add text with statistics
stats_text = f"""
Total Mapped: {len(mapping_df)}
Perfect: {(mapping_df['match_type'] == 'perfect').sum()}
Fuzzy: {(mapping_df['match_type'] == 'fuzzy').sum()}
Total CSVs: {mapping_df['csv_count'].sum()}
Avg/Coaster: {mapping_df['csv_count'].mean():.1f}
"""
ax3.text(0.98, 0.98, stats_text, transform=ax3.transAxes,
         fontsize=10, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4. Top parks by data availability
ax4 = axes[1, 1]
park_stats = mapping_df.groupby('rfdb_park').agg({
    'csv_count': 'sum',
    'coaster_id': 'count'
}).sort_values('csv_count', ascending=False).head(15)

y_pos = range(len(park_stats))
ax4.barh(y_pos, park_stats['csv_count'], color='#9b59b6', alpha=0.7, edgecolor='black')
ax4.set_yticks(y_pos)
ax4.set_yticklabels([f"{park} ({count} coasters)" 
                      for park, count in zip(park_stats.index, park_stats['coaster_id'])],
                     fontsize=9)
ax4.set_xlabel('Total CSV Files', fontsize=12)
ax4.set_title('Top 15 Parks by Accelerometer Data', fontsize=14, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('rating_to_rfdb_mapping_stats.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved to: rating_to_rfdb_mapping_stats.png")

# Show detailed statistics
print("\n" + "=" * 70)
print("DETAILED STATISTICS")
print("=" * 70)

print("\n📊 Overall Coverage:")
print(f"  Coasters mapped: {len(mapping_df)}")
print(f"  Total CSV files: {mapping_df['csv_count'].sum()}")
print(f"  Average CSVs per coaster: {mapping_df['csv_count'].mean():.2f}")
print(f"  Median CSVs per coaster: {mapping_df['csv_count'].median():.0f}")

print("\n📊 Match Quality:")
perfect = mapping_df[mapping_df['match_type'] == 'perfect']
fuzzy = mapping_df[mapping_df['match_type'] == 'fuzzy']
print(f"  Perfect matches (≥95%): {len(perfect)} ({len(perfect)/len(mapping_df)*100:.1f}%)")
print(f"  Fuzzy matches (60-95%): {len(fuzzy)} ({len(fuzzy)/len(mapping_df)*100:.1f}%)")
print(f"  Average similarity: {mapping_df['similarity'].mean():.1f}%")
print(f"  Median similarity: {mapping_df['similarity'].median():.1f}%")

print("\n📊 Data Richness:")
well_sampled = mapping_df[mapping_df['csv_count'] >= 3]
print(f"  Coasters with ≥3 recordings: {len(well_sampled)} ({len(well_sampled)/len(mapping_df)*100:.1f}%)")
print(f"  Coasters with ≥5 recordings: {len(mapping_df[mapping_df['csv_count'] >= 5])}")
print(f"  Coasters with only 1 recording: {len(mapping_df[mapping_df['csv_count'] == 1])}")

print("\n📊 Top 10 Parks by Total Data:")
park_summary = mapping_df.groupby('rfdb_park').agg({
    'csv_count': 'sum',
    'coaster_id': 'count'
}).sort_values('csv_count', ascending=False).head(10)

for park, row in park_summary.iterrows():
    print(f"  {park:30s} - {int(row['csv_count']):3d} CSVs, {int(row['coaster_id']):3d} coasters")

print("\n📊 Top 10 Coasters by Data Availability:")
top_coasters = mapping_df.nlargest(10, 'csv_count')[['ratings_name', 'csv_count', 'rfdb_park', 'similarity']]
for _, row in top_coasters.iterrows():
    print(f"  {row['ratings_name']:40s} - {int(row['csv_count'])} CSVs ({row['similarity']:.1f}% match)")

print("\n" + "=" * 70)
