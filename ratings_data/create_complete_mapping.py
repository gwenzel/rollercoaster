"""
Create a complete mapping file combining:
- Rating distributions (stars, percentages, counts)
- RFDB accelerometer file paths
- Coaster metadata (names, parks)
"""

import pandas as pd
import os
from pathlib import Path

def create_complete_mapping():
    """Merge rating distributions with RFDB mapping to create complete dataset."""
    
    # Load rating distributions (scraped star ratings)
    rating_dist_file = 'star_ratings_per_rc/rating_distributions_full_20251111_155957.csv'
    print(f"Loading rating distributions from {rating_dist_file}...")
    df_ratings = pd.read_csv(rating_dist_file)
    print(f"  Loaded {len(df_ratings)} coasters with rating distributions")
    
    # Load RFDB mapping (accelerometer file paths)
    mapping_file = 'rating_to_rfdb_mapping_enhanced.csv'
    print(f"\nLoading RFDB mapping from {mapping_file}...")
    df_rfdb = pd.read_csv(mapping_file)
    print(f"  Loaded {len(df_rfdb)} coasters with RFDB mappings")
    
    # Merge on coaster_id
    print("\nMerging datasets...")
    df_complete = pd.merge(
        df_ratings,
        df_rfdb,
        on='coaster_id',
        how='inner',  # Only keep coasters that have BOTH ratings AND accelerometer data
        suffixes=('_rating', '_rfdb')
    )
    
    print(f"  Merged result: {len(df_complete)} coasters with both ratings and accelerometer data")
    
    # Reorder columns for clarity
    columns_order = [
        # Identifiers
        'coaster_id',
        'coaster_name',
        'ratings_coaster',
        'ratings_park',
        
        # Rating data
        'avg_rating',
        'total_ratings',
        
        # Rating distribution percentages
        'pct_0.5_stars',
        'pct_1.0_stars',
        'pct_1.5_stars',
        'pct_2.0_stars',
        'pct_2.5_stars',
        'pct_3.0_stars',
        'pct_3.5_stars',
        'pct_4.0_stars',
        'pct_4.5_stars',
        'pct_5.0_stars',
        
        # Rating distribution counts
        'count_0.5_stars',
        'count_1.0_stars',
        'count_1.5_stars',
        'count_2.0_stars',
        'count_2.5_stars',
        'count_3.0_stars',
        'count_3.5_stars',
        'count_4.0_stars',
        'count_4.5_stars',
        'count_5.0_stars',
        
        # RFDB accelerometer file information
        'rfdb_park_folder',
        'rfdb_coaster_folder',
        'csv_count',
        'full_path',
        
        # Matching quality metrics
        'coaster_similarity',
        'park_similarity',
        'combined_similarity',
        'match_type',
        'match_reason',
        
        # URLs and metadata
        'url',
        'scraped_at'
    ]
    
    # Only include columns that exist
    columns_order = [col for col in columns_order if col in df_complete.columns]
    df_complete = df_complete[columns_order]
    
    # Sort by avg_rating descending
    df_complete = df_complete.sort_values('avg_rating', ascending=False)
    
    # Save complete mapping
    output_file = 'complete_coaster_mapping.csv'
    df_complete.to_csv(output_file, index=False)
    print(f"\n✓ Complete mapping saved to: {output_file}")
    
    # Generate statistics
    print("\n" + "="*60)
    print("COMPLETE MAPPING STATISTICS")
    print("="*60)
    
    print(f"\nTotal coasters with complete data: {len(df_complete)}")
    print(f"Total accelerometer CSV files: {df_complete['csv_count'].sum():.0f}")
    print(f"Average CSV files per coaster: {df_complete['csv_count'].mean():.2f}")
    
    print(f"\nRating statistics:")
    print(f"  Average rating: {df_complete['avg_rating'].mean():.2f} stars")
    print(f"  Median rating: {df_complete['avg_rating'].median():.2f} stars")
    print(f"  Rating range: {df_complete['avg_rating'].min():.2f} - {df_complete['avg_rating'].max():.2f} stars")
    
    print(f"\nTotal ratings collected:")
    print(f"  Total: {df_complete['total_ratings'].sum():.0f} ratings")
    print(f"  Average per coaster: {df_complete['total_ratings'].mean():.0f} ratings")
    print(f"  Median per coaster: {df_complete['total_ratings'].median():.0f} ratings")
    
    print(f"\nMatching quality:")
    perfect_matches = (df_complete['match_type'] == 'perfect').sum()
    fuzzy_matches = (df_complete['match_type'] == 'fuzzy').sum()
    print(f"  Perfect matches (≥95% similarity): {perfect_matches} ({perfect_matches/len(df_complete)*100:.1f}%)")
    print(f"  Fuzzy matches (60-95% similarity): {fuzzy_matches} ({fuzzy_matches/len(df_complete)*100:.1f}%)")
    print(f"  Average coaster similarity: {df_complete['coaster_similarity'].mean():.1f}%")
    print(f"  Average park similarity: {df_complete['park_similarity'].mean():.1f}%")
    print(f"  Average combined similarity: {df_complete['combined_similarity'].mean():.1f}%")
    
    print("\n" + "="*60)
    
    # Show sample of top-rated coasters
    print("\nTop 10 highest-rated coasters with accelerometer data:")
    print("-" * 60)
    for idx, row in df_complete.head(10).iterrows():
        print(f"{row['coaster_name']:30s} {row['avg_rating']:.2f}★ ({row['total_ratings']:.0f} ratings) - {row['csv_count']:.0f} CSV files")
    
    print("\n" + "="*60)
    return df_complete

if __name__ == '__main__':
    # Change to ratings_data directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    df = create_complete_mapping()
    print("\n✓ Complete! Use 'complete_coaster_mapping.csv' for model training.")
