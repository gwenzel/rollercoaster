"""
Enhanced mapping that considers BOTH coaster name AND park name similarity.
This reduces false positives from common coaster names like "Boomerang" or "Wild Mouse".
"""

import pandas as pd
import os
from difflib import SequenceMatcher

def normalize_name(name):
    """Normalize coaster/park name for matching"""
    if pd.isna(name):
        return ""
    name = str(name).lower()
    # Remove common suffixes/prefixes
    for word in ['roller coaster', 'coaster', 'the', '(blue)', '(red)', '(yellow)', 
                 '(white)', '(backwards)', '(forwards)', '(left)', '(right)',
                 'park', 'amusement', 'theme', 'world', 'land']:
        name = name.replace(word, '')
    # Remove special characters but keep spaces
    name = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in name)
    # Remove extra spaces
    name = ' '.join(name.split())
    return name.strip()

def similarity_score(name1, name2):
    """Calculate similarity between two names"""
    norm1 = normalize_name(name1)
    norm2 = normalize_name(name2)
    return SequenceMatcher(None, norm1, norm2).ratio()

def combined_similarity(coaster_name1, park_name1, coaster_name2, park_name2, 
                       coaster_weight=0.7, park_weight=0.3):
    """
    Calculate combined similarity using both coaster and park names.
    
    Args:
        coaster_name1, coaster_name2: Coaster names to compare
        park_name1, park_name2: Park names to compare
        coaster_weight: Weight for coaster name similarity (default 0.7)
        park_weight: Weight for park name similarity (default 0.3)
    
    Returns:
        Combined similarity score (0-1)
    """
    coaster_sim = similarity_score(coaster_name1, coaster_name2)
    park_sim = similarity_score(park_name1, park_name2)
    
    return (coaster_sim * coaster_weight) + (park_sim * park_weight)

def load_ratings_coasters():
    """Load unique coasters with park info from ratings_data."""
    ratings_file = 'processed_all_reviews_metadata_20251110_161035.csv'
    df = pd.read_csv(ratings_file)
    coasters = df[['coaster_id', 'coaster_name', 'park_name']].drop_duplicates('coaster_id')
    coasters = coasters.sort_values('coaster_name')
    print(f"Found {len(coasters)} unique coasters in ratings_data")
    return coasters

def load_rfdb_coasters():
    """Load list of coasters with RFDB accelerometer data."""
    rfdb_dir = '../rfdb_csvs'
    
    park_dirs = [d for d in os.listdir(rfdb_dir) 
                 if os.path.isdir(os.path.join(rfdb_dir, d))]
    
    coasters = []
    for park_folder in park_dirs:
        park_path = os.path.join(rfdb_dir, park_folder)
        coaster_dirs = [d for d in os.listdir(park_path) 
                       if os.path.isdir(os.path.join(park_path, d))]
        
        for coaster_folder in coaster_dirs:
            coaster_path = os.path.join(park_path, coaster_folder)
            csv_files = [f for f in os.listdir(coaster_path) if f.endswith('.csv')]
            
            coasters.append({
                'rfdb_park_folder': park_folder,
                'rfdb_coaster_folder': coaster_folder,
                'csv_count': len(csv_files),
                'full_path': coaster_path
            })
    
    df = pd.DataFrame(coasters)
    print(f"Found {len(df)} coasters with RFDB data across {len(park_dirs)} parks")
    return df

def create_enhanced_mapping(ratings_df, rfdb_df, coaster_threshold=0.6, combined_threshold=0.5):
    """
    Create mapping considering both coaster and park names.
    
    Strategy:
    1. Find best match by coaster name similarity
    2. If coaster similarity >= coaster_threshold, accept it
    3. Otherwise, use combined similarity (coaster + park) with lower threshold
    """
    
    print(f"\nCreating enhanced mapping...")
    print(f"  Coaster name threshold: {coaster_threshold*100}%")
    print(f"  Combined (coaster+park) threshold: {combined_threshold*100}%")
    print(f"Matching {len(ratings_df)} rating coasters to {len(rfdb_df)} RFDB coasters...")
    
    mapping = []
    unmatched = []
    
    for idx, ratings_row in ratings_df.iterrows():
        coaster_id = ratings_row['coaster_id']
        ratings_coaster = ratings_row['coaster_name']
        ratings_park = ratings_row['park_name']
        
        best_match = None
        best_coaster_score = 0
        best_combined_score = 0
        
        # Compare to all RFDB coasters
        for _, rfdb_row in rfdb_df.iterrows():
            rfdb_coaster = rfdb_row['rfdb_coaster_folder']
            rfdb_park = rfdb_row['rfdb_park_folder']
            
            # Calculate coaster name similarity
            coaster_sim = similarity_score(ratings_coaster, rfdb_coaster)
            
            # Calculate combined similarity (coaster + park)
            combined_sim = combined_similarity(
                ratings_coaster, ratings_park,
                rfdb_coaster, rfdb_park,
                coaster_weight=0.7, park_weight=0.3
            )
            
            # Update best match if this is better
            if coaster_sim > best_coaster_score or \
               (coaster_sim == best_coaster_score and combined_sim > best_combined_score):
                best_coaster_score = coaster_sim
                best_combined_score = combined_sim
                best_match = {
                    'rfdb_row': rfdb_row,
                    'coaster_similarity': coaster_sim,
                    'combined_similarity': combined_sim,
                    'park_similarity': similarity_score(ratings_park, rfdb_park)
                }
        
        # Decide if match is good enough
        accepted = False
        match_reason = None
        
        if best_coaster_score >= coaster_threshold:
            accepted = True
            match_reason = 'coaster_name'
        elif best_combined_score >= combined_threshold:
            accepted = True
            match_reason = 'combined'
        
        if accepted and best_match is not None:
            rfdb_row = best_match['rfdb_row']
            mapping.append({
                'coaster_id': coaster_id,
                'ratings_coaster': ratings_coaster,
                'ratings_park': ratings_park,
                'rfdb_coaster_folder': rfdb_row['rfdb_coaster_folder'],
                'rfdb_park_folder': rfdb_row['rfdb_park_folder'],
                'csv_count': rfdb_row['csv_count'],
                'full_path': rfdb_row['full_path'],
                'coaster_similarity': round(best_match['coaster_similarity'] * 100, 1),
                'park_similarity': round(best_match['park_similarity'] * 100, 1),
                'combined_similarity': round(best_match['combined_similarity'] * 100, 1),
                'match_reason': match_reason,
                'match_type': 'perfect' if best_coaster_score >= 0.95 else 'fuzzy'
            })
        else:
            unmatched.append({
                'coaster_id': coaster_id,
                'ratings_coaster': ratings_coaster,
                'ratings_park': ratings_park,
                'best_coaster_score': round(best_coaster_score * 100, 1),
                'best_combined_score': round(best_combined_score * 100, 1) if best_match else 0
            })
    
    print(f"  Matched: {len(mapping)} coasters")
    print(f"  Unmatched: {len(unmatched)} coasters")
    
    return pd.DataFrame(mapping), pd.DataFrame(unmatched)

def main():
    print("=" * 70)
    print("ENHANCED RATING TO RFDB MAPPING (WITH PARK INFO)")
    print("=" * 70)
    print()
    
    # Load data
    print("Loading data...")
    ratings_df = load_ratings_coasters()
    rfdb_df = load_rfdb_coasters()
    
    # Create mapping
    print()
    mapping_df, unmatched_df = create_enhanced_mapping(ratings_df, rfdb_df)
    
    # Save results
    output_file = 'rating_to_rfdb_mapping_enhanced.csv'
    mapping_df.to_csv(output_file, index=False)
    
    unmatched_file = 'rating_to_rfdb_unmatched_enhanced.csv'
    unmatched_df.to_csv(unmatched_file, index=False)
    
    print()
    print("=" * 70)
    print("MAPPING COMPLETE")
    print("=" * 70)
    
    # Statistics
    total_ratings = len(ratings_df)
    perfect = (mapping_df['match_type'] == 'perfect').sum()
    fuzzy = (mapping_df['match_type'] == 'fuzzy').sum()
    by_coaster_name = (mapping_df['match_reason'] == 'coaster_name').sum()
    by_combined = (mapping_df['match_reason'] == 'combined').sum()
    
    print(f"\n📊 Statistics:")
    print(f"  Total coasters in ratings: {total_ratings}")
    print(f"  Coasters mapped: {len(mapping_df)} ({len(mapping_df)/total_ratings*100:.1f}%)")
    print(f"  Coasters unmatched: {len(unmatched_df)}")
    print()
    print(f"  Match quality:")
    print(f"    Perfect (≥95% coaster sim): {perfect} ({perfect/len(mapping_df)*100:.1f}%)")
    print(f"    Fuzzy (60-95% sim): {fuzzy} ({fuzzy/len(mapping_df)*100:.1f}%)")
    print()
    print(f"  Match method:")
    print(f"    By coaster name alone: {by_coaster_name} ({by_coaster_name/len(mapping_df)*100:.1f}%)")
    print(f"    By combined (coaster+park): {by_combined} ({by_combined/len(mapping_df)*100:.1f}%)")
    print()
    print(f"  Data availability:")
    print(f"    Total CSV files: {mapping_df['csv_count'].sum()}")
    print(f"    Average per coaster: {mapping_df['csv_count'].mean():.1f}")
    
    # Check for duplicate RFDB mappings
    print()
    print("=" * 70)
    print("Checking for duplicate RFDB mappings...")
    print("=" * 70)
    duplicate_rfdb = mapping_df.groupby(['rfdb_park_folder', 'rfdb_coaster_folder']).size()
    duplicates = duplicate_rfdb[duplicate_rfdb > 1]
    
    print(f"\n⚠ {len(duplicates)} RFDB coasters mapped to multiple rating coasters")
    
    if len(duplicates) > 0:
        print(f"\nTop 10 cases (may indicate genuinely duplicate coasters):")
        for (park, coaster), count in duplicates.sort_values(ascending=False).head(10).items():
            print(f"\n  {park}/{coaster}: {count} matches")
            matched = mapping_df[
                (mapping_df['rfdb_park_folder'] == park) & 
                (mapping_df['rfdb_coaster_folder'] == coaster)
            ]
            for _, m in matched.iterrows():
                print(f"    - {m['ratings_coaster']:40s} @ {m['ratings_park']:30s} " +
                      f"(coaster:{m['coaster_similarity']}%, park:{m['park_similarity']}%)")
    
    # Show sample mappings
    print()
    print("=" * 70)
    print("Sample mappings:")
    print("=" * 70)
    sample = mapping_df.head(15)[['ratings_coaster', 'ratings_park', 'rfdb_coaster_folder', 
                                   'rfdb_park_folder', 'coaster_similarity', 'park_similarity', 
                                   'csv_count']]
    print(sample.to_string(index=False))
    
    print()
    print("=" * 70)
    print(f"✓ Enhanced mapping saved to: {output_file}")
    print(f"✓ Unmatched coasters saved to: {unmatched_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
