"""
Create mapping between rating distribution coasters and RFDB accelerometer data.

This mapping connects:
1. Coasters with rating distributions (from Captain Coaster) 
2. Coasters with RFDB wearable accelerometer data

Purpose: Enable training models to predict ratings/distributions from track data
"""

import pandas as pd
import os
from difflib import SequenceMatcher

def normalize_name(name):
    """Normalize coaster name for matching"""
    if pd.isna(name):
        return ""
    name = str(name).lower()
    # Remove common suffixes/prefixes
    for word in ['roller coaster', 'coaster', 'the', '(blue)', '(red)', '(yellow)', 
                 '(white)', '(backwards)', '(forwards)', '(left)', '(right)']:
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

def load_ratings_coasters():
    """Load unique coaster names from ratings_data."""
    ratings_file = 'processed_all_reviews_metadata_20251110_161035.csv'
    df = pd.read_csv(ratings_file)
    coasters = df[['coaster_id', 'coaster_name']].drop_duplicates('coaster_id')
    coasters = coasters.sort_values('coaster_name')
    print(f"Found {len(coasters)} unique coasters in ratings_data")
    return coasters

def load_rfdb_coasters():
    """Load list of coasters with RFDB accelerometer data."""
    rfdb_dir = '../rfdb_csvs'
    
    # Get all subdirectories (park folders)
    park_dirs = [d for d in os.listdir(rfdb_dir) 
                 if os.path.isdir(os.path.join(rfdb_dir, d))]
    
    coasters = []
    for park in park_dirs:
        park_path = os.path.join(rfdb_dir, park)
        # Get all subdirectories (coaster folders) within this park
        coaster_dirs = [d for d in os.listdir(park_path) 
                       if os.path.isdir(os.path.join(park_path, d))]
        
        for coaster_folder in coaster_dirs:
            coaster_path = os.path.join(park_path, coaster_folder)
            # Count CSV files
            csv_files = [f for f in os.listdir(coaster_path) if f.endswith('.csv')]
            
            coasters.append({
                'park': park,
                'rfdb_folder': coaster_folder,
                'csv_count': len(csv_files),
                'full_path': coaster_path
            })
    
    df = pd.DataFrame(coasters)
    print(f"Found {len(df)} coasters with RFDB data across {len(park_dirs)} parks")
    
    # Show park distribution
    park_counts = df.groupby('park')['rfdb_folder'].count().sort_values(ascending=False)
    print(f"  Parks with most coasters:")
    for park, count in park_counts.head(5).items():
        print(f"    {park}: {count} coasters")
    
    return df

def create_mapping(ratings_df, rfdb_df, threshold=0.6):
    """
    Create mapping between ratings coasters and RFDB coasters.
    
    Each rating coaster is matched to the BEST matching RFDB coaster (by name similarity).
    Structure: rating_coaster_name -> rfdb_park/rfdb_coaster_folder
    """
    
    print(f"\nCreating mapping (similarity threshold: {threshold})...")
    print(f"Matching {len(ratings_df)} rating coasters to {len(rfdb_df)} RFDB coasters...")
    
    mapping = []
    unmatched = []
    
    for idx, ratings_row in ratings_df.iterrows():
        coaster_id = ratings_row['coaster_id']
        ratings_name = ratings_row['coaster_name']
        
        best_match = None
        best_score = 0
        
        # Compare this rating coaster to ALL RFDB coasters (across all parks)
        for _, rfdb_row in rfdb_df.iterrows():
            rfdb_folder = rfdb_row['rfdb_folder']
            # Calculate similarity between rating name and RFDB folder name
            score = similarity_score(ratings_name, rfdb_folder)
            
            if score > best_score:
                best_score = score
                best_match = rfdb_row
        
        # If good enough match found, add to mapping
        if best_score >= threshold and best_match is not None:
            mapping.append({
                'coaster_id': coaster_id,
                'ratings_name': ratings_name,
                'rfdb_folder': best_match['rfdb_folder'],
                'rfdb_park': best_match['park'],
                'csv_count': best_match['csv_count'],
                'full_path': best_match['full_path'],
                'similarity': round(best_score * 100, 1),
                'match_type': 'perfect' if best_score >= 0.95 else 'fuzzy'
            })
        else:
            unmatched.append({
                'coaster_id': coaster_id,
                'ratings_name': ratings_name,
                'best_score': round(best_score * 100, 1) if best_score > 0 else 0
            })
    
    print(f"  Matched: {len(mapping)} coasters")
    print(f"  Unmatched: {len(unmatched)} coasters (similarity < {threshold*100}%)")
    
    return pd.DataFrame(mapping), pd.DataFrame(unmatched)

def main():
    print("=" * 70)
    print("RATING DISTRIBUTION TO RFDB MAPPING GENERATOR")
    print("=" * 70)
    print()
    
    # Load coasters from both sources
    print("Loading coaster lists...")
    ratings_df = load_ratings_coasters()
    rfdb_df = load_rfdb_coasters()
    
    print()
    print("Creating mapping...")
    mapping_df, unmatched_df = create_mapping(ratings_df, rfdb_df, threshold=0.6)
    
    # Save mapping
    output_file = 'rating_distribution_to_rfdb_mapping.csv'
    mapping_df.to_csv(output_file, index=False)
    
    # Save unmatched for reference
    unmatched_file = 'rating_to_rfdb_unmatched.csv'
    unmatched_df.to_csv(unmatched_file, index=False)
    
    print()
    print("=" * 70)
    print("MAPPING COMPLETE")
    print("=" * 70)
    
    # Statistics
    perfect_matches = (mapping_df['match_type'] == 'perfect').sum()
    fuzzy_matches = (mapping_df['match_type'] == 'fuzzy').sum()
    total_ratings = len(ratings_df)
    total_rfdb = len(rfdb_df)
    
    print(f"\n📊 Statistics:")
    print(f"  Total coasters in ratings data: {total_ratings}")
    print(f"  Total coasters in RFDB: {total_rfdb}")
    print(f"  Coasters mapped: {len(mapping_df)}")
    print(f"  Coverage: {len(mapping_df)/total_ratings*100:.1f}%")
    print()
    print(f"  Perfect matches (≥95%): {perfect_matches} ({perfect_matches/len(mapping_df)*100:.1f}%)")
    print(f"  Fuzzy matches (60-95%): {fuzzy_matches} ({fuzzy_matches/len(mapping_df)*100:.1f}%)")
    print()
    print(f"  Total CSV files available: {mapping_df['csv_count'].sum()}")
    print(f"  Average CSVs per coaster: {mapping_df['csv_count'].mean():.1f}")
    
    # Top coasters by data availability
    print()
    print("=" * 70)
    print("Top 10 coasters by accelerometer data availability:")
    print("=" * 70)
    top10 = mapping_df.nlargest(10, 'csv_count')[['ratings_name', 'csv_count', 'rfdb_park', 'similarity']]
    print(top10.to_string(index=False))
    
    # Sample of mappings with full paths to show structure
    print()
    print("=" * 70)
    print("Sample mappings (showing park/coaster structure):")
    print("=" * 70)
    sample = mapping_df.head(10)[['ratings_name', 'rfdb_park', 'rfdb_folder', 'csv_count', 'similarity', 'match_type']]
    print(sample.to_string(index=False))
    
    # Show a few full paths to demonstrate structure
    print()
    print("Example full paths (park/coaster/csv files):")
    for _, row in mapping_df.head(3).iterrows():
        print(f"  {row['ratings_name']} -> {row['full_path']} ({row['csv_count']} files)")
    
    # Check for any duplicate mappings (multiple rating coasters mapped to same RFDB coaster)
    duplicate_rfdb = mapping_df.groupby(['rfdb_park', 'rfdb_folder']).size()
    duplicates = duplicate_rfdb[duplicate_rfdb > 1]
    if len(duplicates) > 0:
        print()
        print(f"⚠ Warning: {len(duplicates)} RFDB coasters mapped to multiple rating coasters:")
        for (park, folder), count in duplicates.head(5).items():
            print(f"  {park}/{folder}: {count} rating coasters")
            matched = mapping_df[(mapping_df['rfdb_park'] == park) & (mapping_df['rfdb_folder'] == folder)]
            for _, m in matched.iterrows():
                print(f"    - {m['ratings_name']} ({m['similarity']}% match)")
    
    # Show parks with multiple coasters in mapping
    print()
    print("Parks with most mapped coasters:")
    park_mapped_counts = mapping_df.groupby('rfdb_park')['ratings_name'].count().sort_values(ascending=False)
    for park, count in park_mapped_counts.head(10).items():
        print(f"  {park}: {count} coasters mapped")
    
    print()
    print("=" * 70)
    print(f"✓ Mapping saved to: {output_file}")
    print(f"✓ Unmatched coasters saved to: {unmatched_file}")
    print("=" * 70)
    
    # Create Python mapping file for easy import
    print("\nGenerating Python mapping dictionary...")
    create_python_mapping(mapping_df)

def create_python_mapping(mapping_df):
    """Create a Python dictionary file for easy import"""
    
    output_file = 'rating_to_rfdb_mapping.py'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('"""\n')
        f.write('Mapping between Captain Coaster rating distribution coasters and RFDB accelerometer data.\n')
        f.write('Auto-generated mapping.\n')
        f.write('"""\n\n')
        
        # Main mapping: ratings_name -> rfdb info
        f.write('# Ratings coaster name -> RFDB folder info\n')
        f.write('RATING_TO_RFDB_MAPPING = {\n')
        
        for _, row in mapping_df.iterrows():
            ratings_name = row['ratings_name'].replace('"', '\\"')
            rfdb_folder = row['rfdb_folder']
            rfdb_park = row['rfdb_park']
            csv_count = row['csv_count']
            similarity = row['similarity']
            
            f.write(f'    "{ratings_name}": {{\n')
            f.write(f'        "rfdb_folder": "{rfdb_folder}",\n')
            f.write(f'        "rfdb_park": "{rfdb_park}",\n')
            f.write(f'        "csv_count": {csv_count},\n')
            f.write(f'        "full_path": "rfdb_csvs/{rfdb_park}/{rfdb_folder}",\n')
            f.write(f'        "similarity": {similarity}\n')
            f.write(f'    }},\n')
        
        f.write('}\n\n')
        
        # Reverse mapping: rfdb_folder -> ratings_name
        f.write('# Reverse mapping: RFDB folder -> Ratings coaster name\n')
        f.write('RFDB_TO_RATING_MAPPING = {\n')
        
        for _, row in mapping_df.iterrows():
            ratings_name = row['ratings_name'].replace('"', '\\"')
            rfdb_folder = row['rfdb_folder']
            
            f.write(f'    "{rfdb_folder}": "{ratings_name}",\n')
        
        f.write('}\n\n')
        
        # Helper functions
        f.write('''
def get_rfdb_path(ratings_coaster_name):
    """
    Get RFDB data path for a coaster from ratings data.
    
    Args:
        ratings_coaster_name: Name of coaster from ratings data
        
    Returns:
        Full path to RFDB folder, or None if not found
    """
    mapping = RATING_TO_RFDB_MAPPING.get(ratings_coaster_name)
    if mapping:
        return mapping['full_path']
    return None

def get_rfdb_info(ratings_coaster_name):
    """
    Get complete RFDB info for a coaster.
    
    Args:
        ratings_coaster_name: Name of coaster from ratings data
        
    Returns:
        Dictionary with rfdb_folder, rfdb_park, csv_count, full_path, similarity
        or None if not found
    """
    return RATING_TO_RFDB_MAPPING.get(ratings_coaster_name)

def get_ratings_name(rfdb_folder):
    """
    Get ratings coaster name from RFDB folder name.
    
    Args:
        rfdb_folder: RFDB folder name
        
    Returns:
        Coaster name from ratings data, or None if not found
    """
    return RFDB_TO_RATING_MAPPING.get(rfdb_folder)

def get_all_mapped_coasters():
    """
    Get list of all coasters that have both rating and RFDB data.
    
    Returns:
        List of coaster names from ratings data
    """
    return list(RATING_TO_RFDB_MAPPING.keys())

def get_coasters_with_min_data(min_csv_count=3):
    """
    Get coasters that have at least N CSV files.
    
    Args:
        min_csv_count: Minimum number of CSV files required
        
    Returns:
        List of (coaster_name, csv_count) tuples
    """
    return [(name, info['csv_count']) 
            for name, info in RATING_TO_RFDB_MAPPING.items() 
            if info['csv_count'] >= min_csv_count]
''')
    
    print(f"✓ Python mapping saved to: {output_file}")

if __name__ == "__main__":
    main()
