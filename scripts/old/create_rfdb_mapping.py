"""
Create mapping between rfdb_csvs coaster names and ratings_data coaster names.
This script analyzes both datasets and creates a comprehensive mapping.
"""

import os
import glob
import pandas as pd
import re
from collections import defaultdict
from difflib import SequenceMatcher

def normalize_name(name):
    """Normalize coaster name for comparison."""
    name = name.lower()
    name = re.sub(r'[^\w\s]', '', name)  # Remove punctuation
    name = re.sub(r'\s+', '', name)  # Remove spaces
    return name

def extract_coaster_info_from_path(filepath):
    """Extract park and coaster name from rfdb_csvs path."""
    # Path format: rfdb_csvs/parkname/coastername/timestamp_id.csv
    parts = filepath.replace('\\', '/').split('/')
    if len(parts) >= 3:
        park = parts[1]
        coaster = parts[2]
        return park, coaster
    return None, None

def similarity_score(s1, s2):
    """Calculate similarity between two strings."""
    return SequenceMatcher(None, normalize_name(s1), normalize_name(s2)).ratio()

def load_ratings_coasters():
    """Load unique coaster names from ratings_data."""
    ratings_file = 'ratings_data/processed_all_reviews_metadata_20251110_161035.csv'
    df = pd.read_csv(ratings_file)
    
    # Get unique coaster names
    coasters = df['coaster_name'].unique()
    
    print(f"Found {len(coasters)} unique coasters in ratings_data")
    return coasters

def load_rfdb_coasters():
    """Load unique coasters from rfdb_csvs."""
    csv_files = glob.glob('rfdb_csvs/**/*.csv', recursive=True)
    
    coaster_info = defaultdict(list)  # coaster_name -> [files]
    
    for filepath in csv_files:
        park, coaster = extract_coaster_info_from_path(filepath)
        if park and coaster:
            coaster_info[coaster].append((park, filepath))
    
    print(f"Found {len(coaster_info)} unique coasters in rfdb_csvs")
    print(f"Total CSV files: {len(csv_files)}")
    
    return coaster_info

def create_mapping():
    """Create mapping between rfdb and ratings coasters."""
    
    print("=" * 70)
    print("CREATING COASTER NAME MAPPING")
    print("=" * 70)
    
    # Load both datasets
    ratings_coasters = load_ratings_coasters()
    rfdb_coasters = load_rfdb_coasters()
    
    print(f"\n📊 Dataset Summary:")
    print(f"  Ratings data: {len(ratings_coasters)} coasters")
    print(f"  RFDB data: {len(rfdb_coasters)} coasters")
    
    # Create mapping
    mapping = {}
    unmatched_ratings = []
    unmatched_rfdb = []
    
    # Try to match each ratings coaster to an rfdb coaster
    for ratings_name in sorted(ratings_coasters):
        best_match = None
        best_score = 0
        
        for rfdb_name in rfdb_coasters.keys():
            score = similarity_score(ratings_name, rfdb_name)
            if score > best_score:
                best_score = score
                best_match = rfdb_name
        
        # Only consider it a match if similarity is high enough
        if best_score > 0.6:  # 60% similarity threshold
            mapping[ratings_name] = {
                'rfdb_name': best_match,
                'similarity': best_score,
                'park': rfdb_coasters[best_match][0][0],
                'num_files': len(rfdb_coasters[best_match])
            }
        else:
            unmatched_ratings.append(ratings_name)
    
    # Find unmatched rfdb coasters
    matched_rfdb = set(m['rfdb_name'] for m in mapping.values())
    unmatched_rfdb = [name for name in rfdb_coasters.keys() if name not in matched_rfdb]
    
    print(f"\n✅ Matching Results:")
    print(f"  Matched: {len(mapping)} coasters")
    print(f"  Unmatched ratings: {len(unmatched_ratings)}")
    print(f"  Unmatched RFDB: {len(unmatched_rfdb)}")
    
    return mapping, unmatched_ratings, unmatched_rfdb, rfdb_coasters

def display_mapping(mapping, limit=None):
    """Display the mapping in a readable format."""
    print("\n" + "=" * 70)
    print("MATCHED COASTERS")
    print("=" * 70)
    
    items = sorted(mapping.items(), key=lambda x: x[1]['similarity'], reverse=True)
    if limit:
        items = items[:limit]
    
    for ratings_name, info in items:
        print(f"\n{ratings_name}")
        print(f"  → {info['rfdb_name']} ({info['park']})")
        print(f"  Similarity: {info['similarity']:.2%} | Files: {info['num_files']}")

def save_mapping(mapping, unmatched_ratings, unmatched_rfdb, rfdb_coasters):
    """Save mapping to a Python file."""
    
    with open('coaster_name_mapping_rfdb.py', 'w', encoding='utf-8') as f:
        f.write('"""\n')
        f.write('Mapping between ratings_data coaster names and rfdb_csvs coaster names.\n')
        f.write('Auto-generated mapping.\n')
        f.write('"""\n\n')
        
        # Write the main mapping
        f.write('# Ratings name -> RFDB folder name\n')
        f.write('COASTER_NAME_MAPPING = {\n')
        for ratings_name in sorted(mapping.keys()):
            info = mapping[ratings_name]
            f.write(f'    "{ratings_name}": "{info["rfdb_name"]}",  ')
            f.write(f'# {info["park"]}, {info["num_files"]} files, {info["similarity"]:.1%} match\n')
        f.write('}\n\n')
        
        # Write reverse mapping
        f.write('# RFDB folder name -> Ratings name (reverse lookup)\n')
        f.write('RFDB_TO_RATINGS_NAME = {\n')
        for ratings_name, info in sorted(mapping.items(), key=lambda x: x[1]['rfdb_name']):
            f.write(f'    "{info["rfdb_name"]}": "{ratings_name}",\n')
        f.write('}\n\n')
        
        # Write helper functions
        f.write('def get_rfdb_folder(ratings_name: str) -> str:\n')
        f.write('    """Get RFDB folder name from ratings name."""\n')
        f.write('    return COASTER_NAME_MAPPING.get(ratings_name)\n\n')
        
        f.write('def get_ratings_name(rfdb_name: str) -> str:\n')
        f.write('    """Get ratings name from RFDB folder name."""\n')
        f.write('    return RFDB_TO_RATINGS_NAME.get(rfdb_name)\n\n')
        
        # Write statistics
        f.write(f'# Statistics\n')
        f.write(f'# Total matched: {len(mapping)}\n')
        f.write(f'# Unmatched ratings: {len(unmatched_ratings)}\n')
        f.write(f'# Unmatched RFDB: {len(unmatched_rfdb)}\n')
        
        # Write unmatched lists
        if unmatched_ratings:
            f.write('\n# Unmatched ratings coasters (no RFDB data found):\n')
            for name in sorted(unmatched_ratings)[:20]:
                f.write(f'# - {name}\n')
        
        if unmatched_rfdb:
            f.write('\n# Unmatched RFDB coasters (no ratings found):\n')
            for name in sorted(unmatched_rfdb)[:20]:
                park = rfdb_coasters[name][0][0]
                f.write(f'# - {name} ({park})\n')
    
    print(f"\n✅ Mapping saved to: coaster_name_mapping_rfdb.py")

if __name__ == "__main__":
    # Create mapping
    mapping, unmatched_ratings, unmatched_rfdb, rfdb_coasters = create_mapping()
    
    # Display first 20 matches
    display_mapping(mapping, limit=20)
    
    # Show some unmatched
    if unmatched_ratings:
        print(f"\n\n⚠️ Sample Unmatched Ratings Coasters (first 10):")
        for name in sorted(unmatched_ratings)[:10]:
            print(f"  - {name}")
    
    if unmatched_rfdb:
        print(f"\n\n⚠️ Sample Unmatched RFDB Coasters (first 10):")
        for name in sorted(unmatched_rfdb)[:10]:
            park = rfdb_coasters[name][0][0]
            print(f"  - {name} ({park})")
    
    # Save to file
    save_mapping(mapping, unmatched_ratings, unmatched_rfdb, rfdb_coasters)
    
    print("\n" + "=" * 70)
    print("✅ MAPPING COMPLETE!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"1. Review coaster_name_mapping_rfdb.py")
    print(f"2. Manually fix any incorrect mappings")
    print(f"3. Use this mapping to train the BiGRU model on larger dataset")
