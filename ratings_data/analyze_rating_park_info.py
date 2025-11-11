"""
Improved mapping that considers park names to reduce duplicate mappings.
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

# Load ratings with park info
print("Loading ratings data with park information...")
ratings_file = 'processed_all_reviews_metadata_20251110_161035.csv'
ratings_df = pd.read_csv(ratings_file)

# Get unique coasters with park info
coasters_with_parks = ratings_df[['coaster_id', 'coaster_name', 'park_name']].drop_duplicates('coaster_id')
print(f"Found {len(coasters_with_parks)} unique coasters with park info")

# Show some examples
print("\nSample of rating data (with parks):")
print(coasters_with_parks.head(10)[['coaster_name', 'park_name']].to_string(index=False))

# Check for duplicate coaster names
name_counts = coasters_with_parks.groupby('coaster_name').size()
duplicates = name_counts[name_counts > 1].sort_values(ascending=False)
print(f"\n⚠ Found {len(duplicates)} coaster names that appear in multiple parks:")
print(duplicates.head(10))

print("\nExample: Multiple 'Batman' coasters:")
batman = coasters_with_parks[coasters_with_parks['coaster_name'].str.contains('Batman', case=False, na=False)]
print(batman[['coaster_name', 'park_name']].head(10).to_string(index=False))
