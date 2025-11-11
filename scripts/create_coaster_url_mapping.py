"""
Create mapping of coaster IDs to URLs from existing crawler data
"""

import pandas as pd
import os

# Load the existing crawler data
crawler_data_file = r"c:\Users\Lenovo\rollercoaster\crawler\data\captaincoaster_reviews_progress_page1000_20251110_150850.csv"

print(f"Loading crawler data from {crawler_data_file}...")
df = pd.read_csv(crawler_data_file)

print(f"Loaded {len(df)} reviews")

# Extract unique coaster IDs and URLs
coaster_mapping = df[['coaster_id', 'coaster_name', 'coaster_url']].drop_duplicates('coaster_id')
coaster_mapping = coaster_mapping.sort_values('coaster_id')

print(f"Found {len(coaster_mapping)} unique coasters")

# Save mapping
output_file = r"c:\Users\Lenovo\rollercoaster\coaster_id_to_url_mapping.csv"
coaster_mapping.to_csv(output_file, index=False)
print(f"\n✓ Saved coaster ID to URL mapping to {output_file}")

# Display sample
print("\nSample mappings:")
print(coaster_mapping.head(10).to_string(index=False))

# Statistics
print(f"\nTotal unique coasters with URLs: {len(coaster_mapping)}")
