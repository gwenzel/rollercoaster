"""
Create a mapping dictionary between roller coaster names in ratings_data and accel_data filenames.
"""

import os
import pandas as pd
from difflib import get_close_matches

# Load the ratings data with coaster names
ratings_file = r"c:\Users\Lenovo\rollercoaster\ratings_data\processed_all_reviews_metadata_20251110_161035.csv"
df = pd.read_csv(ratings_file)

# Get unique coaster names from ratings data
unique_coasters = df['coaster_name'].unique()
print(f"Found {len(unique_coasters)} unique coasters in ratings data\n")

# Get accel_data filenames
accel_dir = r"c:\Users\Lenovo\rollercoaster\accel_data"
accel_files = [f.replace('.csv', '') for f in os.listdir(accel_dir) if f.endswith('.csv')]
print(f"Found {len(accel_files)} files in accel_data:")
for f in sorted(accel_files):
    print(f"  - {f}")
print()

# Manual mapping based on analysis
name_mapping = {
    # Direct matches or clear mappings
    "Iron Gwazi": "IronGwazi",
    "Skyrush": "Skyrush",
    "Taron": "Taron",
    "Maverick": "Maverick",
    "El Toro": "ElToro",
    "Shambhala": "Shambhala",
    "Steel Vengeance": "SteelVengeance",
    "Untamed": "Untamed",
    "VelociCoaster": "VelociCoas",  # Truncated filename
    "Hyperia": "Hyperia",
    "Pantheon": "Pantheon",
    "Zadra": "Zadra",
    "Taiga": "Taiga",
    
    # Need to find these in ratings_data
    # "???": "AlpenFury",
    # "???": "Anubis",
    # "???": "ArieForce",
    # "Lightning ???": "Lightning",
    # "???": "Pantherian",
    # "???": "RidetoHa",
    # "???": "SteelVeng",  # Duplicate/truncated Steel Vengeance?
    # "Twisted ???": "TwistedCo",
    # "???": "WickedCyc",
}

# Search for potential matches for remaining files
unmapped_files = [f for f in accel_files if f not in name_mapping.values()]
print(f"\nSearching for matches for unmapped files ({len(unmapped_files)}):\n")

for accel_file in unmapped_files:
    # Try to find close matches in coaster names
    matches = get_close_matches(accel_file, unique_coasters, n=5, cutoff=0.4)
    if matches:
        print(f"{accel_file}:")
        for match in matches:
            print(f"  → {match}")
    else:
        # Try partial matching
        partial_matches = [name for name in unique_coasters if accel_file.lower() in name.lower() or name.lower() in accel_file.lower()]
        if partial_matches:
            print(f"{accel_file} (partial):")
            for match in partial_matches[:5]:
                print(f"  → {match}")
    print()

# Additional searches for specific patterns
print("\nSearching for specific patterns:")
print("\n--- Lightning ---")
lightning_matches = [name for name in unique_coasters if 'lightning' in name.lower()]
for match in lightning_matches[:10]:
    print(f"  {match}")

print("\n--- Twisted ---")
twisted_matches = [name for name in unique_coasters if 'twisted' in name.lower()]
for match in twisted_matches[:10]:
    print(f"  {match}")

print("\n--- Arie ---")
arie_matches = [name for name in unique_coasters if 'arie' in name.lower() or 'aries' in name.lower() or 'arieforce' in name.lower()]
for match in arie_matches[:10]:
    print(f"  {match}")

print("\n--- Wicked or Cyc ---")
wicked_matches = [name for name in unique_coasters if 'wicked' in name.lower() or 'cyclone' in name.lower()]
for match in wicked_matches[:10]:
    print(f"  {match}")

print("\n--- Ride to ---")
ride_matches = [name for name in unique_coasters if 'ride to' in name.lower() or 'ride-to' in name.lower()]
for match in ride_matches[:10]:
    print(f"  {match}")

print("\n--- Alpen or Fury ---")
alpen_matches = [name for name in unique_coasters if 'alpen' in name.lower() or 'fury' in name.lower()]
for match in alpen_matches[:10]:
    print(f"  {match}")

print("\n--- Anubis ---")
anubis_matches = [name for name in unique_coasters if 'anubis' in name.lower()]
for match in anubis_matches[:10]:
    print(f"  {match}")

print("\n--- Panther ---")
panther_matches = [name for name in unique_coasters if 'panther' in name.lower()]
for match in panther_matches[:10]:
    print(f"  {match}")

print("\n" + "="*60)
print("\nCurrent mapping dictionary:")
print("name_mapping = {")
for ratings_name, accel_name in sorted(name_mapping.items()):
    print(f'    "{ratings_name}": "{accel_name}",')
print("}")
