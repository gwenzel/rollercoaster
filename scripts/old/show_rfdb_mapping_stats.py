"""
Display statistics about the RFDB mapping.
"""

from coaster_name_mapping_rfdb import COASTER_NAME_MAPPING, RFDB_TO_RATINGS_NAME
import glob

print("=" * 70)
print("RFDB MAPPING STATISTICS")
print("=" * 70)

# Basic stats
print(f"\n📊 Mapping Statistics:")
print(f"  Total mapped coasters: {len(COASTER_NAME_MAPPING)}")
print(f"  Reverse mappings: {len(RFDB_TO_RATINGS_NAME)}")

# Count perfect matches
perfect_matches = sum(
    1 for k, v in COASTER_NAME_MAPPING.items() 
    if k.lower().replace(' ', '').replace('-', '') == v.lower().replace(' ', '').replace('-', '')
)
print(f"  Perfect name matches: {perfect_matches} ({perfect_matches/len(COASTER_NAME_MAPPING)*100:.1f}%)")

# Count CSV files
csv_files = glob.glob('rfdb_csvs/**/*.csv', recursive=True)
print(f"\n📁 Available Data:")
print(f"  Total CSV files: {len(csv_files)}")

# Sample some mappings
print(f"\n✨ Sample Perfect Matches (first 10):")
count = 0
for k, v in sorted(COASTER_NAME_MAPPING.items()):
    if k.lower().replace(' ', '') == v.lower():
        print(f"  '{k}' → '{v}'")
        count += 1
        if count >= 10:
            break

print(f"\n🔍 Sample Fuzzy Matches (first 10):")
count = 0
for k, v in sorted(COASTER_NAME_MAPPING.items()):
    if k.lower().replace(' ', '') != v.lower():
        print(f"  '{k}' → '{v}'")
        count += 1
        if count >= 10:
            break

# Count files per coaster
from collections import defaultdict
files_per_rfdb = defaultdict(int)
for filepath in csv_files:
    parts = filepath.replace('\\', '/').split('/')
    if len(parts) >= 3:
        rfdb_name = parts[2]
        files_per_rfdb[rfdb_name] += 1

# Stats on data availability
mapped_rfdb_names = set(COASTER_NAME_MAPPING.values())
available_files = sum(files_per_rfdb[rfdb] for rfdb in mapped_rfdb_names if rfdb in files_per_rfdb)

print(f"\n📈 Data Availability:")
print(f"  CSV files for mapped coasters: {available_files}")
print(f"  Average files per coaster: {available_files/len(mapped_rfdb_names):.1f}")

# Show top coasters by file count
print(f"\n🏆 Top 10 Coasters by Data Files:")
top_coasters = sorted(
    [(rfdb, count, RFDB_TO_RATINGS_NAME.get(rfdb, 'Unknown')) 
     for rfdb, count in files_per_rfdb.items() 
     if rfdb in RFDB_TO_RATINGS_NAME],
    key=lambda x: x[1],
    reverse=True
)[:10]

for rfdb, count, ratings_name in top_coasters:
    print(f"  {ratings_name}: {count} files")

print("\n" + "=" * 70)
print("✅ Mapping ready for training!")
print("=" * 70)
