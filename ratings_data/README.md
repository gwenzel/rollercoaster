# Ratings Data - Complete Guide

## Overview

This directory contains the **complete dataset** linking roller coaster ratings with accelerometer data, enabling ML models to predict ride ratings from physical track characteristics.

## 🎯 Main Dataset

### `complete_coaster_mapping.csv` ⭐ **USE THIS FILE**

The **master dataset** with everything you need:
- **1,299 coasters** with both rating distributions AND accelerometer data
- **3,700 accelerometer CSV files** (avg 2.85 per coaster)
- **428,938 total ratings** from Captain Coaster

**Columns (37 total)**:
- **Identifiers**: `coaster_id`, `coaster_name`, `ratings_coaster`, `ratings_park`
- **Ratings**: `avg_rating`, `total_ratings`
- **Rating Distributions**: `pct_0.5_stars` through `pct_5.0_stars` (percentages)
- **Rating Counts**: `count_0.5_stars` through `count_5.0_stars` (actual counts)
- **Accelerometer Paths**: `rfdb_park_folder`, `rfdb_coaster_folder`, `csv_count`, `full_path`
- **Match Quality**: `coaster_similarity`, `park_similarity`, `combined_similarity`, `match_type`
- **Metadata**: `url`, `scraped_at`

**Quick Load**:
```python
import pandas as pd
df = pd.read_csv('complete_coaster_mapping.csv')
print(f"{len(df)} coasters with complete data")
```

## 📊 Dataset Statistics

- **Coverage**: 73.5% of rated coasters have accelerometer data
- **Match Quality**: 
  - Perfect matches (≥95% similarity): 518 (39.9%)
  - Fuzzy matches (60-95% similarity): 781 (60.1%)
- **Average Similarity**: 76.2% combined (coaster+park names)
- **Rating Range**: 0.50 - 4.90 stars
- **Average Rating**: 2.76 stars

## 📁 Directory Structure

```
ratings_data/
├── ⭐ complete_coaster_mapping.csv          # MASTER DATASET - Use this!
│
├── 📊 Supporting Data Files
│   ├── rating_to_rfdb_mapping_enhanced.csv  # Rating→RFDB mapping (base for complete)
│   ├── coaster_name_mapping_rfdb.py         # Name mapping functions
│   ├── coaster_id_to_url_mapping.csv        # Coaster URLs (1,500 coasters)
│   └── rating_to_rfdb_mapping_stats.png     # Visualization
│
├── 📜 Scripts
│   ├── create_complete_mapping.py           # Generates complete_coaster_mapping.csv
│   ├── create_enhanced_rating_mapping.py    # Creates rating→RFDB mapping
│   ├── create_rating_to_rfdb_mapping.py     # Original mapping script
│   ├── run_full_rating_scrape.py            # Scrapes all rating distributions
│   ├── analyze_rating_park_info.py          # Park info analysis
│   └── show_rating_rfdb_stats.py            # Display statistics
│
├── 📁 Data Directories
│   ├── star_ratings_per_rc/                 # Rating distributions (1,768 coasters)
│   ├── all_reviews/                         # Original review data
│   ├── tests/                               # Test scripts & data
│   └── archive/                             # Obsolete files (kept for reference)
│
└── 📖 README.md                              # This file
```

## 🚀 Quick Start

### 1. Load Complete Dataset

```python
import pandas as pd
import glob

# Load master mapping
df = pd.read_csv('ratings_data/complete_coaster_mapping.csv')

# Get top-rated coasters
top_rated = df[df['avg_rating'] >= 4.5].sort_values('avg_rating', ascending=False)
print(f"Found {len(top_rated)} coasters rated 4.5+ stars")

# Example: Load accelerometer data for highest-rated coaster
coaster = df.iloc[0]  # Zadra (4.90 stars)
print(f"\nCoaster: {coaster['coaster_name']}")
print(f"Rating: {coaster['avg_rating']} stars from {coaster['total_ratings']:.0f} ratings")
print(f"Path: {coaster['full_path']}")

# Load all accelerometer recordings for this coaster
csv_files = glob.glob(f"../{coaster['full_path']}/*.csv")
print(f"Found {len(csv_files)} accelerometer recordings")

if csv_files:
    accel_df = pd.read_csv(csv_files[0])
    print(f"\nAccelerometer data columns: {list(accel_df.columns)}")
```

### 2. Filter Dataset

```python
# Get coasters with multiple recordings
well_sampled = df[df['csv_count'] >= 3]
print(f"{len(well_sampled)} coasters with 3+ recordings")

# Get coasters with many ratings (reliable)
highly_rated = df[df['total_ratings'] >= 500]
print(f"{len(highly_rated)} coasters with 500+ ratings")

# Get high-quality matches only
perfect_matches = df[df['match_type'] == 'perfect']
print(f"{len(perfect_matches)} perfect name matches")
```

### 3. Analyze Rating Distributions

```python
# Get coasters where most people gave 5 stars
df['pct_5_stars'] = pd.to_numeric(df['pct_5.0_stars'], errors='coerce')
crowd_favorites = df[df['pct_5_stars'] >= 70].sort_values('pct_5_stars', ascending=False)

print("Top crowd favorites (70%+ gave 5 stars):")
for _, row in crowd_favorites.head(10).iterrows():
    print(f"  {row['coaster_name']:30s} {row['pct_5_stars']:.1f}% gave 5★")
```

## 🔧 Regenerating Data

### Scrape Rating Distributions

```bash
cd ratings_data
python run_full_rating_scrape.py
```

This will:
- Scrape ~1,500+ coasters from Captain Coaster
- Extract rating distributions from JavaScript
- Save to `star_ratings_per_rc/rating_distributions_full_[timestamp].csv`
- Create checkpoints every 50 coasters

### Recreate Complete Mapping

```bash
cd ratings_data
python create_complete_mapping.py
```

This will:
- Load latest rating distributions
- Load RFDB mapping
- Merge on `coaster_id`
- Generate `complete_coaster_mapping.csv`

## 📖 Data Sources

### Captain Coaster (Ratings)
- **URL**: https://captaincoaster.com
- **Data**: Star ratings, rating distributions, review counts
- **Coasters**: 1,768 with rating data
- **Crawled**: November 11, 2025

### RideForcesDB (Accelerometer)
- **Source**: Real wearable sensor recordings
- **Data**: 3-axis accelerometer (Lateral, Vertical, Longitudinal)
- **Coasters**: 794 unique coasters across 152 parks
- **Files**: 2,088 CSV recordings

### Mapping Methodology
- **Fuzzy String Matching**: 60% threshold using SequenceMatcher
- **Combined Scoring**: 70% coaster name, 30% park name
- **Park Hierarchy**: Understands `park_folder/coaster_folder` structure
- **Name Normalization**: Lowercasing, special char removal, spacing

## 🎯 Use Cases

### Train ML Models
```python
# Use complete_coaster_mapping.csv to:
# 1. Load accelerometer sequences as input features
# 2. Use avg_rating as prediction target
# 3. Use rating distribution percentages for multi-output prediction
```

### Analyze Ride Forces
```python
# Correlate accelerometer data with ratings:
# - Do higher g-forces lead to higher ratings?
# - What acceleration patterns do people prefer?
# - Can we identify optimal force profiles?
```

### Predict Rating Distributions
```python
# Train models to predict full rating distribution (10 outputs):
# - pct_0.5_stars through pct_5.0_stars
# - More informative than single average rating
```

## 📚 Additional Documentation

- **Rating Crawler**: `RATING_DISTRIBUTION_COMPLETE.md`
- **Project Structure**: See `../docs/PROJECT_STRUCTURE.md`

## 🗄️ Archive Folder

The `archive/` folder contains obsolete files kept for reference:
- `README_old.md` - Previous README with older structure
- `RATING_TO_RFDB_MAPPING_GUIDE.md` - Detailed mapping guide (merged into this README)
- `RATING_TO_RFDB_MAPPING_FINAL.md` - Mapping methodology (merged into this README)
- `rating_distribution_to_rfdb_mapping.csv` - Original mapping (replaced by enhanced)
- `rating_to_rfdb_mapping.py` - Python dict version (replaced by CSV)
- `coaster_name_mapping.py` - Old mapping format
- `rating_to_rfdb_unmatched*.csv` - Lists of unmatched coasters

## 🎉 Summary

Everything you need is in **`complete_coaster_mapping.csv`**:
- 1,299 coasters with both ratings and accelerometer data
- Complete rating distributions (percentages and counts)
- File paths to 3,700 accelerometer recordings
- Match quality metrics for data filtering
- Ready for ML model training and analysis

---

**Last Updated**: November 11, 2025  
**Dataset Version**: 1.0 (Complete)
