# Ratings Data Directory

This directory contains all rating-related data, scripts, and documentation for the rollercoaster project.

## Directory Structure

```
ratings_data/
├── star_ratings_per_rc/          # Star rating distributions per rollercoaster
│   ├── rating_distributions_*.csv
│   └── (checkpoint files)
│
├── processed_all_reviews_*.csv   # Original review data
├── processed_all_reviews_*.json
│
├── Scripts for Rating Distribution Crawling
├── create_rating_to_rfdb_mapping.py       # Basic mapping generator
├── create_enhanced_rating_mapping.py      # Enhanced mapping (considers parks)
├── run_full_rating_scrape.py              # Main script to scrape all ratings
├── analyze_rating_park_info.py            # Analyze park information
├── show_rating_rfdb_stats.py              # Visualize mapping statistics
│
├── Test Scripts
├── test_rating_crawler.py                 # Single coaster test
├── test_crawler_batch.py                  # Batch test
├── test_rating_distributions_5coasters*.csv
│
├── Mapping Files (Rating coasters → RFDB data)
├── rating_distribution_to_rfdb_mapping.csv       # Basic mapping
├── rating_to_rfdb_mapping_enhanced.csv           # Enhanced mapping (recommended)
├── rating_to_rfdb_mapping.py                     # Python dict version
├── rating_to_rfdb_unmatched.csv                  # Unmapped coasters
├── rating_to_rfdb_unmatched_enhanced.csv         # Enhanced unmapped
├── rating_to_rfdb_mapping_stats.png              # Visualization
│
└── Documentation
    ├── RATING_DISTRIBUTION_COMPLETE.md           # Rating crawler docs
    ├── RATING_TO_RFDB_MAPPING_GUIDE.md           # Mapping usage guide
    └── RATING_TO_RFDB_MAPPING_FINAL.md           # Final mapping summary
```

## Quick Start

### 1. Scrape Rating Distributions

To collect star rating distributions for all coasters:

```bash
cd ratings_data
python run_full_rating_scrape.py
```

Output goes to: `star_ratings_per_rc/rating_distributions_full_TIMESTAMP.csv`

### 2. Generate Mapping to RFDB Data

To create mapping between rating coasters and RFDB accelerometer data:

```bash
cd ratings_data
python create_enhanced_rating_mapping.py
```

This creates: `rating_to_rfdb_mapping_enhanced.csv` (1,299 mapped coasters)

### 3. Visualize Statistics

```bash
python show_rating_rfdb_stats.py
```

Generates: `rating_to_rfdb_mapping_stats.png`

## Key Files

### Rating Distribution Data

**`star_ratings_per_rc/rating_distributions_full_*.csv`**
- Complete rating distributions for all coasters
- Columns: coaster_id, coaster_name, avg_rating, total_ratings, pct_0.5_stars...pct_5.0_stars, count_0.5_stars...count_5.0_stars
- ~1,500 coasters with rating data

### Mapping Files

**`rating_to_rfdb_mapping_enhanced.csv`** (Recommended)
- Maps 1,299 coasters (73.5% coverage)
- Considers both coaster AND park names
- Columns: coaster_id, ratings_coaster, ratings_park, rfdb_coaster_folder, rfdb_park_folder, csv_count, full_path, coaster_similarity, park_similarity, combined_similarity

**`rating_to_rfdb_mapping.py`**
- Python dictionary version
- Helper functions:
  - `get_rfdb_path(ratings_coaster_name)` - Get path to accelerometer data
  - `get_rfdb_info(ratings_coaster_name)` - Get complete info
  - `get_coasters_with_min_data(min_csv_count)` - Filter by data availability

### Original Review Data

**`processed_all_reviews_metadata_*.csv`**
- Individual reviews from Captain Coaster
- ~12,500 reviews for 1,768 coasters
- Columns: coaster_id, coaster_name, park_name, reviewer info

## Usage Examples

### Load Rating Distributions

```python
import pandas as pd

# Load star rating distributions
ratings_df = pd.read_csv('star_ratings_per_rc/rating_distributions_full_20251111_155957.csv')

# Show coaster with highest rating
top_coaster = ratings_df.nlargest(1, 'avg_rating')
print(f"{top_coaster['coaster_name'].iloc[0]}: {top_coaster['avg_rating'].iloc[0]} stars")
```

### Load Mapping and Get RFDB Data

```python
import pandas as pd
import glob

# Load enhanced mapping
mapping_df = pd.read_csv('rating_to_rfdb_mapping_enhanced.csv')

# Get a specific coaster
coaster = mapping_df[mapping_df['ratings_coaster'] == 'Fury 325'].iloc[0]

# Load its accelerometer data
csv_files = glob.glob(f"../{coaster['full_path']}/*.csv")
for csv_file in csv_files:
    accel_data = pd.read_csv(csv_file)
    print(f"Loaded {len(accel_data)} data points from {csv_file}")
```

### Train Model on Combined Data

```python
import pandas as pd
from rating_to_rfdb_mapping import get_rfdb_path, get_coasters_with_min_data

# Load rating distributions
ratings_df = pd.read_csv('star_ratings_per_rc/rating_distributions_full.csv')

# Get coasters with ≥3 RFDB recordings
well_sampled = get_coasters_with_min_data(min_csv_count=3)

# Filter ratings to only include mapped coasters
coaster_names = [name for name, _ in well_sampled]
train_data = ratings_df[ratings_df['coaster_name'].isin(coaster_names)]

print(f"Training dataset: {len(train_data)} coasters")
```

## Data Flow

```
Captain Coaster Website
    ↓
[Rating Distribution Crawler]
    ↓
star_ratings_per_rc/rating_distributions_full.csv
    ↓
[Enhanced Mapping Generator]
    ↓
rating_to_rfdb_mapping_enhanced.csv
    ↓
[Combine with RFDB accelerometer data]
    ↓
Training data for BiGRU model
```

## Statistics

### Rating Distribution Coverage
- Total coasters crawled: ~1,500
- Average rating: 3.2-3.5 stars
- Total ratings collected: ~500,000+

### RFDB Mapping Coverage
- Coasters in ratings data: 1,768
- Coasters in RFDB: 794
- Successfully mapped: 1,299 (73.5%)
- Perfect matches (≥95%): 518 (39.9%)
- Total accelerometer recordings: 3,700 CSV files
- Average recordings per coaster: 2.8

### Top Parks by Data Availability
1. Cedar Point: 50 coasters mapped
2. Six Flags Magic Mountain: 36 coasters
3. Dollywood: 36 coasters
4. Carowinds: 33 coasters
5. Six Flags Discovery Kingdom: 32 coasters

## Documentation

See the following docs for detailed information:

- **RATING_DISTRIBUTION_COMPLETE.md** - How the rating crawler works
- **RATING_TO_RFDB_MAPPING_GUIDE.md** - Complete mapping usage guide
- **RATING_TO_RFDB_MAPPING_FINAL.md** - Final mapping summary and recommendations

## Maintenance

### Update Rating Distributions

Re-run the crawler periodically to get updated ratings:

```bash
cd ratings_data
python run_full_rating_scrape.py
```

### Regenerate Mapping

If new RFDB data or rating data is added:

```bash
cd ratings_data
python create_enhanced_rating_mapping.py
```

### Update Statistics

```bash
python show_rating_rfdb_stats.py
```

## Notes

- All scripts should be run from the `ratings_data` directory
- Paths are relative to the ratings_data folder
- RFDB data is located at `../rfdb_csvs/`
- Output files go to `star_ratings_per_rc/` subfolder

---

**Last Updated**: 2025-11-11  
**Maintainer**: Project Team
