# Rating Distribution to RFDB Mapping

## Overview

This mapping connects **Captain Coaster rating distributions** with **RFDB wearable accelerometer data**, enabling us to train models that predict ratings/rating distributions from physical track characteristics.

## Mapping Statistics

### Coverage
- **Total coasters in ratings data**: 1,768
- **Total coasters in RFDB**: 794
- **Successfully mapped**: 1,261 coasters (71.3% coverage)

### Match Quality
- **Perfect matches (≥95% similarity)**: 519 (41.2%)
- **Fuzzy matches (60-95% similarity)**: 742 (58.8%)

### Data Availability
- **Total CSV files available**: 3,532 accelerometer recordings
- **Average CSVs per coaster**: 2.8 recordings
- **Range**: 1-5 recordings per coaster

## Files Generated

### 1. CSV Mapping
**File**: `rating_distribution_to_rfdb_mapping.csv`

**Columns**:
```csv
coaster_id,ratings_name,rfdb_folder,rfdb_park,csv_count,full_path,similarity,match_type
```

**Example**:
```csv
1622,Prowler,prowler,worldsoffun,5,rfdb_csvs/worldsoffun/prowler,100.0,perfect
```

### 2. Python Dictionary
**File**: `rating_to_rfdb_mapping.py`

**Main structures**:
- `RATING_TO_RFDB_MAPPING`: Dict[str, dict] - Ratings name → RFDB info
- `RFDB_TO_RATING_MAPPING`: Dict[str, str] - RFDB folder → Ratings name

**Helper functions**:
- `get_rfdb_path(ratings_coaster_name)` - Get path to accelerometer data
- `get_rfdb_info(ratings_coaster_name)` - Get complete RFDB info
- `get_ratings_name(rfdb_folder)` - Reverse lookup
- `get_all_mapped_coasters()` - List all available coasters
- `get_coasters_with_min_data(min_csv_count)` - Filter by data availability

## Usage Examples

### Load Mapping

```python
import pandas as pd

# CSV approach
mapping_df = pd.read_csv('rating_distribution_to_rfdb_mapping.csv')

# Python dictionary approach
from rating_to_rfdb_mapping import (
    RATING_TO_RFDB_MAPPING, 
    get_rfdb_path,
    get_rfdb_info,
    get_coasters_with_min_data
)
```

### Find RFDB Data for a Coaster

```python
from rating_to_rfdb_mapping import get_rfdb_info, get_rfdb_path

# Get complete info
info = get_rfdb_info("Prowler")
print(info)
# {'rfdb_folder': 'prowler', 
#  'rfdb_park': 'worldsoffun', 
#  'csv_count': 5,
#  'full_path': 'rfdb_csvs/worldsoffun/prowler',
#  'similarity': 100.0}

# Get just the path
path = get_rfdb_path("Prowler")
# 'rfdb_csvs/worldsoffun/prowler'

# Load accelerometer data
import glob
csv_files = glob.glob(f"{path}/*.csv")
for csv_file in csv_files:
    accel_data = pd.read_csv(csv_file)
    # Process accelerometer data...
```

### Filter by Data Availability

```python
from rating_to_rfdb_mapping import get_coasters_with_min_data

# Get coasters with at least 3 recordings
well_sampled = get_coasters_with_min_data(min_csv_count=3)

print(f"Found {len(well_sampled)} coasters with ≥3 recordings")
for name, count in well_sampled[:10]:
    print(f"  {name}: {count} recordings")
```

### Load Both Rating and Track Data

```python
import pandas as pd
import glob
from rating_to_rfdb_mapping import get_rfdb_path

# Load rating distribution data (when available)
ratings_df = pd.read_csv('rating_distributions_full_20251111.csv')

# Get a specific coaster
coaster_name = "Prowler"
rating_data = ratings_df[ratings_df['coaster_name'] == coaster_name].iloc[0]

# Load accelerometer data
rfdb_path = get_rfdb_path(coaster_name)
if rfdb_path:
    csv_files = glob.glob(f"{rfdb_path}/*.csv")
    
    for csv_file in csv_files:
        accel_data = pd.read_csv(csv_file)
        
        # Now you have both:
        print(f"Coaster: {coaster_name}")
        print(f"  Average rating: {rating_data['avg_rating']}")
        print(f"  5-star %: {rating_data['pct_5.0_stars']}")
        print(f"  Accelerometer data: {len(accel_data)} rows")
```

## Training Pipeline Integration

### Complete Training Data Preparation

```python
import pandas as pd
import glob
import numpy as np
from rating_to_rfdb_mapping import RATING_TO_RFDB_MAPPING, get_coasters_with_min_data
from utils.accelerometer_transform import track_to_accelerometer_data

# 1. Load rating distributions
ratings_df = pd.read_csv('rating_distributions_full_20251111.csv')

# 2. Get coasters with sufficient data (e.g., ≥3 recordings)
well_sampled = get_coasters_with_min_data(min_csv_count=3)
coaster_names = [name for name, _ in well_sampled]

# 3. Filter ratings to only include mapped coasters
mapped_ratings = ratings_df[ratings_df['coaster_name'].isin(coaster_names)]

print(f"Training dataset: {len(mapped_ratings)} coasters")

# 4. Prepare training data
training_data = []

for _, row in mapped_ratings.iterrows():
    coaster_name = row['coaster_name']
    rfdb_info = RATING_TO_RFDB_MAPPING[coaster_name]
    
    # Load all accelerometer recordings for this coaster
    csv_files = glob.glob(f"{rfdb_info['full_path']}/*.csv")
    
    for csv_file in csv_files:
        track_data = pd.read_csv(csv_file)
        
        # Transform to accelerometer format if needed
        if 'Lateral' not in track_data.columns:
            accel_data = track_to_accelerometer_data(track_data)
        else:
            accel_data = track_data
        
        # Extract features
        features = accel_data[['Lateral', 'Vertical', 'Longitudinal']].values
        
        # Target: average rating
        target_rating = row['avg_rating']
        
        # Optional: use distribution as target
        target_dist = np.array([
            row['pct_0.5_stars'], row['pct_1.0_stars'], row['pct_1.5_stars'],
            row['pct_2.0_stars'], row['pct_2.5_stars'], row['pct_3.0_stars'],
            row['pct_3.5_stars'], row['pct_4.0_stars'], row['pct_4.5_stars'],
            row['pct_5.0_stars']
        ])
        
        training_data.append({
            'coaster_name': coaster_name,
            'features': features,
            'target_rating': target_rating,
            'target_distribution': target_dist,
            'csv_file': csv_file
        })

print(f"Total training samples: {len(training_data)}")
```

### BiGRU Model Training with New Data

```python
from bigru_score_predictor import BiGRUScorePredictor
import torch

# Initialize predictor
predictor = BiGRUScorePredictor(
    input_size=3,  # 3-axis accelerometer
    hidden_size=128,
    num_layers=2
)

# Prepare dataset
X_train = []
y_train = []

for sample in training_data:
    X_train.append(sample['features'])
    y_train.append(sample['target_rating'])

# Train model
predictor.train_model(
    X_train=X_train,
    y_train=y_train,
    epochs=100,
    batch_size=32
)

# Save model
predictor.save_model('models/bigru_rfdb_rating_predictor.pth')
```

## Top Coasters by Data Availability

These coasters have the most accelerometer recordings (5 each):

| Coaster | Park | Similarity |
|---------|------|------------|
| Adventure Express | Kings Island | 97.0% |
| Afterburn | Carowinds | 100.0% |
| All American Triple Loop | Indiana Beach | 93.3% |
| AlpenFury | Canada's Wonderland | 100.0% |
| Alpengeist | Busch Gardens Williamsburg | 100.0% |
| American Eagle (Blue) | Six Flags Great America | 86.7% |
| American Eagle (Red) | Six Flags Great America | 89.7% |
| American Thunder | Six Flags St. Louis | 100.0% |

## Data Quality Considerations

### Perfect Matches (≥95% similarity)
These coasters have exact or near-exact name matches between rating and RFDB data:
- 519 coasters (41.2%)
- High confidence in correct pairing
- Recommended for initial model training

### Fuzzy Matches (60-95% similarity)
These coasters have approximate name matches:
- 742 coasters (58.8%)
- Should be manually verified if critical
- Generally reliable but may have edge cases

### Unmapped Coasters
- **977 coasters in ratings** but not in RFDB (55.3%)
- **Not enough data for training** but can be used for prediction once model is trained
- Predictions can still be made from track data (if available via other means)

## Model Training Strategies

### Strategy 1: High-Quality Subset
Train on perfect matches only for maximum reliability:
```python
# Filter to perfect matches
perfect_only = mapping_df[mapping_df['match_type'] == 'perfect']
# Results in 519 coasters
```

### Strategy 2: Maximum Data
Use all mapped coasters for maximum training data:
```python
# Use all 1,261 mapped coasters
all_mapped = mapping_df.copy()
```

### Strategy 3: Well-Sampled Only
Focus on coasters with multiple recordings:
```python
# Coasters with ≥3 recordings for robustness
well_sampled = mapping_df[mapping_df['csv_count'] >= 3]
# More reliable average characteristics
```

## Prediction Workflow

### Option A: Predict Rating from RFDB Track Data
```python
# 1. Load track data from RFDB
track_data = pd.read_csv('rfdb_csvs/carowinds/afterburn/afterburn_001.csv')

# 2. Transform to accelerometer format
accel_data = track_to_accelerometer_data(track_data)

# 3. Load trained model
predictor = BiGRUScorePredictor.load_model('models/bigru_rfdb_rating_predictor.pth')

# 4. Predict rating
predicted_rating = predictor.predict(accel_data[['Lateral', 'Vertical', 'Longitudinal']].values)
print(f"Predicted rating: {predicted_rating:.2f} stars")
```

### Option B: Predict Distribution from Track Data
```python
# Train a distribution predictor
class DistributionPredictor(BiGRUScorePredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Output layer: 10 values (one per rating level)
        self.fc = nn.Linear(self.hidden_size, 10)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # ... BiGRU processing ...
        distribution = self.softmax(output)  # 10 probabilities
        return distribution

# Predict full distribution
predicted_dist = dist_predictor.predict(accel_data)
# predicted_dist: [pct_0.5, pct_1.0, ..., pct_5.0]
```

## Integration with Streamlit App

```python
# In app.py, allow users to select coasters with both rating and track data

from rating_to_rfdb_mapping import get_all_mapped_coasters, get_rfdb_info

# Get list of coasters with both data types
available_coasters = get_all_mapped_coasters()

# Streamlit selector
selected_coaster = st.selectbox("Select coaster:", available_coasters)

# Load rating distribution
ratings_df = pd.read_csv('rating_distributions_full.csv')
rating_data = ratings_df[ratings_df['coaster_name'] == selected_coaster].iloc[0]

# Display rating distribution
st.write(f"Average rating: {rating_data['avg_rating']} stars")
st.bar_chart({
    '0.5★': rating_data['pct_0.5_stars'],
    '1★': rating_data['pct_1.0_stars'],
    # ... etc
})

# Load and display track data
rfdb_info = get_rfdb_info(selected_coaster)
st.write(f"Available recordings: {rfdb_info['csv_count']}")

# Allow prediction from track data
if st.button("Predict from track data"):
    # Load track, predict, compare with actual
    pass
```

## Future Enhancements

### Potential Improvements
- [ ] Manual verification of fuzzy matches
- [ ] Add park name to matching algorithm (improve accuracy)
- [ ] Handle coasters with multiple names (aliases)
- [ ] Track temporal changes (re-scrape ratings periodically)
- [ ] Cross-validation: predict on unmapped coasters

### Advanced Modeling
- [ ] Multi-task learning: predict both rating AND distribution
- [ ] Ensemble models: combine track features + metadata
- [ ] Transfer learning: use well-sampled coasters to improve predictions for poorly-sampled ones
- [ ] Distribution shape analysis: cluster coasters by rating curve

## Maintenance

### Updating the Mapping

When new rating distributions or RFDB data becomes available:

```bash
# Regenerate mapping
python create_rating_to_rfdb_mapping.py
```

This will:
1. Scan `ratings_data/` for coasters with ratings
2. Scan `rfdb_csvs/` for coasters with accelerometer data
3. Create new mapping with updated statistics
4. Regenerate `rating_to_rfdb_mapping.py` dictionary

## Summary

This mapping enables the core objective: **training models that predict coaster ratings/distributions from physical track characteristics (accelerometer data)**.

With 1,261 coasters having both rating distributions and accelerometer data (averaging 2.8 recordings each), we have a solid foundation for building predictive models that can:

1. **Predict ratings** from track acceleration patterns
2. **Predict rating distributions** (full histogram of user ratings)
3. **Identify which track features** correlate with high ratings
4. **Design new coasters** by optimizing predicted ratings

---

**Created**: 2025-11-11  
**Coasters Mapped**: 1,261 (71.3% coverage)  
**Total Recordings**: 3,532 CSV files
