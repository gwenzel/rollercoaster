# Project Organization Guide

## Quick Reference

### 🎯 Main Entry Points

| File | Purpose | Command |
|------|---------|---------|
| `app.py` | Web application | `streamlit run app.py` |
| `scripts/train_bigru_model.py` | Train ML model | `python scripts/train_bigru_model.py` |
| `ratings_data/complete_coaster_mapping.csv` | Master dataset | Load with pandas |

### 📂 Directory Structure

```
rollercoaster/
├── 🌐 app.py                    # Main web app (START HERE)
├── 📄 README.md                 # Project overview
├── 📋 requirements.txt          # Python dependencies
│
├── 🛠️  utils/                   # Core functionality
│   ├── bigru_predictor.py      # BiGRU prediction interface
│   ├── accelerometer_transform.py  # Physics transformation
│   ├── track.py                # Track generation
│   ├── visualize.py            # Plotting
│   └── model.py                # Original model
│
├── 🤖 models/                   # Trained ML models
│   └── bigru_score_model.pth   # BiGRU weights
│
├── 📜 scripts/                  # Standalone scripts
│   ├── bigru_score_predictor.py    # Model definition
│   ├── train_bigru_model.py        # Training
│   ├── create_dummy_model.py       # Generate test model
│   ├── test_*.py                   # Tests
│   ├── create_*.py                 # Data processing
│   └── show_*.py                   # Visualization
│
├── 📊 ratings_data/             # Complete dataset (1,299 coasters)
│   ├── ⭐ complete_coaster_mapping.csv  # MASTER FILE
│   ├── rating_to_rfdb_mapping_enhanced.csv
│   ├── coaster_name_mapping_rfdb.py
│   ├── coaster_id_to_url_mapping.csv
│   ├── all_reviews/            # Review data
│   ├── star_ratings_per_rc/    # Rating distributions
│   ├── tests/                  # Test files
│   └── *.py, *.md              # Scripts & docs
│
├── 🎢 rfdb_csvs/                # Accelerometer data (2,088 files)
│   └── [park]/[coaster]/*.csv  # 3-axis recordings
│
├── 🕷️  crawler/                 # Web scrapers
│   ├── captaincoaster/         # Rating crawler
│   ├── rideforcesdb/           # Accelerometer crawler
│   └── shared/                 # Utilities
│
├── 📚 docs/                     # Documentation
│   ├── BIGRU_INTEGRATION.md
│   ├── BIGRU_README.md
│   ├── INTEGRATION_SUMMARY.md
│   ├── RFDB_MAPPING_COMPLETE.md
│   └── COMPLETE.md
│
└── 📁 accel_data/               # Sample data
```

## 🚀 Common Tasks

### Run the Web App
```bash
streamlit run app.py
```

### Train the Model
```bash
python scripts/train_bigru_model.py
```

### Create Test Model
```bash
python scripts/create_dummy_model.py
```

### Scrape Rating Data
```bash
cd ratings_data
python run_full_rating_scrape.py
```

### Load Complete Dataset
```python
import pandas as pd
df = pd.read_csv('ratings_data/complete_coaster_mapping.csv')
```

### Access Accelerometer Data
```python
import pandas as pd

# Load mapping
mapping = pd.read_csv('ratings_data/complete_coaster_mapping.csv')

# Get a coaster
coaster = mapping[mapping['coaster_name'] == 'Zadra'].iloc[0]

# Load accelerometer data
import glob
csv_files = glob.glob(f"{coaster['full_path']}/*.csv")
accel_data = pd.read_csv(csv_files[0])
```

## 📊 Dataset Statistics

- **Total Coasters with Complete Data**: 1,299
- **Total Accelerometer Files**: 3,700 (avg 2.85 per coaster)
- **Total Ratings Collected**: 428,938
- **Average Rating**: 2.76 stars
- **Coverage**: 73.5% of rated coasters have accelerometer data

## 📖 Key Documentation

| Document | Location | Purpose |
|----------|----------|---------|
| BiGRU Model Guide | `docs/BIGRU_README.md` | Model architecture & training |
| Integration Guide | `docs/BIGRU_INTEGRATION.md` | System integration |
| RFDB Mapping | `docs/RFDB_MAPPING_COMPLETE.md` | Data mapping details |
| Rating Crawler | `ratings_data/RATING_DISTRIBUTION_COMPLETE.md` | Crawler usage |
| Rating Mapping | `ratings_data/RATING_TO_RFDB_MAPPING_FINAL.md` | Mapping methodology |

## 🔧 Development

### Project Structure Rules

1. **Root Level**: Only main entry points (`app.py`, config files, README)
2. **utils/**: Reusable functions and classes
3. **scripts/**: Standalone executable scripts
4. **ratings_data/**: All rating and mapping data
5. **docs/**: All documentation (except root README)
6. **models/**: Trained model weights only

### File Naming Conventions

- **Scripts**: `verb_noun.py` (e.g., `train_bigru_model.py`, `create_mapping.py`)
- **Tests**: `test_*.py` (e.g., `test_accelerometer_transform.py`)
- **Utilities**: `noun.py` (e.g., `track.py`, `model.py`)
- **Documentation**: `TOPIC.md` or `TOPIC_DETAILS.md`

### Import Paths

When importing from scripts:
```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.module_name import ClassName
```

When importing utils:
```python
from utils.module_name import function_name
```

## 🎯 Next Steps

1. ✅ Project organized and cleaned
2. ✅ Complete dataset created (1,299 coasters)
3. ⏳ Train BiGRU on full dataset
4. ⏳ Improve model accuracy
5. ⏳ Add more features to web app

---

Last Updated: November 11, 2025
