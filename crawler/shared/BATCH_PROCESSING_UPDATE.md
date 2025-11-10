# ✅ Updated: Batch Processing Support

## What Changed

The `process_reviews.py` script now supports **batch processing** of multiple CSV files from a folder!

### Before
```bash
# Could only process one file at a time
python process_reviews.py data/captaincoaster_small.csv
```

### After
```bash
# Can process entire folder of CSV files
python process_reviews.py data/

# Automatically:
# - Finds all CSV files in folder
# - Combines them into one dataset
# - Removes duplicates based on review_id
# - Processes everything together
```

## New Features

### 1. Folder Processing
Point the script at a folder, and it will:
- ✅ Find all CSV files (*.csv)
- ✅ Skip already processed files (processed_*.csv)
- ✅ Load each file with progress tracking
- ✅ Combine all dataframes
- ✅ Remove duplicate reviews (same review_id)
- ✅ Process the complete dataset

### 2. Single File Still Works
```bash
# Still works exactly as before
python process_reviews.py data/captaincoaster_small.csv
```

### 3. Duplicate Removal
When combining multiple progress files:
- Automatically detects duplicate `review_id` values
- Keeps the first occurrence
- Reports how many duplicates were removed

### 4. Progress Tracking
Shows detailed progress during loading:
```
Found 200 CSV files
  ✓ Loaded captaincoaster_reviews_progress_page10_20251110_143459.csv: 100 reviews
  ✓ Loaded captaincoaster_reviews_progress_page20_20251110_143521.csv: 200 reviews
  ...
  Removed 1,234 duplicate reviews
✓ Combined dataset: 19,766 unique reviews
```

## Usage Examples

### Process All Scraped Data
```bash
cd crawler
python process_reviews.py data
```

This will:
1. Load all 200 progress CSV files from your crawl
2. Combine ~20,000 reviews
3. Remove duplicates
4. Clean, analyze sentiment, engineer features
5. Export ML-ready dataset

### Process Specific Folder
```bash
python process_reviews.py path/to/folder/
```

### Process Single File (as before)
```bash
python process_reviews.py data/captaincoaster_small.csv
```

## Output Files

When processing a folder, outputs are named:
- `processed_all_reviews_TIMESTAMP.csv` - Full dataset
- `processed_all_reviews_ml_TIMESTAMP.csv` - ML-ready (numeric only)
- `processed_all_reviews_metadata_TIMESTAMP.csv` - Names and text
- `processed_all_reviews_info_TIMESTAMP.json` - Statistics

## Real Usage (Current Session)

You just ran:
```bash
python process_reviews.py data
```

This is now processing:
- **200 CSV files** from your crawler
- **~20,000 reviews** total (after deduplication)
- Estimated time: 2-3 minutes for full processing

Once complete, you'll have a **massive ML-ready dataset** with:
- 20,000 reviews
- Sentiment analysis on all
- 14+ engineered features
- Ready for training powerful ML models

## Benefits

### Before (Manual Combining)
```python
# Had to manually write code to:
import glob
files = glob.glob('data/*.csv')
dfs = [pd.read_csv(f) for f in files]
combined = pd.concat(dfs)
combined = combined.drop_duplicates(subset=['review_id'])
# Then process...
```

### After (Automatic)
```bash
# Just one command!
python process_reviews.py data
```

## Next Steps

Once processing completes, you'll have ML-ready data for:
1. **Training regression models** (predict rating from features)
2. **Sentiment classification** (classify review sentiment)
3. **Tag prediction** (predict which tags apply)
4. **Recommendation systems** (recommend coasters to users)
5. **Integration with Streamlit app** (better thrill predictions)

---

**The script is currently running and processing your 20,000 reviews! 🚀**
