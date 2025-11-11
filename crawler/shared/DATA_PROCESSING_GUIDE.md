# Review Data Processing - Quick Start

## What We Built

A complete data processing pipeline for Captain Coaster reviews that:
1. ✅ **Cleans data** - Removes URLs, handles missing values
2. ✅ **Analyzes sentiment** - Classifies reviews as positive/neutral/negative  
3. ✅ **Engineers features** - Creates ML-friendly numeric features
4. ✅ **Exports multiple formats** - CSV for ML, metadata separately, JSON stats

## Files Created

### Main Scripts
- **`process_reviews.py`** - Main processing pipeline
- **`visualize_data.py`** - Generate charts and statistics
- **`PROCESSING_README.md`** - Complete documentation

### Data Location
All data files are in the `crawler/data/` directory.

## Quick Usage

### 1. Process Reviews

From the `crawler` directory:

```bash
# Process a specific file
python process_reviews.py data/captaincoaster_small.csv

# Process the latest scrape
python process_reviews.py data/captaincoaster_reviews_progress_page2000_20251110_153829.csv
```

### 2. Visualize Results

```bash
# Visualize the processed data
python visualize_data.py data/processed_captaincoaster_small_20251110_155805.csv
```

This generates:
- 9 charts showing rating distribution, sentiment, correlations, etc.
- Summary statistics printed to console
- PNG file saved with visualizations

### 3. Load for ML

```python
import pandas as pd

# Load ML-ready data (numeric features only)
ml_data = pd.read_csv('data/processed_captaincoaster_small_ml_20251110_155805.csv')

# Separate features and target
X = ml_data.drop(['rating', 'review_id', 'coaster_id', 'park_id'], axis=1)
y = ml_data['rating']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
```

## What Gets Generated

After running `process_reviews.py`, you get 4 files:

### 1. Full Processed Data
**`processed_*_TIMESTAMP.csv`**
- All features including text
- Sentiment scores and categories
- Engineered features

### 2. ML-Ready Data ⭐
**`processed_*_ml_TIMESTAMP.csv`**
- **14 numeric features** ready for machine learning
- No text or URLs
- Perfect for sklearn, tensorflow, pytorch

**Features included:**
- `coaster_id`, `park_id`, `review_id` (IDs)
- `rating`, `rating_normalized` (target variables)
- `upvotes`, `upvotes_log` (engagement)
- `sentiment_score` (sentiment analysis)
- `review_length`, `review_word_count` (text stats)
- `num_positive_tags`, `num_negative_tags`, `tag_balance` (tags)
- `engagement_score` (combined metric)

### 3. Metadata
**`processed_*_metadata_TIMESTAMP.csv`**
- Coaster names, park names, reviewer names
- Review text
- Tags (as text)
- Use to join back after predictions

### 4. Feature Info
**`processed_*_info_TIMESTAMP.json`**
- Dataset statistics
- Feature list and descriptions
- Processing metadata

## Sentiment Analysis

The processor analyzes review text sentiment using a rule-based approach:

**Sentiment Score**: -1 (very negative) to +1 (very positive)

**Categories**:
- **Positive**: Score > 0.02 (enthusiastic language)
- **Neutral**: -0.02 ≤ Score ≤ 0.02 (balanced)
- **Negative**: Score < -0.02 (critical language)

**Example Results** (from 500 reviews):
- 46.9% Positive
- 42.0% Neutral  
- 11.2% Negative
- Average sentiment: 0.028 (slightly positive)

## Feature Engineering

### Text Features
- `review_length`: Total characters
- `review_word_count`: Total words
- Helps identify detailed vs. brief reviews

### Tag Features
- `num_positive_tags`: Count of positive attributes (Theming, Fun, etc.)
- `num_negative_tags`: Count of negative attributes (Rattle, Rough, etc.)
- `tag_balance`: Positive - Negative (overall sentiment from tags)

### Engagement Metrics
- `rating_normalized`: Rating scaled 0-1 (for neural networks)
- `upvotes_log`: Log-transformed upvotes (reduces skew)
- `engagement_score`: 70% rating + 30% log(upvotes) - combined quality metric

### One-Hot Encoded Tags
For common tags appearing in >1% of reviews:
- `tag_pos_theming`, `tag_pos_fun`, `tag_pos_intensity`, etc.
- `tag_neg_rattle`, `tag_neg_rough`, `tag_neg_headbanging`, etc.

## Next Steps: Train a Model

### Simple Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv('data/processed_captaincoaster_small_ml_20251110_155805.csv')

# Prepare features and target
feature_cols = ['sentiment_score', 'review_word_count', 'num_positive_tags', 
                'num_negative_tags', 'tag_balance', 'upvotes_log']
X = df[feature_cols]
y = df['rating']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f} stars")
print(f"R² Score: {r2:.3f}")

# Feature importance
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(importance_df)
```

### Integration with Streamlit App

Use the trained model to improve your roller coaster designer:

```python
# In utils/model.py
import joblib
import pandas as pd

# Load trained model
ml_model = joblib.load('models/coaster_rating_model.pkl')

def predict_thrill(features):
    """
    Predict thrill using ML model trained on real reviews
    
    Args:
        features: dict with coaster characteristics
    
    Returns:
        float: Predicted rating (0-10 scale)
    """
    # Convert features to DataFrame
    feature_df = pd.DataFrame([{
        'max_height': features['max_height'],
        'max_slope': features['max_slope'],
        'loop_radius': 1.0 / features['loop_radius'],
        'num_hills': features['num_hills'],
        'total_length': features['total_length'],
        'mean_curvature': features['mean_curvature']
    }])
    
    # Predict (assuming model outputs 0-5 rating)
    predicted_rating = ml_model.predict(feature_df)[0]
    
    # Scale to 0-10
    return predicted_rating * 2.0
```

## Current Dataset

Based on your crawl, you have:
- **~20,000 reviews** collected so far
- Across multiple files in `data/` directory
- Can combine them all for training

### Combine All Progress Files

```python
import pandas as pd
import glob

# Find all progress files
files = glob.glob('data/captaincoaster_reviews_progress_*.csv')

# Load and combine
all_reviews = []
for file in files:
    df = pd.read_csv(file)
    all_reviews.append(df)

combined = pd.concat(all_reviews, ignore_index=True)

# Remove duplicates (same review_id)
combined = combined.drop_duplicates(subset=['review_id'])

# Save combined dataset
combined.to_csv('data/captaincoaster_all_reviews.csv', index=False)

print(f"Combined {len(combined)} unique reviews from {len(files)} files")
```

Then process:
```bash
python process_reviews.py data/captaincoaster_all_reviews.csv
```

## Tips for Better Models

1. **More data = Better models**: Process all 20,000+ reviews
2. **Feature selection**: Not all features are useful - try different combinations
3. **Hyperparameter tuning**: Use GridSearchCV or RandomizedSearchCV
4. **Cross-validation**: Use k-fold CV for robust evaluation
5. **Ensemble methods**: Combine multiple models (Random Forest, XGBoost, LightGBM)
6. **Deep learning**: Try neural networks for text + numeric features

## Troubleshooting

**Problem**: No tags found  
**Solution**: Small datasets may not have enough reviews with tags

**Problem**: Sentiment all neutral  
**Solution**: Simple lexicon-based approach. Upgrade to VADER or BERT for better results

**Problem**: Low correlation between sentiment and rating  
**Solution**: Normal - people express satisfaction differently in text vs. stars

**Problem**: Memory issues with large files  
**Solution**: Process in chunks or use more RAM

## Summary

You now have:
✅ A complete data cleaning pipeline  
✅ Sentiment analysis on all reviews  
✅ 14 ML-ready features  
✅ Visualization tools  
✅ ~20,000 reviews ready for training  

**Next**: Train ML models and integrate with your Streamlit app! 🚀

---

For detailed documentation, see `PROCESSING_README.md`
