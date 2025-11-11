# Review Data Processor Documentation

## Overview

The `process_reviews.py` script cleans and prepares Captain Coaster review data for machine learning. It removes URLs, performs sentiment analysis, engineers features, and exports in ML-friendly formats.

## Features

### 🧹 Data Cleaning
- **Remove URLs**: Strips all URL columns (coaster_url, park_url, reviewer_url)
- **Remove empty reviews**: Filters out reviews with no text content
- **Normalize timestamps**: Categorizes relative times into buckets
- **Handle missing values**: Fills NaN values in tag columns

### 💭 Sentiment Analysis
- **Rule-based approach**: Uses positive/negative word lexicons
- **Sentiment score**: Calculates score from -1 (negative) to +1 (positive)
- **Sentiment category**: Classifies as positive, neutral, or negative
- **Statistics**: Provides sentiment distribution and averages

### 🔧 Feature Engineering
- **Text features**: Review length, word count
- **Tag features**: Count positive/negative tags, calculate balance
- **Rating normalization**: Scale ratings to 0-1
- **Engagement score**: Combines rating and upvotes with log transformation
- **Tag one-hot encoding**: Binary features for common tags

### 📊 Export Formats
- **Full processed CSV**: All features and metadata
- **ML-ready CSV**: Numeric features only (ready for sklearn/tensorflow)
- **Metadata CSV**: Text fields and IDs for reference
- **Feature info JSON**: Dataset statistics and feature descriptions

## Usage

### Basic Usage

```bash
# Process a CSV file
python process_reviews.py data/captaincoaster_small.csv

# Process a large dataset
python process_reviews.py data/captaincoaster_reviews_progress_page2000_20251110_153829.csv
```

### From Python

```python
from process_reviews import ReviewDataProcessor

# Load and process data
processor = ReviewDataProcessor('data/captaincoaster_small.csv')
processed_df, files = processor.process_all()

# Or step by step
processor.load_data()
processor.clean_data()
processor.analyze_sentiment()
processor.engineer_features()
processor.create_tag_features()
processor.get_summary_statistics()
files = processor.save_processed_data()
```

## Generated Files

After processing, you'll get 4 files:

### 1. Full Processed CSV
**File**: `processed_{dataset}_TIMESTAMP.csv`

Contains all features including:
- Original data (without URLs)
- Engineered features
- Sentiment scores
- Tag encodings

### 2. ML-Ready CSV
**File**: `processed_{dataset}_ml_TIMESTAMP.csv`

**Numeric features only** - ready for ML algorithms:
- `coaster_id`: Numeric ID
- `park_id`: Numeric ID  
- `review_id`: Numeric ID
- `rating`: Original rating (0-5)
- `rating_normalized`: Scaled rating (0-1)
- `upvotes`: Number of upvotes
- `upvotes_log`: Log-transformed upvotes
- `sentiment_score`: Sentiment (-1 to +1)
- `review_length`: Character count
- `review_word_count`: Word count
- `num_positive_tags`: Count of positive tags
- `num_negative_tags`: Count of negative tags
- `tag_balance`: Difference between pos/neg tags
- `engagement_score`: Combined rating + upvotes metric
- `tag_pos_*`: Binary features for common positive tags
- `tag_neg_*`: Binary features for common negative tags

### 3. Metadata CSV
**File**: `processed_{dataset}_metadata_TIMESTAMP.csv`

Contains text and identifiers:
- `coaster_name`
- `park_name`
- `reviewer_name`
- `reviewer_id`
- `coaster_id`
- `park_id`
- `review_id`
- `review_text`
- `tags_positive`
- `tags_negative`
- `time_posted`

Use this to join back names/text after ML predictions.

### 4. Feature Info JSON
**File**: `processed_{dataset}_info_TIMESTAMP.json`

Contains metadata about the dataset:
```json
{
  "total_samples": 286,
  "total_features": 25,
  "numeric_features": 14,
  "numeric_feature_names": [...],
  "metadata_features": [...],
  "rating_stats": {
    "mean": 3.21,
    "std": 1.42
  },
  "processing_date": "20251110_155805"
}
```

## Feature Descriptions

### Engineered Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `sentiment_score` | float | -1 to +1 | Calculated from positive/negative word counts |
| `sentiment_category` | string | pos/neu/neg | Categorical sentiment |
| `review_length` | int | 0+ | Total characters in review |
| `review_word_count` | int | 0+ | Total words in review |
| `num_positive_tags` | int | 0+ | Count of positive attribute tags |
| `num_negative_tags` | int | 0+ | Count of negative attribute tags |
| `tag_balance` | int | any | Positive tags minus negative tags |
| `rating_normalized` | float | 0-1 | Rating scaled to 0-1 |
| `upvotes_log` | float | 0+ | Log-transformed upvotes (reduces skew) |
| `engagement_score` | float | 0-1 | 70% rating + 30% log(upvotes) |
| `has_detailed_review` | bool | 0/1 | True if review has >5 words |
| `time_category` | string | various | Time bucket (very_recent, recent, etc.) |

## Sentiment Analysis

### Word Lexicons

**Positive words** (26 total):
amazing, awesome, excellent, fantastic, great, love, best, incredible, wonderful, perfect, epic, brilliant, outstanding, superb, spectacular, thrilling, exhilarating, fun, smooth, intense, legendary, phenomenal, beautiful, gorgeous, blast

**Negative words** (28 total):
bad, terrible, awful, worst, hate, boring, disappointing, rough, painful, uncomfortable, slow, lame, meh, mediocre, garbage, trash, poor, weak, dull, rattle, shake, jerk, headache, hurt, violent, bumpy

### Calculation

```python
sentiment_score = (positive_count - negative_count) / max(total_words, 1)
```

Clipped to range [-1, 1].

**Category thresholds**:
- Positive: score > 0.02
- Negative: score < -0.02
- Neutral: -0.02 ≤ score ≤ 0.02

### Upgrading Sentiment Analysis

For better results, replace with:

**VADER (Valence Aware Dictionary and sEntiment Reasoner)**:
```bash
pip install vaderSentiment
```

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
scores = analyzer.polarity_scores(text)
sentiment = scores['compound']  # -1 to +1
```

**Transformers (BERT-based)**:
```bash
pip install transformers torch
```

```python
from transformers import pipeline

classifier = pipeline('sentiment-analysis')
result = classifier(text)[0]
sentiment = result['score'] if result['label'] == 'POSITIVE' else -result['score']
```

## Example Workflow

### 1. Process Reviews

```bash
cd crawler
python process_reviews.py data/captaincoaster_small.csv
```

### 2. Load ML-Ready Data

```python
import pandas as pd

# Load numeric features for ML
ml_data = pd.read_csv('data/processed_captaincoaster_small_ml_20251110_155805.csv')

# Separate features and target
X = ml_data.drop('rating', axis=1)  # Features
y = ml_data['rating']  # Target variable

print(f"Features: {X.shape}")
print(f"Target: {y.shape}")
```

### 3. Train a Model

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))
```

### 4. Join Back Metadata

```python
# Load metadata for interpretability
metadata = pd.read_csv('data/processed_captaincoaster_small_metadata_20251110_155805.csv')

# Add predictions to metadata
metadata['predicted_rating'] = model.predict(X)
metadata['actual_rating'] = y

# Find biggest prediction errors
metadata['error'] = abs(metadata['predicted_rating'] - metadata['actual_rating'])
worst_predictions = metadata.nlargest(10, 'error')

print("\nWorst Predictions:")
print(worst_predictions[['coaster_name', 'actual_rating', 'predicted_rating', 'review_text']])
```

## ML Model Ideas

### 1. Rating Prediction
**Task**: Predict star rating from review text and features  
**Model**: Random Forest, XGBoost, Neural Network  
**Target**: `rating` or `rating_normalized`  
**Features**: All engineered features + tag encodings

### 2. Sentiment Classification
**Task**: Classify review sentiment  
**Model**: Logistic Regression, SVM, BERT  
**Target**: `sentiment_category`  
**Features**: Text embeddings, word counts, tags

### 3. Coaster Recommendation
**Task**: Recommend coasters based on user preferences  
**Model**: Collaborative Filtering, Matrix Factorization  
**Features**: User-coaster interaction matrix, features

### 4. Review Helpfulness
**Task**: Predict which reviews will get upvotes  
**Model**: Classification or Regression  
**Target**: `upvotes` (binary: >0 or continuous)  
**Features**: Sentiment, length, rating, tags

### 5. Tag Prediction
**Task**: Predict which tags apply to a review  
**Model**: Multi-label Classification  
**Target**: Tag columns  
**Features**: Review text, rating

## Performance Tips

### Large Datasets

For datasets with >10,000 reviews:

1. **Process in chunks**:
```python
chunksize = 1000
for chunk in pd.read_csv(input_file, chunksize=chunksize):
    process_chunk(chunk)
```

2. **Use categorical encoding**:
```python
df['coaster_name'] = df['coaster_name'].astype('category')
```

3. **Save in Parquet format** (faster than CSV):
```python
df.to_parquet('processed_reviews.parquet', compression='gzip')
```

### Memory Optimization

```python
# Downcast numeric types
for col in df.select_dtypes(include=['int']).columns:
    df[col] = pd.to_numeric(df[col], downcast='integer')

for col in df.select_dtypes(include=['float']).columns:
    df[col] = pd.to_numeric(df[col], downcast='float')
```

## Troubleshooting

**Issue**: "No such file or directory"  
**Solution**: Make sure you're in the `crawler` directory or use full paths

**Issue**: No sentiment detected  
**Solution**: The lexicon is simple. Consider using VADER or transformers

**Issue**: No tags found  
**Solution**: Tags might not be present in small datasets

**Issue**: Memory error on large datasets  
**Solution**: Process in chunks or use a machine with more RAM

## Next Steps

1. ✅ Process your review data
2. ✅ Train an ML model
3. ✅ Integrate predictions with your Streamlit app
4. ✅ Use model to predict thrill ratings for generated coasters
5. ✅ Build a recommendation system

---

**Ready to train some models! 🚀**
