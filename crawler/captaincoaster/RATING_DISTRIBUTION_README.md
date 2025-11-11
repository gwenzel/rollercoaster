# Captain Coaster Rating Distribution Crawler

## Overview

This crawler extracts **detailed rating distributions** for rollercoasters from Captain Coaster individual coaster pages. Unlike the review crawler that gets individual reviews, this tool collects aggregate statistics showing how ratings are distributed across all rating levels.

## What It Collects

For each rollercoaster, the crawler extracts:

### Core Metrics
- **Average Rating**: Mean rating (0-5 stars)
- **Total Ratings**: Number of users who rated the coaster

### Distribution Data
For each rating level (0.5★, 1★, 1.5★, ..., 5★):
- **Percentage**: What % of users gave this rating
- **Count**: Absolute number of users who gave this rating

### Example Output
```json
{
  "coaster_id": 5403,
  "coaster_name": "Voltron",
  "url": "https://captaincoaster.com/en/coasters/5403/voltron-europa-park",
  "avg_rating": 4.69,
  "total_ratings": 2290,
  "pct_0.5_stars": 0.31,
  "pct_1.0_stars": 0.39,
  ...
  "pct_5.0_stars": 70.26,
  "count_0.5_stars": 7,
  "count_1.0_stars": 9,
  ...
  "count_5.0_stars": 1609
}
```

## How It Works

### Data Extraction Method

Captain Coaster embeds rating distribution data in JavaScript (ApexCharts configuration). The crawler:

1. Fetches the coaster page HTML
2. Parses embedded JavaScript to extract rating counts
3. Calculates percentages and average rating
4. Returns structured data

### URL Requirements

Captain Coaster coaster pages use slugified URLs:
```
https://captaincoaster.com/en/coasters/{id}/{slug}
```

For example:
- ID: 5403
- Slug: `voltron-europa-park`
- Full URL: `https://captaincoaster.com/en/coasters/5403/voltron-europa-park`

The crawler uses the `coaster_id_to_url_mapping.csv` file (generated from scraped review data) to get correct slugs.

## Files

### Main Script
- **`rating_distribution_crawler.py`**: Main crawler implementation

### Helper Scripts
- **`create_coaster_url_mapping.py`**: Extracts coaster ID→URL mapping from review data
- **`test_rating_crawler.py`**: Simple test script for single coaster

### Data Files
- **`coaster_id_to_url_mapping.csv`**: Maps 1,500 coaster IDs to their URLs
- **`ratings_data/processed_all_reviews_metadata_*.csv`**: Source of coaster IDs

## Usage

### Prerequisites

1. **Install dependencies**:
   ```bash
   pip install requests beautifulsoup4 pandas
   ```

2. **Generate URL mapping** (if not already done):
   ```bash
   python create_coaster_url_mapping.py
   ```

### Running the Crawler

#### Interactive Mode

```bash
cd c:\Users\Lenovo\rollercoaster\crawler\captaincoaster
python rating_distribution_crawler.py
```

Options:
1. **Scrape all coasters** - Process all ~1,500 coasters (takes several hours)
2. **Scrape a range** - Specify start/end indices
3. **Test mode** - Scrape first 10 coasters only
4. **Single coaster** - Test with one specific coaster

#### Programmatic Use

```python
from rating_distribution_crawler import RatingDistributionCrawler

# Initialize crawler
crawler = RatingDistributionCrawler(delay=1.0)

# Scrape batch of coasters
df = crawler.scrape_coasters_from_csv(
    ratings_csv="path/to/ratings_data.csv",
    url_mapping_csv="path/to/url_mapping.csv",
    output_csv="rating_distributions.csv",
    start_index=0,
    end_index=100,
    save_interval=50
)

# Scrape single coaster
data = crawler.extract_rating_distribution(
    coaster_id=5403,
    coaster_name="Voltron",
    coaster_slug="voltron-europa-park"
)
```

### Output Files

#### Main Output
- **`rating_distributions_TIMESTAMP.csv`**: Final results with all coasters

#### Checkpoint Files
- **`rating_distributions_checkpoint_N_TIMESTAMP.csv`**: Saved every 50 coasters

Format:
```csv
coaster_id,coaster_name,url,avg_rating,total_ratings,pct_0.5_stars,...,pct_5.0_stars,count_0.5_stars,...,count_5.0_stars,scraped_at
5403,Voltron,https://...,4.69,2290,0.31,...,70.26,7,...,1609,2025-11-11T15:56:19
```

## Features

### Robust Scraping
- **JavaScript parsing**: Extracts data from embedded ApexCharts config
- **Error handling**: Continues on failures, saves progress
- **Rate limiting**: Configurable delay between requests (default: 1s)

### Progress Tracking
- **Real-time feedback**: Shows progress for each coaster
- **Checkpoint saves**: Saves every N coasters (default: 50)
- **Resume capability**: Can continue from checkpoints

### Data Quality
- **Calculated metrics**: Computes averages from raw counts
- **Both formats**: Provides percentages AND absolute counts
- **Validation**: Checks rating ranges and data consistency

## Data Structure

### CSV Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `coaster_id` | int | Captain Coaster ID | 5403 |
| `coaster_name` | str | Coaster name | "Voltron" |
| `url` | str | Full coaster page URL | "https://..." |
| `avg_rating` | float | Average rating (0-5) | 4.69 |
| `total_ratings` | int | Total number of ratings | 2290 |
| `pct_0.5_stars` | float | % of 0.5★ ratings | 0.31 |
| `pct_1.0_stars` | float | % of 1★ ratings | 0.39 |
| ... | ... | ... | ... |
| `pct_5.0_stars` | float | % of 5★ ratings | 70.26 |
| `count_0.5_stars` | int | Number of 0.5★ ratings | 7 |
| `count_1.0_stars` | int | Number of 1★ ratings | 9 |
| ... | ... | ... | ... |
| `count_5.0_stars` | int | Number of 5★ ratings | 1609 |
| `scraped_at` | str | Timestamp (ISO format) | "2025-11-11T15:56:19" |

### Rating Levels

The crawler extracts data for 10 rating levels:
- 0.5★, 1.0★, 1.5★, 2.0★, 2.5★, 3.0★, 3.5★, 4.0★, 4.5★, 5.0★

## Performance

### Timing
- **Single coaster**: ~1 second (with delay)
- **100 coasters**: ~2 minutes
- **1,500 coasters**: ~30-40 minutes

### Rate Limiting
Default delay: 1.0 seconds between requests

Adjust if needed:
```python
crawler = RatingDistributionCrawler(delay=2.0)  # 2 seconds
```

## Error Handling

### Automatic Recovery
- **Network errors**: Logs and continues to next coaster
- **Parse errors**: Catches and reports, saves progress
- **Keyboard interrupt**: Saves checkpoint before exiting

### Missing Data
If rating distribution not found:
- Still captures coaster ID/name
- Sets rating fields to `None`
- Continues scraping

## Use Cases

### Machine Learning
- **Training data**: Use distributions as features
- **Rating prediction**: Train models to predict rating curves
- **Popularity analysis**: Analyze distribution shapes

### Analysis
- **Distribution patterns**: Compare rating curves across coasters
- **Consensus metrics**: Identify polarizing vs universally loved rides
- **Quality indicators**: Analyze correlation between distribution shape and average rating

### Integration
- **Combine with reviews**: Merge with individual review data
- **Track data join**: Match with RFDB accelerometer data via `coaster_id`
- **Time series**: Re-scrape periodically to track rating changes

## Example Analysis

```python
import pandas as pd

# Load rating distributions
df = pd.read_csv('rating_distributions_20251111.csv')

# Top-rated coasters
top_rated = df.nlargest(10, 'avg_rating')[['coaster_name', 'avg_rating', 'total_ratings']]
print(top_rated)

# Polarizing coasters (high variance)
df['variance'] = (
    (df['pct_0.5_stars'] + df['pct_1.0_stars'] + df['pct_1.5_stars']) * 
    (df['pct_4.5_stars'] + df['pct_5.0_stars'])
)
polarizing = df.nlargest(10, 'variance')[['coaster_name', 'avg_rating', 'variance']]
print(polarizing)

# Consensus favorites (low variance, high rating)
consensus = df[(df['pct_5.0_stars'] > 70) & (df['avg_rating'] > 4.5)]
print(consensus[['coaster_name', 'avg_rating', 'pct_5.0_stars', 'total_ratings']])
```

## Maintenance

### Updating URL Mapping

If new coasters are added to review data:

```bash
python create_coaster_url_mapping.py
```

This regenerates `coaster_id_to_url_mapping.csv` from latest crawler data.

### Validating Output

Check for missing data:
```python
df = pd.read_csv('rating_distributions.csv')
print(f"Coasters with ratings: {df['avg_rating'].notna().sum()}/{len(df)}")
print(f"Coasters with distributions: {df['pct_5.0_stars'].notna().sum()}/{len(df)}")
```

## Limitations

### Coverage
- Requires coaster URL slugs (from previous crawler data)
- Only processes coasters in `ratings_data` CSV
- May miss newly added coasters

### Data Accuracy
- Dependent on Captain Coaster's JavaScript structure
- If site changes ApexCharts format, parser may need updates
- No historical data - only current ratings

### Rate Limits
- Respectful delays required to avoid overwhelming server
- Large-scale scraping takes time
- No parallel requests to be server-friendly

## Troubleshooting

### "Page not found" errors
- **Cause**: Missing or incorrect URL slug
- **Fix**: Ensure `coaster_id_to_url_mapping.csv` is up-to-date
- **Workaround**: Provide correct slug manually for single coaster

### No rating data extracted
- **Cause**: JavaScript structure changed on website
- **Fix**: Update regex patterns in `extract_rating_distribution()`
- **Debug**: Check `test_coaster_page.html` for current structure

### Slow performance
- **Cause**: Network latency or rate limiting
- **Fix**: Increase `delay` parameter
- **Alternative**: Scrape in smaller batches

## Future Enhancements

### Potential Improvements
- [ ] Add caching to avoid re-scraping
- [ ] Parallel requests with rate limiting
- [ ] Auto-detect URL slug from coaster name
- [ ] Historical tracking (scrape periodically, compare changes)
- [ ] Visualization of distribution curves
- [ ] API wrapper for programmatic access

### Integration Opportunities
- [ ] Merge with BiGRU training pipeline
- [ ] Use distributions as model features
- [ ] Compare predicted vs actual rating distributions

## Credits

Built for the rollercoaster rating prediction project. Complements existing review crawler by providing aggregate statistics for model training and analysis.

---

**Last Updated**: 2025-11-11  
**Version**: 1.0
