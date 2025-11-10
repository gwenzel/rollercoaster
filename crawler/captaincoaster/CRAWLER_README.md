# Captain Coaster Review Crawler

A Python web crawler to scrape all review data from [Captain Coaster](https://captaincoaster.com/en/) - the ultimate guide for roller coaster enthusiasts.

## Overview

This crawler extracts comprehensive review data from Captain Coaster, including:
- ✅ Coaster names and IDs
- ✅ Park names and IDs  
- ✅ Reviewer information
- ✅ Star ratings (0-5 scale)
- ✅ Review text
- ✅ Positive and negative tags (Theming, Intensity, Ejectors, Rattle, etc.)
- ✅ Upvote counts
- ✅ Timestamps
- ✅ All URLs and IDs for further analysis

**Total Reviews Available**: ~73,981 reviews across 7,399 pages

## Features

- 🚀 **Robust error handling** with automatic retries
- 💾 **Automatic progress saving** every 10 pages
- ⏸️ **Interrupt-safe** - Press Ctrl+C to save progress and stop
- 📊 **Multiple export formats** - CSV and JSON
- 🕐 **Respectful crawling** - Configurable delay between requests
- 🔄 **Resume capability** - Can restart from any page
- 📝 **Detailed logging** - Track progress in real-time

## Installation

1. Install required packages:
```bash
pip install -r requirements_crawler.txt
```

Or install manually:
```bash
pip install requests beautifulsoup4 pandas lxml
```

## Usage

### Quick Test (First 10 pages)
```bash
python test_crawler.py
```

This will scrape a single page and create test files to verify everything works.

### Full Crawl

Run the main crawler:
```bash
python crawler.py
```

You'll be presented with options:
1. **Scrape all pages** (1-7399) - ~74k reviews (Warning: Takes several hours!)
2. **Scrape a specific range** - Enter start and end page numbers
3. **Scrape first 10 pages** - Test mode with ~100 reviews

### Programmatic Usage

```python
from crawler import CaptainCoasterCrawler

# Initialize crawler
crawler = CaptainCoasterCrawler(delay=1.0)  # 1 second delay

# Scrape a specific range
df = crawler.scrape_all_reviews(start_page=1, end_page=100)

# Save to files
crawler.save_to_csv(df, "my_reviews.csv")
crawler.save_to_json(df, "my_reviews.json")
```

## Output Format

The crawler generates data with the following fields:

| Field | Description | Example |
|-------|-------------|---------|
| `coaster_name` | Name of the roller coaster | "Voltron" |
| `coaster_id` | Unique coaster ID | "5403" |
| `coaster_url` | Full URL to coaster page | "https://captaincoaster.com/en/coasters/5403/..." |
| `park_name` | Name of the theme park | "Europa Park" |
| `park_id` | Unique park ID | "16" |
| `park_url` | Full URL to park page | "https://captaincoaster.com/en/parks/16/..." |
| `reviewer_name` | Name of the reviewer | "Matias Handley" |
| `reviewer_id` | Unique reviewer username | "matias-handley" |
| `reviewer_url` | Full URL to reviewer profile | "https://captaincoaster.com/en/users/..." |
| `rating` | Star rating (0-5, half stars allowed) | 4.5 |
| `time_posted` | Relative time posted | "3 hours ago" |
| `tags_positive` | Positive attributes (comma-separated) | "Theming, Capacity, Ejectors" |
| `tags_negative` | Negative attributes (comma-separated) | "Rattle" |
| `review_text` | Full review text | "Amazing ride and amazing theming..." |
| `upvotes` | Number of upvotes | 5 |
| `review_id` | Unique review ID | "656511" |
| `scraped_at` | ISO timestamp of when scraped | "2025-11-10T14:29:05.927887" |

## Configuration

You can customize the crawler behavior:

```python
crawler = CaptainCoasterCrawler(
    base_url="https://captaincoaster.com/en/reviews",  # Base URL
    delay=1.0,  # Seconds between requests (be respectful!)
    max_retries=3  # Number of retry attempts for failed requests
)
```

## Progress Saving

The crawler automatically saves progress every 10 pages to files named:
```
captaincoaster_reviews_progress_page{N}_{timestamp}.csv
```

If interrupted, you can resume by:
1. Noting the last completed page from the console output
2. Running the crawler again with that page as the start point

## Output Files

When complete, the crawler generates:
- `captaincoaster_reviews_{timestamp}.csv` - CSV format
- `captaincoaster_reviews_{timestamp}.json` - JSON format

Test files:
- `test_reviews.csv` - Test output from test_crawler.py
- `test_reviews.json` - Test output from test_crawler.py

## Best Practices

1. **Be Respectful**: Keep the default 1-second delay between requests
2. **Start Small**: Test with 10-100 pages first before full crawl
3. **Monitor Progress**: Watch the console for any errors
4. **Save Regularly**: The crawler auto-saves every 10 pages
5. **Check Test First**: Always run `test_crawler.py` before full crawl

## Estimated Time

- **1 page**: ~1-2 seconds
- **10 pages**: ~20 seconds  
- **100 pages**: ~3-4 minutes
- **1000 pages**: ~30-40 minutes
- **Full crawl (7399 pages)**: ~3-4 hours

## Example Data Analysis

Once you have the data, you can analyze it:

```python
import pandas as pd

# Load the data
df = pd.read_csv('captaincoaster_reviews_20251110_142905.csv')

# Top rated coasters
top_coasters = df.groupby('coaster_name')['rating'].mean().sort_values(ascending=False).head(10)

# Most reviewed coasters
most_reviewed = df['coaster_name'].value_counts().head(10)

# Most active reviewers
top_reviewers = df['reviewer_name'].value_counts().head(10)

# Common positive tags
positive_tags = df['tags_positive'].str.split(', ').explode().value_counts()

# Reviews with high upvotes
popular_reviews = df[df['upvotes'] > 5].sort_values('upvotes', ascending=False)
```

## Troubleshooting

**Problem**: `ConnectionError` or timeouts  
**Solution**: Increase delay between requests or check your internet connection

**Problem**: No reviews extracted  
**Solution**: Run `python debug_structure.py` to inspect the HTML structure

**Problem**: Python not found  
**Solution**: Make sure Python is installed and in your PATH, or use the conda command

## Integration with Roller Coaster App

This crawler can be integrated with your existing roller coaster designer app (`app.py`) to:
- Train ML models on real review data
- Predict thrill levels based on actual user reviews
- Analyze what features make coasters highly rated
- Build recommendation systems

## Legal & Ethical Considerations

- 🤝 **Respect robots.txt** and terms of service
- ⏱️ **Rate limiting** is built-in (1 second delay)
- 📊 **Personal use** recommended - don't republish data without permission
- 💬 **Attribution** - Credit Captain Coaster when using this data

## Contributing

Found a bug or want to improve the crawler? Feel free to:
1. Report issues
2. Submit pull requests
3. Suggest new features

## License

This crawler is for educational and personal use. Respect Captain Coaster's terms of service.

---

**Happy Crawling! 🎢**
