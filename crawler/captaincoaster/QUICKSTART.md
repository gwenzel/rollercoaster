# Captain Coaster Crawler - Quick Start Guide

## Files Created

1. **`crawler.py`** - Main crawler script with full functionality
2. **`test_crawler.py`** - Test script to verify crawler works
3. **`quick_scrape.py`** - Easy-to-use preset scraper
4. **`debug_structure.py`** - HTML structure debugging tool
5. **`requirements_crawler.txt`** - Python dependencies
6. **`CRAWLER_README.md`** - Full documentation

## Quick Start Commands

### 1. Test the Crawler (Recommended First Step)
```bash
python test_crawler.py
```
This scrapes 1 page to verify everything works correctly.

### 2. Quick Scrape with Presets
```bash
# Test mode (10 pages, ~100 reviews)
python quick_scrape.py test

# Small dataset (50 pages, ~500 reviews)
python quick_scrape.py small

# Medium dataset (200 pages, ~2000 reviews)
python quick_scrape.py medium

# Large dataset (1000 pages, ~10,000 reviews)
python quick_scrape.py large

# Full scrape (ALL 7,399 pages, ~74,000 reviews)
python quick_scrape.py full
```

### 3. Full Control with Main Crawler
```bash
python crawler.py
```
Interactive menu with options to scrape all pages or custom ranges.

## What Data Gets Collected?

Each review contains:
- Roller coaster name, ID, and URL
- Theme park name, ID, and URL  
- Reviewer name, ID, and profile URL
- Star rating (0-5 scale with half stars)
- Review text
- Positive tags (e.g., "Theming", "Ejectors")
- Negative tags (e.g., "Rattle", "Rough")
- Number of upvotes
- Time posted
- Unique review ID
- Timestamp of when data was scraped

## Output Files

The crawler generates:
- **CSV file**: `captaincoaster_reviews_TIMESTAMP.csv`
- **JSON file**: `captaincoaster_reviews_TIMESTAMP.json`
- **Progress files**: Auto-saved every 10 pages during long scrapes

## Usage Examples

### Example 1: Quick Test
```bash
python test_crawler.py
```
Output: `test_reviews.csv` and `test_reviews.json`

### Example 2: Get Sample Data for Analysis
```bash
python quick_scrape.py medium
```
Output: `captaincoaster_medium.csv` (2,000 reviews in ~6 minutes)

### Example 3: Full Database Scrape
```bash
python quick_scrape.py full
```
Output: `captaincoaster_full.csv` (~74,000 reviews in ~3 hours)

## Data Analysis Ideas

Once you have the data, you can:

1. **Train ML Models**
   - Predict ratings based on tags
   - Classify reviews as positive/negative
   - Recommend similar coasters

2. **Statistical Analysis**
   - Top-rated coasters by park
   - Most reviewed attractions
   - Correlation between features and ratings

3. **Visualizations**
   - Rating distributions
   - Tag frequency charts
   - Geographic analysis of parks

4. **Integration with Your App**
   - Use real review data to improve thrill predictions
   - Train better models for your roller coaster designer
   - Compare generated coasters to real-world favorites

## Important Notes

⚠️ **Respect the Website**
- Default 1-second delay between requests (be patient!)
- Don't hammer the server with multiple simultaneous runs
- Consider running during off-peak hours for large scrapes

⚠️ **Data Usage**
- This data is for personal/educational use
- Credit Captain Coaster if publishing analysis
- Follow their terms of service

⚠️ **Time Requirements**
- Test: ~20 seconds
- Small: ~2 minutes
- Medium: ~6 minutes
- Large: ~30 minutes
- Full: ~3 hours

## Troubleshooting

**Can't run Python scripts?**
- Use: `python script_name.py`
- Or with conda: `C:/Users/Lenovo/anaconda3/Scripts/conda.exe run -p C:\Users\Lenovo\anaconda3 --no-capture-output python script_name.py`

**Getting connection errors?**
- Check your internet connection
- Wait a few minutes and try again
- Increase delay in crawler.py (change `delay=1.0` to `delay=2.0`)

**Crawler stops unexpectedly?**
- Check the last progress file saved
- Resume from that page number using the main crawler

## Next Steps

1. ✅ Run `test_crawler.py` to verify setup
2. ✅ Try `quick_scrape.py test` for a quick dataset
3. ✅ Analyze the data with pandas/matplotlib
4. ✅ Integrate with your roller coaster app
5. ✅ Share insights or build cool visualizations!

---

For full documentation, see `CRAWLER_README.md`
