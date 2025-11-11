# Ride Forces Database Crawler

## Overview

A web crawler for [RideForcesDB.com](https://rideforcesdb.com/) - a database of G-force recordings from roller coasters and other rides.

### Available Data

- **4,623 recordings** of rides with G-force data
- **794 unique rides** (612 coasters)
- **154 theme parks** worldwide
- Force data: Vertical, Lateral, Longitudinal, Combined G-forces
- Recording metadata: Device, date, uploader, seat position
- Peak force statistics
- Downloadable `.zforces` files

## Challenge: JavaScript-Rendered Site

RideForcesDB uses heavy JavaScript rendering, making traditional HTTP scraping difficult. The site loads data dynamically via API calls that need to be discovered.

## Installation

```bash
# Install Selenium for JavaScript rendering
pip install -r requirements_rfdb.txt

# Install ChromeDriver
# Download from: https://chromedriver.chromium.org/
# Or use: pip install webdriver-manager
```

## Usage

### Option 1: Discover API Endpoints (Recommended)

The site likely uses API endpoints that can be accessed directly. Use browser DevTools to find them:

```bash
python rfdb_crawler.py
# Choose option 2: "Show site inspection guide"
```

**Steps:**
1. Open https://rideforcesdb.com/browse in Chrome
2. Press `F12` → Go to **Network** tab
3. Refresh the page
4. Filter by **Fetch/XHR** requests
5. Look for JSON responses with ride data
6. Copy the API endpoint URL
7. Run crawler with option 3 and paste the URL

### Option 2: Automatic Crawling

```bash
python rfdb_crawler.py
# Choose option 1: "Try automatic crawling"
```

Uses Selenium to render JavaScript and extract data.

### Option 3: Manual API Entry

```bash
python rfdb_crawler.py
# Choose option 3: "Manual API endpoint entry"
# Enter the API URL you found
```

## Example API Patterns

Common patterns for this type of site:

```
https://rideforcesdb.com/api/rides
https://rideforcesdb.com/api/recordings
https://rideforcesdb.com/api/browse?limit=100
https://rideforcesdb.com/graphql
```

## Data Structure (Expected)

### Ride Information
```json
{
  "id": "123",
  "name": "Steel Vengeance",
  "park": "Cedar Point",
  "manufacturer": "Rocky Mountain Construction",
  "type": "Roller Coaster",
  "recordings_count": 45
}
```

### Recording Metadata
```json
{
  "recording_id": "456",
  "ride_id": "123",
  "upload_date": "2025-11-10",
  "device": "iPhone 14 Pro",
  "uploader": "Anonymous",
  "seat": "Front Row",
  "duration": "2m 15s",
  "peak_vertical": "4.5g",
  "peak_lateral": "1.8g",
  "peak_longitudinal": "1.2g"
}
```

### Force Data
The `.zforces` files contain timestamped G-force measurements:
```
Time (s), Vertical (g), Lateral (g), Longitudinal (g)
0.00, 1.0, 0.0, 0.0
0.01, 1.1, 0.1, -0.2
...
```

## Programmatic Usage

```python
from rfdb_crawler import RideForcesDBCrawler

# Initialize crawler
crawler = RideForcesDBCrawler(use_selenium=True)

# Scrape all data
df = crawler.scrape_all_data()

# Download specific recording
crawler.download_recording('recording_id_123')

# Close when done
crawler.close()
```

## Finding the Right API

### Method 1: Browser DevTools

1. **Open the site**:
   ```
   https://rideforcesdb.com/browse
   ```

2. **Open DevTools** (F12)

3. **Network Tab** → Filter: `Fetch/XHR`

4. **Interact with the site**:
   - Click "Browse"
   - Scroll through rides
   - Click on a recording

5. **Look for API calls**:
   - Endpoint URLs (usually contain `/api/`)
   - Response type: JSON
   - Status: 200

6. **Example findings**:
   ```
   GET https://rideforcesdb.com/api/rides?limit=100&offset=0
   Response: [{"id": 1, "name": "...", ...}, ...]
   ```

### Method 2: Inspect Element

1. Right-click on a ride → **Inspect**
2. Look for `data-*` attributes:
   ```html
   <div data-ride-id="123" data-ride-name="Steel Vengeance">
   ```
3. Use these IDs to construct API URLs

### Method 3: Check JavaScript Files

1. DevTools → **Sources** tab
2. Look for `.js` files
3. Search for `fetch(`, `axios`, or API URLs
4. Find endpoint patterns

## Output Files

### Automatic Output
- `rfdb_rides_TIMESTAMP.csv` - All rides data
- `rfdb_recordings_TIMESTAMP.csv` - All recordings metadata
- `rfdb_browse.html` - Saved page HTML for inspection
- `data/rfdb_recordings/*.zforces` - Downloaded force recordings

### Manual API Output
- `rfdb_data_TIMESTAMP.json` - Raw JSON from API

## Data Processing

Once you have the data:

```python
import pandas as pd

# Load rides
rides = pd.read_csv('rfdb_rides_20251110_120000.csv')

# Merge with Captain Coaster data for enhanced analysis
captain_coaster = pd.read_csv('processed_all_reviews_ml_TIMESTAMP.csv')

# Join by coaster name (fuzzy matching may be needed)
combined = pd.merge(
    rides, 
    captain_coaster, 
    left_on='name', 
    right_on='coaster_name', 
    how='inner'
)

# Now you have both G-force data AND ratings!
```

## Integration with ML Models

Use G-force data to improve your roller coaster designer:

```python
# Features from RFDB
- peak_vertical_g
- peak_lateral_g  
- peak_longitudinal_g
- duration_seconds
- num_inversions
- max_speed

# Combine with Captain Coaster features
- rating
- sentiment_score
- num_positive_tags
- num_negative_tags

# Train enhanced model
from sklearn.ensemble import RandomForestRegressor

X = combined[['peak_vertical_g', 'peak_lateral_g', 'duration_seconds', 
              'sentiment_score', 'num_positive_tags']]
y = combined['rating']

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
```

## Troubleshooting

### Selenium Not Working

```bash
# Install webdriver-manager for automatic ChromeDriver management
pip install webdriver-manager

# Update crawler to use it:
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)
```

### No Data Found

- Site might use authentication - check if login is needed
- API might have rate limiting
- Try different User-Agent headers
- Check if site has robots.txt restrictions

### Connection Timeout

```python
# Increase timeout
crawler = RideForcesDBCrawler(delay=5.0)  # 5 second delay
```

## Legal & Ethical Considerations

- **Check robots.txt**: https://rideforcesdb.com/robots.txt
- **Respect rate limits**: Built-in 2-second delay
- **Privacy**: Recordings can be anonymous - respect that
- **Terms of Service**: Read before scraping
- **Attribution**: Credit RideForcesDB when using data
- **Personal use**: Recommended for research/education

## Next Steps

1. **Find API endpoints** using browser DevTools
2. **Update crawler** with correct URLs
3. **Download all recordings**
4. **Process force data** into ML features
5. **Combine with Captain Coaster** reviews
6. **Train enhanced models** with G-force + sentiment data
7. **Integrate into Streamlit app** for realistic predictions

## Comparison: RFDB vs Captain Coaster

| Feature | Captain Coaster | Ride Forces DB |
|---------|----------------|----------------|
| **Type** | User reviews & ratings | G-force measurements |
| **Data** | Text, ratings, tags | Time-series force data |
| **Quantity** | 73,981 reviews | 4,623 recordings |
| **Coverage** | Global, extensive | Growing database |
| **Format** | Structured text | Numerical physics |
| **Use Case** | Sentiment, preferences | Actual ride forces |

**Combined**: Most powerful dataset for ML! 🎢

---

For implementation help, see the inspection guide:
```bash
python rfdb_crawler.py
# Choose option 2
```
