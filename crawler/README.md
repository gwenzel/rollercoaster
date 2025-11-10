# Roller Coaster Data Crawlers

This directory contains web scraping tools for collecting roller coaster data from multiple sources.

## 📁 Directory Structure

```
crawler/
├── captaincoaster/      # Captain Coaster review crawler
│   ├── crawler.py       # Main crawler for captaincoaster.com
│   ├── test_crawler.py  # Unit tests
│   ├── quick_scrape.py  # Quick test script
│   └── CRAWLER_README.md
├── rideforcesdb/        # Ride Forces DB G-force crawler
│   ├── rfdb_crawler.py  # Main crawler for rideforcesdb.com
│   ├── rfdb_browse.html # Browser inspection tool
│   └── RFDB_CRAWLER_README.md
├── shared/              # Shared data processing tools
│   ├── process_reviews.py      # Data cleaning & sentiment analysis
│   ├── visualize_data.py       # Data visualization dashboard
│   └── PROCESSING_README.md
└── data/                # Output directory for collected data
```

## 🎯 Quick Start

### 1. Captain Coaster Reviews
Scrape rider reviews and ratings:
```bash
cd captaincoaster
python crawler.py
```

### 2. Ride Forces DB (G-forces)
Collect G-force measurements:
```bash
cd rideforcesdb
python rfdb_crawler.py
```

### 3. Process Collected Data
Clean and analyze reviews:
```bash
cd shared
python process_reviews.py
```

### 4. Visualize Results
Generate charts and insights:
```bash
cd shared
python visualize_data.py
```

## 📊 Data Sources

| Source | Data Type | Records | Fields |
|--------|-----------|---------|---------|
| **Captain Coaster** | Reviews & Ratings | 73,981+ | 17 fields |
| **Ride Forces DB** | G-force Measurements | TBD | Peak forces, duration |

## 🛠️ Installation

Each subfolder has its own `requirements.txt`:

```bash
# Install Captain Coaster dependencies
pip install -r captaincoaster/requirements_crawler.txt

# Install Ride Forces DB dependencies (includes Selenium)
pip install -r rideforcesdb/requirements_rfdb.txt

# Shared tools use standard pandas/matplotlib
pip install pandas matplotlib seaborn textblob
```

## 📖 Documentation

- **Captain Coaster**: See `captaincoaster/CRAWLER_README.md` and `captaincoaster/QUICKSTART.md`
- **Ride Forces DB**: See `rideforcesdb/RFDB_CRAWLER_README.md`
- **Data Processing**: See `shared/PROCESSING_README.md` and `shared/DATA_PROCESSING_GUIDE.md`

## 🎢 Integration with Main App

Collected data can be used to:
1. **Train ML models** - Predict ratings from G-force profiles
2. **Enhance simulations** - Real-world validation of physics models
3. **Recommend coasters** - Sentiment analysis for recommendations
4. **Compare designs** - Benchmark custom coasters against real ones

## 🔄 Workflow

```
1. Scrape Reviews           2. Scrape G-forces
   (captaincoaster/)           (rideforcesdb/)
         ↓                            ↓
    reviews.csv              gforce_data.csv
         ↓                            ↓
         └────────────┬───────────────┘
                      ↓
            3. Combine & Process
               (shared/)
                      ↓
         combined_dataset.csv
                      ↓
            4. Train ML Models
                      ↓
         5. Integrate with app.py
```

## 📝 Notes

- Captain Coaster crawler saves progress every 10 pages
- Ride Forces DB uses Selenium for JavaScript-heavy pages
- All crawlers respect rate limits (1-2 second delays)
- Data is deduplicated by unique IDs before processing
