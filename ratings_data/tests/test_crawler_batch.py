"""Test rating distribution crawler with first 5 coasters"""

import sys
sys.path.insert(0, r'c:\Users\Lenovo\rollercoaster\crawler\captaincoaster')

from rating_distribution_crawler import RatingDistributionCrawler
import pandas as pd

print("=" * 60)
print("Testing Rating Distribution Crawler - First 5 Coasters")
print("=" * 60)

crawler = RatingDistributionCrawler(delay=1.0)

ratings_csv = r"c:\Users\Lenovo\rollercoaster\ratings_data\processed_all_reviews_metadata_20251110_161035.csv"
url_mapping_csv = r"c:\Users\Lenovo\rollercoaster\coaster_id_to_url_mapping.csv"

df = crawler.scrape_coasters_from_csv(
    ratings_csv=ratings_csv,
    url_mapping_csv=url_mapping_csv,
    output_csv="test_rating_distributions_5coasters.csv",
    start_index=0,
    end_index=5,
    save_interval=2
)

print("\n" + "=" * 60)
print("Test Results")
print("=" * 60)

if not df.empty:
    print(f"\nSuccessfully scraped {len(df)} coasters")
    print(f"Coasters with ratings: {df['avg_rating'].notna().sum()}")
    print("\nSummary:")
    print(df[['coaster_name', 'avg_rating', 'total_ratings', 'pct_5.0_stars']].to_string(index=False))
else:
    print("No data collected")
