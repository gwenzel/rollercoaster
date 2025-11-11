"""
Quick Start: Full Rating Distribution Scrape

Run this script to scrape rating distributions for ALL coasters in the ratings database.
Takes approximately 30-40 minutes for ~1,500 coasters.

Note: Run this script from the ratings_data directory
"""

import sys
import os
sys.path.insert(0, r'c:\Users\Lenovo\rollercoaster\crawler\captaincoaster')

from rating_distribution_crawler import RatingDistributionCrawler
from datetime import datetime

# Change to ratings_data directory if not already there
if not os.getcwd().endswith('ratings_data'):
    os.chdir(r'c:\Users\Lenovo\rollercoaster\ratings_data')

print("=" * 70)
print("FULL RATING DISTRIBUTION SCRAPE")
print("=" * 70)
print()
print("This will scrape rating distributions for ALL coasters in the database.")
print("Estimated time: 30-40 minutes for ~1,500 coasters")
print("Progress will be saved every 50 coasters.")
print()
print("You can press Ctrl+C at any time to stop and save progress.")
print()

# Confirm
confirm = input("Start full scrape? (yes/no): ").strip().lower()

if confirm != 'yes':
    print("Scraping cancelled.")
    sys.exit(0)

print("\n" + "=" * 70)
print("Starting full scrape...")
print("=" * 70)

# Initialize crawler
crawler = RatingDistributionCrawler(delay=1.0)

# File paths
ratings_csv = r"c:\Users\Lenovo\rollercoaster\ratings_data\processed_all_reviews_metadata_20251110_161035.csv"
url_mapping_csv = r"c:\Users\Lenovo\rollercoaster\coaster_id_to_url_mapping.csv"

# Output file (in star_ratings_per_rc subfolder)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_csv = f"star_ratings_per_rc/rating_distributions_full_{timestamp}.csv"

# Start scraping
try:
    df = crawler.scrape_coasters_from_csv(
        ratings_csv=ratings_csv,
        url_mapping_csv=url_mapping_csv,
        output_csv=output_csv,
        start_index=0,
        end_index=None,  # All coasters
        save_interval=50
    )
    
    print("\n" + "=" * 70)
    print("SCRAPING COMPLETE!")
    print("=" * 70)
    
    if not df.empty:
        print(f"\n✓ Total coasters processed: {len(df)}")
        print(f"✓ Coasters with ratings: {df['avg_rating'].notna().sum()}")
        print(f"✓ Coasters with full distributions: {df['pct_5.0_stars'].notna().sum()}")
        print(f"\n✓ Data saved to: {output_csv}")
        
        # Statistics
        if df['avg_rating'].notna().any():
            print(f"\nRating Statistics:")
            print(f"  Mean: {df['avg_rating'].mean():.2f} stars")
            print(f"  Median: {df['avg_rating'].median():.2f} stars")
            print(f"  Min: {df['avg_rating'].min():.2f} stars")
            print(f"  Max: {df['avg_rating'].max():.2f} stars")
            
        print("\n" + "=" * 70)
        print("Top 10 Highest Rated Coasters:")
        print("=" * 70)
        top10 = df.nlargest(10, 'avg_rating')[['coaster_name', 'avg_rating', 'total_ratings', 'pct_5.0_stars']]
        print(top10.to_string(index=False))
        
        print("\n" + "=" * 70)
        print("Top 10 Most Rated Coasters:")
        print("=" * 70)
        most_rated = df.nlargest(10, 'total_ratings')[['coaster_name', 'avg_rating', 'total_ratings']]
        print(most_rated.to_string(index=False))
    else:
        print("\n⚠ No data collected")
        
except KeyboardInterrupt:
    print("\n\n" + "=" * 70)
    print("SCRAPING INTERRUPTED BY USER")
    print("=" * 70)
    print(f"\nProgress has been saved to checkpoint files.")
    print(f"To resume, check the checkpoint files and adjust start_index.")
except Exception as e:
    print(f"\n✗ Error during scraping: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
