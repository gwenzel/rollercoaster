"""Quick test of the rating distribution crawler"""

import sys
sys.path.insert(0, r'c:\Users\Lenovo\rollercoaster\crawler\captaincoaster')

from rating_distribution_crawler import RatingDistributionCrawler
import json

# Test with Voltron (ID 5403)
crawler = RatingDistributionCrawler(delay=1.0)

print("Testing with Voltron (ID 5403)...")
data = crawler.extract_rating_distribution(
    coaster_id=5403, 
    coaster_name="Voltron",
    coaster_slug="voltron-europa-park"
)

if data:
    print("\n✓ Success!")
    print(json.dumps(data, indent=2))
    
    # Check key fields
    print(f"\nKey results:")
    print(f"  Average rating: {data.get('avg_rating')} stars")
    print(f"  Total ratings: {data.get('total_ratings')}")
    print(f"  5★ percentage: {data.get('pct_5.0_stars')}%")
    print(f"  4.5★ percentage: {data.get('pct_4.5_stars')}%")
else:
    print("✗ Failed to extract data")
