"""
Quick start script for Captain Coaster crawler
Provides easy presets for common scraping scenarios
"""

from crawler import CaptainCoasterCrawler
import sys

def quick_scrape(preset='test'):
    """
    Quick scrape with presets
    
    Presets:
    - test: First 10 pages (~100 reviews)
    - small: First 50 pages (~500 reviews)
    - medium: First 200 pages (~2000 reviews)
    - large: First 1000 pages (~10,000 reviews)
    - full: All pages (~74,000 reviews)
    """
    
    presets = {
        'test': (1, 10),
        'small': (1, 50),
        'medium': (1, 200),
        'large': (1, 1000),
        'full': (1, 7399)
    }
    
    if preset not in presets:
        print(f"Unknown preset: {preset}")
        print(f"Available presets: {', '.join(presets.keys())}")
        return
    
    start_page, end_page = presets[preset]
    
    print(f"Captain Coaster Quick Scrape - Preset: {preset}")
    print("=" * 60)
    print(f"Pages: {start_page} to {end_page}")
    print(f"Estimated reviews: ~{(end_page - start_page + 1) * 10}")
    print(f"Estimated time: ~{(end_page - start_page + 1) * 1.5 / 60:.1f} minutes")
    print("=" * 60)
    
    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    # Initialize and run crawler
    crawler = CaptainCoasterCrawler(delay=1.0)
    
    print("\nStarting scrape...")
    print("Press Ctrl+C to stop and save progress")
    print("-" * 60)
    
    df = crawler.scrape_all_reviews(start_page=start_page, end_page=end_page, save_interval=10)
    
    if not df.empty:
        # Generate filename with preset name
        csv_file = crawler.save_to_csv(df, f"captaincoaster_{preset}.csv")
        json_file = crawler.save_to_json(df, f"captaincoaster_{preset}.json")
        
        print("\n" + "=" * 60)
        print("✓ Scraping Complete!")
        print("=" * 60)
        print(f"Total reviews: {len(df)}")
        print(f"Unique coasters: {df['coaster_name'].nunique()}")
        print(f"Unique parks: {df['park_name'].nunique()}")
        print(f"Unique reviewers: {df['reviewer_name'].nunique()}")
        print(f"Average rating: {df['rating'].mean():.2f}/5.0")
        print(f"\nFiles saved:")
        print(f"  - {csv_file}")
        print(f"  - {json_file}")
        print("=" * 60)
    else:
        print("No data collected.")

def main():
    print("Captain Coaster Quick Scrape")
    print("=" * 60)
    print("\nAvailable presets:")
    print("  test   - 10 pages (~100 reviews, ~20 seconds)")
    print("  small  - 50 pages (~500 reviews, ~2 minutes)")
    print("  medium - 200 pages (~2,000 reviews, ~6 minutes)")
    print("  large  - 1,000 pages (~10,000 reviews, ~30 minutes)")
    print("  full   - All 7,399 pages (~74,000 reviews, ~3 hours)")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        preset = sys.argv[1]
    else:
        preset = input("\nEnter preset (default: test): ").strip().lower()
        if not preset:
            preset = 'test'
    
    quick_scrape(preset)

if __name__ == "__main__":
    main()
