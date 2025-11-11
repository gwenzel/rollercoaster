"""
Example usage script demonstrating the Captain Coaster crawler
"""

from crawler import CaptainCoasterCrawler
import pandas as pd

def example_basic_usage():
    """Example 1: Basic crawling"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Create crawler instance
    crawler = CaptainCoasterCrawler(delay=1.0)
    
    # Scrape first 5 pages
    df = crawler.scrape_all_reviews(start_page=1, end_page=5)
    
    # Save results
    crawler.save_to_csv(df, "example_basic.csv")
    
    print(f"\nCollected {len(df)} reviews")
    print(f"Saved to example_basic.csv")


def example_data_analysis():
    """Example 2: Quick data analysis"""
    print("\n" + "=" * 60)
    print("Example 2: Data Analysis")
    print("=" * 60)
    
    # Load existing data (from test)
    df = pd.read_csv("test_reviews.csv")
    
    print(f"\nDataset Info:")
    print(f"  Total reviews: {len(df)}")
    print(f"  Unique coasters: {df['coaster_name'].nunique()}")
    print(f"  Unique reviewers: {df['reviewer_name'].nunique()}")
    print(f"  Average rating: {df['rating'].mean():.2f}")
    
    print(f"\nRating Distribution:")
    print(df['rating'].value_counts().sort_index())
    
    print(f"\nTop Reviewed Coasters:")
    print(df['coaster_name'].value_counts().head(5))
    
    # Reviews with positive tags
    if df['tags_positive'].notna().any():
        print(f"\nReviews with positive tags:")
        print(df[df['tags_positive'].notna()][['coaster_name', 'tags_positive']].head())


def example_filtered_scraping():
    """Example 3: Custom scraping with filtering"""
    print("\n" + "=" * 60)
    print("Example 3: Scrape and Filter")
    print("=" * 60)
    
    crawler = CaptainCoasterCrawler(delay=1.0)
    
    # Scrape pages
    df = crawler.scrape_all_reviews(start_page=1, end_page=3)
    
    # Filter high-rated reviews only
    high_rated = df[df['rating'] >= 4.0]
    
    print(f"\nTotal reviews: {len(df)}")
    print(f"High-rated (4+ stars): {len(high_rated)}")
    
    if len(high_rated) > 0:
        print("\nSample high-rated review:")
        sample = high_rated.iloc[0]
        print(f"  Coaster: {sample['coaster_name']}")
        print(f"  Park: {sample['park_name']}")
        print(f"  Rating: {sample['rating']} stars")
        print(f"  Review: {sample['review_text'][:100]}...")


def example_resume_scraping():
    """Example 4: Resume from a specific page"""
    print("\n" + "=" * 60)
    print("Example 4: Resume Scraping")
    print("=" * 60)
    
    crawler = CaptainCoasterCrawler(delay=1.0)
    
    # If you stopped at page 50, resume from page 51
    df = crawler.scrape_all_reviews(start_page=51, end_page=60)
    
    print(f"Resumed scraping: collected {len(df)} reviews from pages 51-60")


def main():
    """Run all examples"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "Captain Coaster Crawler Examples" + " " * 15 + "║")
    print("╚" + "═" * 58 + "╝")
    
    print("\nThis script demonstrates various ways to use the crawler.")
    print("Choose an example to run:\n")
    print("  1. Basic Usage - Scrape and save")
    print("  2. Data Analysis - Analyze existing data")  
    print("  3. Filtered Scraping - Scrape and filter results")
    print("  4. Resume Scraping - Continue from a specific page")
    print("  5. Run all examples (uses test data)")
    
    choice = input("\nEnter choice (1-5, or 'q' to quit): ").strip()
    
    if choice == '1':
        example_basic_usage()
    elif choice == '2':
        example_data_analysis()
    elif choice == '3':
        example_filtered_scraping()
    elif choice == '4':
        example_resume_scraping()
    elif choice == '5':
        example_data_analysis()  # Only run analysis since it uses test data
    else:
        print("No example selected.")


if __name__ == "__main__":
    main()
