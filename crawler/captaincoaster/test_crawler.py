"""
Quick test script for the Captain Coaster crawler
Tests crawling a single page to verify functionality
"""

from crawler import CaptainCoasterCrawler

def test_crawler():
    print("Testing Captain Coaster Crawler")
    print("=" * 50)
    
    # Initialize crawler
    crawler = CaptainCoasterCrawler(delay=1.0)
    
    # Test 1: Get total pages
    print("\nTest 1: Getting total number of pages...")
    try:
        total_pages = crawler.get_total_pages()
        print(f"✓ Success! Total pages: {total_pages}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test 2: Scrape first page
    print("\nTest 2: Scraping first page...")
    try:
        reviews = crawler.scrape_page(1)
        print(f"✓ Success! Found {len(reviews)} reviews")
        
        if reviews:
            print("\nSample review data:")
            sample = reviews[0]
            for key, value in sample.items():
                if value and len(str(value)) > 100:
                    print(f"  {key}: {str(value)[:100]}...")
                else:
                    print(f"  {key}: {value}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Test DataFrame conversion and saving
    print("\nTest 3: Testing data export...")
    try:
        import pandas as pd
        df = pd.DataFrame(reviews)
        print(f"✓ DataFrame created with {len(df)} rows and {len(df.columns)} columns")
        print(f"  Columns: {', '.join(df.columns.tolist())}")
        
        # Save test files
        crawler.save_to_csv(df, "test_reviews.csv")
        crawler.save_to_json(df, "test_reviews.json")
        print("✓ Test files saved successfully!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("\nThe crawler is ready to use.")
    print("Run 'python crawler.py' to start the full crawl.")
    print("=" * 50)
    return True

if __name__ == "__main__":
    test_crawler()
