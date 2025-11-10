"""
Captain Coaster Review Crawler
Scrapes all review data from https://captaincoaster.com/en/
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from typing import List, Dict
import json
from datetime import datetime
import os


class CaptainCoasterCrawler:
    """Crawler for Captain Coaster reviews"""
    
    def __init__(self, base_url: str = "https://captaincoaster.com/en/reviews", 
                 delay: float = 1.0, max_retries: int = 3):
        """
        Initialize the crawler
        
        Args:
            base_url: Base URL for reviews page
            delay: Delay between requests in seconds (be respectful!)
            max_retries: Maximum number of retries for failed requests
        """
        self.base_url = base_url
        self.delay = delay
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def get_page(self, url: str) -> BeautifulSoup:
        """Fetch and parse a page with retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                return BeautifulSoup(response.content, 'html.parser')
            except requests.RequestException as e:
                print(f"Attempt {attempt + 1}/{self.max_retries} failed for {url}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay * 2)
                else:
                    raise
        return None
    
    def extract_review_data(self, review_element) -> Dict:
        """Extract data from a single review element"""
        try:
            data = {}
            
            # Find the heading section (coaster and park info)
            heading = review_element.find('div', class_='media-heading')
            if not heading:
                return None
                
            # Extract coaster name and link
            h6 = heading.find('h6')
            if h6:
                links = h6.find_all('a')
                if len(links) >= 1:
                    coaster_link = links[0]
                    data['coaster_name'] = coaster_link.text.strip()
                    data['coaster_url'] = 'https://captaincoaster.com' + coaster_link.get('href', '')
                    # Extract coaster ID from URL
                    match = re.search(r'/coasters/(\d+)/', data['coaster_url'])
                    data['coaster_id'] = match.group(1) if match else None
                
                # Extract park name and link
                if len(links) >= 2:
                    park_link = links[1]
                    data['park_name'] = park_link.text.strip()
                    data['park_url'] = 'https://captaincoaster.com' + park_link.get('href', '')
                    # Extract park ID from URL
                    match = re.search(r'/parks/(\d+)/', data['park_url'])
                    data['park_id'] = match.group(1) if match else None
            
            # Find the review content div
            review_div = review_element.find('div', class_='media', attrs={'data-controller': 'review-actions'})
            if not review_div:
                return None
            
            # Extract review ID from data attribute
            review_id = review_div.get('data-review-actions-id-value')
            if review_id:
                data['review_id'] = review_id
            
            # Extract reviewer information from media-body
            media_body = review_div.find('div', class_='media-body')
            if media_body:
                # Find reviewer link
                reviewer_heading = media_body.find('div', class_='media-heading')
                if reviewer_heading:
                    reviewer_link = reviewer_heading.find('a', class_='text-semibold')
                    if reviewer_link:
                        data['reviewer_name'] = reviewer_link.text.strip()
                        data['reviewer_url'] = 'https://captaincoaster.com' + reviewer_link.get('href', '')
                        # Extract user ID from URL
                        match = re.search(r'/users/([^/]+)', data['reviewer_url'])
                        data['reviewer_id'] = match.group(1) if match else None
                    
                    # Extract rating (count stars)
                    rating_span = reviewer_heading.find('span', class_='media-annotation')
                    if rating_span:
                        full_stars = rating_span.find_all('i', class_='icon-star-full2')
                        half_stars = rating_span.find_all('i', class_='icon-star-half')
                        data['rating'] = len(full_stars) + (0.5 * len(half_stars))
                    
                    # Extract timestamp
                    time_spans = reviewer_heading.find_all('span', class_='media-annotation')
                    if len(time_spans) >= 2:
                        data['time_posted'] = time_spans[-1].text.strip()
                
                # Extract tags/features (positive and negative)
                tags_positive = []
                tags_negative = []
                tag_elements = media_body.find_all('span', class_='label')
                for tag in tag_elements:
                    tag_text = tag.text.strip()
                    if 'text-success' in tag.get('class', []):
                        tags_positive.append(tag_text)
                    elif 'text-danger' in tag.get('class', []):
                        tags_negative.append(tag_text)
                
                data['tags_positive'] = ', '.join(tags_positive) if tags_positive else None
                data['tags_negative'] = ', '.join(tags_negative) if tags_negative else None
                
                # Extract review text (look for <p> tags that are not tags container)
                review_paragraphs = media_body.find_all('p', recursive=False)
                review_texts = []
                for p in review_paragraphs:
                    # Skip the paragraph that contains labels
                    if not p.find('span', class_='label'):
                        text = p.text.strip()
                        if text:
                            review_texts.append(text)
                
                data['review_text'] = ' '.join(review_texts) if review_texts else None
                
                # Extract upvote count
                upvote_count_elem = media_body.find('span', attrs={'data-review-actions-target': 'upvoteCount'})
                if upvote_count_elem:
                    try:
                        data['upvotes'] = int(upvote_count_elem.text.strip())
                    except:
                        data['upvotes'] = 0
            
            data['scraped_at'] = datetime.now().isoformat()
            
            return data
            
        except Exception as e:
            print(f"Error extracting review data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_total_pages(self) -> int:
        """Get the total number of review pages"""
        try:
            soup = self.get_page(self.base_url)
            # Look for pagination
            pagination = soup.find('nav', class_='pagination') or soup.find('ul', class_='pagination')
            if pagination:
                page_links = pagination.find_all('a', href=re.compile(r'/reviews/\d+'))
                if page_links:
                    # Get the highest page number
                    page_numbers = []
                    for link in page_links:
                        match = re.search(r'/reviews/(\d+)', link.get('href', ''))
                        if match:
                            page_numbers.append(int(match.group(1)))
                    return max(page_numbers) if page_numbers else 1
            return 1
        except Exception as e:
            print(f"Error getting total pages: {e}")
            return 1
    
    def scrape_page(self, page_num: int) -> List[Dict]:
        """Scrape all reviews from a single page"""
        if page_num == 1:
            url = self.base_url
        else:
            url = f"{self.base_url}/{page_num}"
        
        print(f"Scraping page {page_num}: {url}")
        
        try:
            soup = self.get_page(url)
            reviews = []
            
            # Find all review items - they are in <li class="media"> elements
            review_items = soup.find_all('li', class_='media')
            
            for item in review_items:
                review_data = self.extract_review_data(item)
                if review_data and review_data.get('coaster_name'):
                    reviews.append(review_data)
            
            print(f"  Found {len(reviews)} reviews on page {page_num}")
            return reviews
            
        except Exception as e:
            print(f"Error scraping page {page_num}: {e}")
            return []
    
    def scrape_all_reviews(self, start_page: int = 1, end_page: int = None, 
                          save_interval: int = 10) -> pd.DataFrame:
        """
        Scrape all reviews from the website
        
        Args:
            start_page: Starting page number
            end_page: Ending page number (None = scrape all)
            save_interval: Save progress every N pages
        
        Returns:
            DataFrame with all review data
        """
        if end_page is None:
            end_page = self.get_total_pages()
        
        print(f"Starting scrape from page {start_page} to {end_page}")
        print(f"Estimated total reviews: {(end_page - start_page + 1) * 10}")
        
        all_reviews = []
        
        for page_num in range(start_page, end_page + 1):
            try:
                reviews = self.scrape_page(page_num)
                all_reviews.extend(reviews)
                
                # Save progress periodically
                if page_num % save_interval == 0:
                    self._save_progress(all_reviews, page_num)
                    print(f"Progress saved at page {page_num}. Total reviews: {len(all_reviews)}")
                
                # Be respectful - add delay between requests
                time.sleep(self.delay)
                
            except KeyboardInterrupt:
                print("\nScraping interrupted by user. Saving progress...")
                self._save_progress(all_reviews, page_num)
                break
            except Exception as e:
                print(f"Error on page {page_num}: {e}")
                continue
        
        df = pd.DataFrame(all_reviews)
        print(f"\nScraping complete! Total reviews collected: {len(df)}")
        return df
    
    def _save_progress(self, reviews: List[Dict], page_num: int):
        """Save progress to a temporary file"""
        if reviews:
            df = pd.DataFrame(reviews)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"captaincoaster_reviews_progress_page{page_num}_{timestamp}.csv"
            df.to_csv(filename, index=False, encoding='utf-8')
            print(f"Progress saved to {filename}")
    
    def save_to_csv(self, df: pd.DataFrame, filename: str = None):
        """Save reviews to CSV file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"captaincoaster_reviews_{timestamp}.csv"
        
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Data saved to {filename}")
        return filename
    
    def save_to_json(self, df: pd.DataFrame, filename: str = None):
        """Save reviews to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"captaincoaster_reviews_{timestamp}.json"
        
        df.to_json(filename, orient='records', indent=2)
        print(f"Data saved to {filename}")
        return filename


def main():
    """Main execution function"""
    print("Captain Coaster Review Crawler")
    print("=" * 50)
    
    # Initialize crawler
    crawler = CaptainCoasterCrawler(delay=1.0)  # 1 second delay between requests
    
    # Get total pages
    print("Checking total number of pages...")
    total_pages = crawler.get_total_pages()
    print(f"Total pages found: {total_pages}")
    
    # Ask user for scraping range
    print("\nOptions:")
    print(f"1. Scrape all pages (1-{total_pages}) - Warning: This will take a long time!")
    print("2. Scrape a specific range")
    print("3. Scrape first 10 pages (test mode)")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        start_page = 1
        end_page = total_pages
    elif choice == "2":
        start_page = int(input(f"Enter start page (1-{total_pages}): "))
        end_page = int(input(f"Enter end page ({start_page}-{total_pages}): "))
    else:
        start_page = 1
        end_page = 10
    
    print(f"\nStarting scrape: pages {start_page} to {end_page}")
    print("Press Ctrl+C to stop and save progress at any time")
    print("-" * 50)
    
    # Scrape reviews
    df = crawler.scrape_all_reviews(start_page=start_page, end_page=end_page, save_interval=10)
    
    # Save results
    if not df.empty:
        csv_file = crawler.save_to_csv(df)
        json_file = crawler.save_to_json(df)
        
        print("\n" + "=" * 50)
        print("Scraping Summary:")
        print(f"Total reviews collected: {len(df)}")
        print(f"Unique coasters: {df['coaster_name'].nunique() if 'coaster_name' in df else 'N/A'}")
        print(f"Unique reviewers: {df['reviewer_name'].nunique() if 'reviewer_name' in df else 'N/A'}")
        print(f"\nFiles saved:")
        print(f"  - CSV: {csv_file}")
        print(f"  - JSON: {json_file}")
        print("=" * 50)
        
        # Display sample
        print("\nSample of collected data:")
        print(df.head())
    else:
        print("No data collected.")


if __name__ == "__main__":
    main()
