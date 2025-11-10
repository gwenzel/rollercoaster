"""
Ride Forces Database Crawler
Downloads all ride recordings and metadata from https://rideforcesdb.com/

Data includes:
- Ride information (name, park, manufacturer)
- Force recordings (vertical, lateral, longitudinal G-forces)
- Recording metadata (device, date, uploader)
- Peak force statistics
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
from datetime import datetime
import os
from typing import List, Dict
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import re


class RideForcesDBCrawler:
    """Crawler for Ride Forces Database"""
    
    def __init__(self, base_url="https://rideforcesdb.com", delay=2.0, use_selenium=True):
        """
        Initialize crawler
        
        Args:
            base_url: Base URL for RideForcesDB
            delay: Delay between requests in seconds
            use_selenium: Use Selenium for JavaScript-rendered pages
        """
        self.base_url = base_url
        self.delay = delay
        self.use_selenium = use_selenium
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.driver = None
        
        if use_selenium:
            self._init_selenium()
    
    def _init_selenium(self):
        """Initialize Selenium WebDriver"""
        print("Initializing Selenium WebDriver...")
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # Run in background
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--window-size=1920,1080')
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            print("✓ Selenium initialized")
        except Exception as e:
            print(f"✗ Error initializing Selenium: {e}")
            print("  Install ChromeDriver: https://chromedriver.chromium.org/")
            self.use_selenium = False
    
    def get_page_selenium(self, url: str, wait_time=10):
        """Load page with Selenium and wait for content"""
        if not self.driver:
            return None
        
        try:
            self.driver.get(url)
            # Wait for main content to load
            WebDriverWait(self.driver, wait_time).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(2)  # Additional wait for dynamic content
            return self.driver.page_source
        except Exception as e:
            print(f"Error loading {url}: {e}")
            return None
    
    def get_all_rides(self):
        """
        Get list of all rides from browse page
        Note: This is a placeholder - need to inspect actual API calls
        """
        print("\nFetching ride list from browse page...")
        
        if self.use_selenium:
            html = self.get_page_selenium(f"{self.base_url}/browse")
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                # TODO: Parse ride list from HTML
                # This will depend on the actual HTML structure
                print("Page loaded, analyzing structure...")
                
                # Save HTML for inspection
                with open('rfdb_browse.html', 'w', encoding='utf-8') as f:
                    f.write(soup.prettify())
                print("Saved page HTML to rfdb_browse.html for inspection")
        
        # Alternative: Try to find API endpoints
        print("\nLooking for API endpoints...")
        return self._discover_api_endpoints()
    
    def _discover_api_endpoints(self):
        """
        Attempt to discover API endpoints
        Many modern sites use REST APIs that can be accessed directly
        """
        print("Checking for API endpoints...")
        
        # Common API patterns to try
        api_patterns = [
            '/api/rides',
            '/api/recordings',
            '/api/browse',
            '/api/v1/rides',
            '/data/rides.json'
        ]
        
        for pattern in api_patterns:
            url = f"{self.base_url}{pattern}"
            try:
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    print(f"✓ Found API endpoint: {pattern}")
                    try:
                        data = response.json()
                        print(f"  Data type: {type(data)}")
                        if isinstance(data, list):
                            print(f"  Items: {len(data)}")
                        elif isinstance(data, dict):
                            print(f"  Keys: {list(data.keys())[:5]}")
                        return url, data
                    except:
                        print(f"  Response not JSON")
            except Exception as e:
                pass
        
        print("No direct API endpoints found")
        return None, None
    
    def extract_ride_data_from_html(self, html):
        """Extract ride data from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        rides = []
        
        # This needs to be customized based on actual HTML structure
        # Placeholder implementation
        ride_elements = soup.find_all('div', class_='ride-card')  # Example class
        
        for elem in ride_elements:
            ride_data = {
                'name': elem.get_text().strip() if elem else None,
                'url': elem.find('a')['href'] if elem.find('a') else None
            }
            rides.append(ride_data)
        
        return rides
    
    def download_recording(self, recording_id: str, output_dir='data/rfdb_recordings'):
        """Download a specific recording"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Recordings can be downloaded in .zforces format
        url = f"{self.base_url}/recording/{recording_id}/download"
        
        try:
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                filepath = os.path.join(output_dir, f"recording_{recording_id}.zforces")
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"✓ Downloaded recording {recording_id}")
                return filepath
        except Exception as e:
            print(f"✗ Error downloading recording {recording_id}: {e}")
        
        return None
    
    def scrape_all_data(self):
        """Main scraping function"""
        print("="*60)
        print("RIDE FORCES DATABASE CRAWLER")
        print("="*60)
        print(f"\nTarget: {self.base_url}")
        print(f"Expected data: ~4,623 recordings, 794 rides, 154 parks")
        
        # Get rides
        api_url, api_data = self.get_all_rides()
        
        if api_data:
            print(f"\nSuccessfully retrieved data from API")
            # Process API data
            df = self._process_api_data(api_data)
            return df
        else:
            print("\nNote: This site uses heavy JavaScript rendering.")
            print("To fully scrape this site, you'll need to:")
            print("  1. Inspect network calls in browser DevTools")
            print("  2. Find API endpoints used by the site")
            print("  3. Update crawler with correct API calls")
            print("\nAlternatively, use browser automation to interact with the site.")
            
            return None
    
    def _process_api_data(self, data):
        """Process API data into DataFrame"""
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame()
        
        return df
    
    def close(self):
        """Clean up resources"""
        if self.driver:
            self.driver.quit()
            print("Selenium driver closed")


def inspect_site_structure():
    """
    Helper function to inspect the site structure
    Use browser DevTools to find actual API calls
    """
    print("\n" + "="*60)
    print("SITE INSPECTION GUIDE")
    print("="*60)
    print("\nTo find the API endpoints:")
    print("\n1. Open https://rideforcesdb.com/browse in your browser")
    print("2. Press F12 to open Developer Tools")
    print("3. Go to the 'Network' tab")
    print("4. Refresh the page")
    print("5. Look for XHR/Fetch requests (filter by 'XHR')")
    print("6. Find requests to endpoints like:")
    print("   - /api/rides")
    print("   - /api/recordings")
    print("   - /graphql")
    print("   - Any JSON responses")
    print("\n7. Click on these requests to see:")
    print("   - Request URL (full endpoint)")
    print("   - Request Method (GET/POST)")
    print("   - Request Headers")
    print("   - Response (JSON data)")
    print("\n8. Update the crawler with the correct endpoints")
    print("="*60)


def main():
    """Main execution"""
    print("\n" + "="*60)
    print("RIDE FORCES DATABASE CRAWLER - SETUP")
    print("="*60)
    
    print("\nThis site uses JavaScript rendering.")
    print("Options:")
    print("  1. Try automatic crawling (may be incomplete)")
    print("  2. Show site inspection guide")
    print("  3. Manual API endpoint entry")
    
    choice = input("\nChoice (1-3): ").strip()
    
    if choice == "2":
        inspect_site_structure()
        return
    elif choice == "3":
        api_endpoint = input("Enter API endpoint URL: ").strip()
        if api_endpoint:
            try:
                response = requests.get(api_endpoint)
                data = response.json()
                print(f"\n✓ Retrieved data from {api_endpoint}")
                print(f"Data preview: {json.dumps(data, indent=2)[:500]}...")
                
                # Save to file
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f"rfdb_data_{timestamp}.json"
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"✓ Saved to {output_file}")
            except Exception as e:
                print(f"✗ Error: {e}")
        return
    
    # Try automatic crawling
    crawler = RideForcesDBCrawler(use_selenium=True)
    
    try:
        df = crawler.scrape_all_data()
        
        if df is not None and not df.empty:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"rfdb_rides_{timestamp}.csv"
            df.to_csv(output_file, index=False)
            print(f"\n✓ Saved {len(df)} rides to {output_file}")
        else:
            print("\nPlease use option 2 to see how to find the API endpoints.")
            inspect_site_structure()
    finally:
        crawler.close()


if __name__ == "__main__":
    main()
