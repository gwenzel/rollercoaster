"""
Debug script to examine the actual HTML structure of Captain Coaster reviews
"""

import requests
from bs4 import BeautifulSoup

def examine_page_structure():
    url = "https://captaincoaster.com/en/reviews"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Save the HTML for inspection
    with open('debug_page.html', 'w', encoding='utf-8') as f:
        f.write(soup.prettify())
    
    print("HTML saved to debug_page.html")
    print("\nLooking for review containers...")
    
    # Try different selectors
    selectors = [
        ('div.card', soup.find_all('div', class_='card')),
        ('div.review', soup.find_all('div', class_='review')),
        ('article', soup.find_all('article')),
        ('div[class*="review"]', soup.find_all('div', class_=lambda x: x and 'review' in x.lower())),
        ('div.mb-3', soup.find_all('div', class_='mb-3')),
    ]
    
    for selector, elements in selectors:
        print(f"\n{selector}: Found {len(elements)} elements")
        if elements and len(elements) > 0:
            print(f"First element preview:")
            print(str(elements[0])[:500])
            print("...")

if __name__ == "__main__":
    examine_page_structure()
