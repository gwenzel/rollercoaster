"""Quick test script to examine Captain Coaster page structure"""

import requests
from bs4 import BeautifulSoup

# Test with Voltron (ID 5403) - using full URL from scraped data
url = "https://captaincoaster.com/en/coasters/5403/voltron-europa-park"

print(f"Fetching: {url}")
print("=" * 60)

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Save HTML for inspection
with open('test_coaster_page.html', 'w', encoding='utf-8') as f:
    f.write(soup.prettify())

print("✓ Page HTML saved to test_coaster_page.html")

# Try to find coaster name
title = soup.find('h1')
if title:
    print(f"\nCoaster name: {title.text.strip()}")

# Look for rating-related elements
print("\n" + "=" * 60)
print("Looking for rating elements...")
print("=" * 60)

# Search for any element with 'rating' in class
rating_elements = soup.find_all(class_=lambda x: x and 'rating' in x.lower())
print(f"\nFound {len(rating_elements)} elements with 'rating' in class:")
for elem in rating_elements[:10]:  # Show first 10
    print(f"  - {elem.name}.{elem.get('class')}: {elem.text.strip()[:100]}")

# Search for star icons
star_elements = soup.find_all(['i', 'span'], class_=lambda x: x and 'star' in str(x).lower())
print(f"\nFound {len(star_elements)} star icon elements:")
for elem in star_elements[:10]:
    print(f"  - {elem.name}.{elem.get('class')}")

# Search for numbers that could be ratings (0-5 range)
import re
rating_numbers = soup.find_all(string=re.compile(r'\b[0-5]\.[0-9]\b'))
print(f"\nFound {len(rating_numbers)} potential rating numbers:")
for num in rating_numbers[:10]:
    parent = num.parent
    print(f"  - '{num.strip()}' in <{parent.name} class='{parent.get('class')}'>'")

# Search for percentage signs (for distribution)
percentages = soup.find_all(string=re.compile(r'\d+\s*%'))
print(f"\nFound {len(percentages)} percentage values:")
for pct in percentages[:10]:
    parent = pct.parent
    print(f"  - '{pct.strip()}' in <{parent.name} class='{parent.get('class')}'>'")

print("\n" + "=" * 60)
print("Check test_coaster_page.html for full HTML structure")
