"""
Captain Coaster Rating Distribution Crawler
Crawls individual coaster pages to collect detailed rating distributions.

For each rollercoaster, collects:
- Average rating (e.g., 3.8 stars)
- Rating distribution percentages (% of users giving 0.5★, 1★, 1.5★, ..., 5★)
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import os
from typing import Dict, Optional, List
from datetime import datetime
import json


class RatingDistributionCrawler:
    """
    Crawls Captain Coaster coaster pages to extract rating distributions.
    """
    
    def __init__(self, delay: float = 1.0):
        """
        Initialize the crawler
        
        Args:
            delay: Delay between requests in seconds (default: 1.0)
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def get_page(self, url: str) -> Optional[BeautifulSoup]:
        """
        Get and parse a web page
        
        Args:
            url: URL to fetch
            
        Returns:
            BeautifulSoup object or None if failed
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def extract_rating_distribution(self, coaster_id: int, coaster_name: str = None, coaster_slug: str = None) -> Optional[Dict]:
        """
        Extract rating distribution for a specific coaster
        
        Args:
            coaster_id: Captain Coaster coaster ID
            coaster_name: Optional coaster name (for display)
            coaster_slug: Optional URL slug (if not provided, will try to get from coaster name)
            
        Returns:
            Dictionary with rating data or None if failed
        """
        # Construct coaster page URL - need the slug for proper access
        # If slug not provided, create a generic one from the name
        if not coaster_slug and coaster_name:
            # Simple slugification: lowercase, replace spaces with hyphens
            coaster_slug = coaster_name.lower().replace(' ', '-').replace('(', '').replace(')', '')
            coaster_slug = re.sub(r'[^a-z0-9-]', '', coaster_slug)
        
        if coaster_slug:
            url = f"https://captaincoaster.com/en/coasters/{coaster_id}/{coaster_slug}"
        else:
            url = f"https://captaincoaster.com/en/coasters/{coaster_id}"
        
        try:
            soup = self.get_page(url)
            if not soup:
                return None
            
            data = {
                'coaster_id': coaster_id,
                'coaster_name': coaster_name,
                'url': url,
                'avg_rating': None,
                'total_ratings': None,
            }
            
            # Initialize rating percentage columns
            rating_levels = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
            for rating in rating_levels:
                data[f'pct_{rating}_stars'] = None
            
            # Extract coaster name if not provided
            if not coaster_name:
                title_elem = soup.find('h1')
                if title_elem:
                    # Title format is often "Coaster Name • Park Name"
                    title_text = title_elem.text.strip()
                    if '•' in title_text:
                        data['coaster_name'] = title_text.split('•')[0].strip()
                    else:
                        data['coaster_name'] = title_text
            
            # Extract rating data from JavaScript - Captain Coaster embeds it in ApexCharts config
            scripts = soup.find_all('script')
            for script in scripts:
                script_text = script.string
                if not script_text:
                    continue
                
                # Look for the ApexCharts series data
                if 'series:' in script_text and 'name:' in script_text and 'data:' in script_text:
                    # Extract rating counts from series
                    # Pattern: { name: 0.5, data: [7] }, { name: 1, data: [9] }, etc.
                    rating_matches = re.findall(r'name:\s*(\d+(?:\.\d+)?),\s*data:\s*\[(\d+)\]', script_text)
                    
                    if rating_matches:
                        # Calculate total ratings
                        total = sum(int(count) for _, count in rating_matches)
                        data['total_ratings'] = total
                        
                        # Calculate average rating
                        weighted_sum = sum(float(rating) * int(count) for rating, count in rating_matches)
                        if total > 0:
                            data['avg_rating'] = round(weighted_sum / total, 2)
                        
                        # Calculate percentages for each rating level
                        for rating_str, count_str in rating_matches:
                            rating = float(rating_str)
                            count = int(count_str)
                            percentage = round((count / total * 100), 2) if total > 0 else 0
                            data[f'pct_{rating}_stars'] = percentage
                            data[f'count_{rating}_stars'] = count
                        
                        break

            # Fallback: parse JSON-LD AggregateRating if present
            if data['avg_rating'] is None or data['total_ratings'] is None:
                try:
                    json_ld_scripts = soup.find_all('script', type='application/ld+json')
                    for s in json_ld_scripts:
                        try:
                            obj = json.loads(s.get_text(strip=True))
                        except Exception:
                            continue
                        # Some pages have a list, others a single object
                        def _iter_objs(o):
                            if isinstance(o, list):
                                for it in o:
                                    yield it
                            else:
                                yield o
                        for o in _iter_objs(obj):
                            ar = o.get('aggregateRating') if isinstance(o, dict) else None
                            if isinstance(ar, dict):
                                rating_val = ar.get('ratingValue')
                                rating_cnt = ar.get('ratingCount') or ar.get('reviewCount')
                                try:
                                    if rating_val is not None:
                                        data['avg_rating'] = round(float(str(rating_val).replace(',', '.')), 2)
                                    if rating_cnt is not None:
                                        data['total_ratings'] = int(str(rating_cnt).replace(',', ''))
                                    break
                                except Exception:
                                    continue
                        if data['avg_rating'] is not None and data['total_ratings'] is not None:
                            break
                except Exception:
                    pass
            
            # Alternative: Look for percentage display (like "Score of 98.7%")
            if data['avg_rating'] is None:
                score_text = soup.find(string=re.compile(r'Score of \d+[.,]\d+%'))
                if score_text:
                    match = re.search(r'Score of ([\d.,]+)%', score_text)
                    if match:
                        # Convert percentage to 5-star rating
                        percentage = float(match.group(1).replace(',', '.'))
                        data['avg_rating'] = round(percentage / 20, 2)  # 100% = 5 stars
            
            # Add scrape timestamp
            data['scraped_at'] = datetime.now().isoformat()

            # Parse physical specs (height, speed, length) from page text
            try:
                specs = self._extract_specs(soup)
                if specs:
                    data.update(specs)
            except Exception as _spec_e:
                # Keep going even if spec parsing fails
                pass
            
            return data
            
        except Exception as e:
            print(f"Error extracting rating distribution for coaster {coaster_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def scrape_coasters_from_csv(self, 
                                  ratings_csv: str,
                                  url_mapping_csv: str = None,
                                  output_csv: str = None,
                                  start_index: int = 0,
                                  end_index: int = None,
                                  save_interval: int = 50) -> pd.DataFrame:
        """
        Scrape rating distributions for coasters from a ratings CSV file
        
        Args:
            ratings_csv: Path to ratings data CSV (with coaster_id column)
            url_mapping_csv: Path to coaster ID to URL mapping CSV (optional)
            output_csv: Output CSV path (default: auto-generated)
            start_index: Starting index in the coaster list
            end_index: Ending index (None = all coasters)
            save_interval: Save progress every N coasters
            
        Returns:
            DataFrame with rating distribution data
        """
        # Load coaster IDs from ratings data
        print(f"Loading coaster data from {ratings_csv}...")
        ratings_df = pd.read_csv(ratings_csv)
        
        # Get unique coasters
        coaster_info = ratings_df[['coaster_id', 'coaster_name']].drop_duplicates('coaster_id')
        coaster_info = coaster_info.sort_values('coaster_id')
        
        # Load URL mapping if provided
        url_map = {}
        if url_mapping_csv and os.path.exists(url_mapping_csv):
            print(f"Loading URL mapping from {url_mapping_csv}...")
            url_df = pd.read_csv(url_mapping_csv)
            url_map = dict(zip(url_df['coaster_id'], url_df['coaster_url']))
            print(f"Loaded {len(url_map)} coaster URLs")
        
        total_coasters = len(coaster_info)
        print(f"Found {total_coasters} unique coasters in ratings data")
        
        # Apply start/end indices
        if end_index is None:
            end_index = total_coasters
        
        coaster_info = coaster_info.iloc[start_index:end_index]
        print(f"Will scrape coasters {start_index} to {end_index-1} ({len(coaster_info)} coasters)")
        
        # Scrape each coaster
        all_data = []
        
        for idx, (_, row) in enumerate(coaster_info.iterrows(), start=start_index):
            coaster_id = row['coaster_id']
            coaster_name = row['coaster_name']
            
            # Get URL slug if available
            coaster_slug = None
            if coaster_id in url_map:
                url = url_map[coaster_id]
                # Extract slug from URL: https://captaincoaster.com/en/coasters/5403/voltron-europa-park
                match = re.search(r'/coasters/\d+/([^/]+)', url)
                if match:
                    coaster_slug = match.group(1)
            
            print(f"[{idx+1}/{end_index}] Scraping: {coaster_name} (ID: {coaster_id})")
            
            try:
                rating_data = self.extract_rating_distribution(coaster_id, coaster_name, coaster_slug)
                if rating_data:
                    all_data.append(rating_data)
                    print(f"  ✓ Success - Avg rating: {rating_data.get('avg_rating', 'N/A')}")
                else:
                    print(f"  ✗ Failed to extract data")
                
                # Save progress periodically
                if (idx + 1) % save_interval == 0 and all_data:
                    self._save_progress(all_data, idx + 1, output_csv)
                
                # Delay between requests
                time.sleep(self.delay)
                
            except KeyboardInterrupt:
                print("\n\nScraping interrupted by user. Saving progress...")
                if all_data:
                    self._save_progress(all_data, idx + 1, output_csv)
                break
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
        
        # Create final DataFrame
        df = pd.DataFrame(all_data)
        
        # Save final results
        if output_csv is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_csv = f"rating_distributions_{timestamp}.csv"
        
        if not df.empty:
            df.to_csv(output_csv, index=False, encoding='utf-8')
            print(f"\n✓ Final data saved to {output_csv}")
            print(f"✓ Total coasters scraped: {len(df)}")
            print(f"✓ Coasters with ratings: {df['avg_rating'].notna().sum()}")
        
        return df

    def scrape_from_url_mapping(self,
                                url_mapping_csv: str,
                                output_csv: str = None,
                                start_index: int = 0,
                                end_index: int = None,
                                save_interval: int = 50) -> pd.DataFrame:
        """Scrape coasters using a CSV that maps `coaster_id` to full Captain Coaster URLs.
        This mode does not require per-user reviews CSV.
        """
        if not url_mapping_csv or not os.path.exists(url_mapping_csv):
            raise FileNotFoundError(f"URL mapping CSV not found: {url_mapping_csv}")

        url_df = pd.read_csv(url_mapping_csv)
        if not {'coaster_id', 'coaster_url'}.issubset(url_df.columns):
            raise ValueError("url_mapping_csv must contain columns: 'coaster_id', 'coaster_url'")

        # Derive coaster_name from URL slug if not provided
        url_df = url_df.copy()
        if 'coaster_name' not in url_df.columns:
            url_df['coaster_name'] = url_df['coaster_url'].apply(lambda u: re.sub(r'-', ' ', re.search(r'/coasters/\d+/([^/]+)', u).group(1)) if isinstance(u, str) and re.search(r'/coasters/\d+/([^/]+)', u) else None)

        total = len(url_df)
        if end_index is None:
            end_index = total
        url_df = url_df.iloc[start_index:end_index]
        print(f"Will scrape {len(url_df)} coasters from URL mapping [{start_index}:{end_index}]...")

        all_data: List[Dict] = []
        for idx, row in enumerate(url_df.itertuples(index=False), start=start_index):
            cid = getattr(row, 'coaster_id')
            url = getattr(row, 'coaster_url')
            cname = getattr(row, 'coaster_name') if hasattr(row, 'coaster_name') else None
            slug_match = re.search(r'/coasters/\d+/([^/]+)', url) if isinstance(url, str) else None
            slug = slug_match.group(1) if slug_match else None
            print(f"[{idx+1}/{end_index}] Scraping: {cname or slug} (ID: {cid})")
            try:
                data = self.extract_rating_distribution(cid, coaster_name=cname, coaster_slug=slug)
                if data:
                    all_data.append(data)
                    print(f"  ✓ Success - Avg rating: {data.get('avg_rating', 'N/A')}\n")
                else:
                    print("  ✗ Failed to extract data\n")
                if (idx + 1) % save_interval == 0 and all_data:
                    self._save_progress(all_data, idx + 1, output_csv)
                time.sleep(self.delay)
            except Exception as e:
                print(f"  ✗ Error: {e}\n")
                continue

        df = pd.DataFrame(all_data)
        if output_csv is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_csv = f"rating_distributions_{timestamp}.csv"
        if not df.empty:
            df.to_csv(output_csv, index=False, encoding='utf-8')
            print(f"\n✓ Final data saved to {output_csv}")
            print(f"✓ Total coasters scraped: {len(df)}")
            print(f"✓ Coasters with ratings: {df['avg_rating'].notna().sum()}")
        return df
    
    def _save_progress(self, data: List[Dict], current_index: int, output_csv: str = None):
        """Save progress to a checkpoint file"""
        if not data:
            return
        
        df = pd.DataFrame(data)
        
        if output_csv:
            checkpoint_file = output_csv.replace('.csv', f'_checkpoint_{current_index}.csv')
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint_file = f"rating_distributions_checkpoint_{current_index}_{timestamp}.csv"
        
        df.to_csv(checkpoint_file, index=False, encoding='utf-8')
        print(f"  💾 Progress saved to {checkpoint_file}")

    def _extract_specs(self, soup: BeautifulSoup) -> Dict[str, float]:
        """Extract physical specifications from the coaster page.
        Returns a dict with keys: height_m, speed_kmh, track_length_m, inversions_count (floats/ints when available).
        Tries DOM-targeted extraction first, then falls back to full-text regex.
        """
        specs: Dict[str, float] = {}

        # DOM-targeted: find the four feature buttons under 'content-group'
        try:
            group = soup.find('div', class_='content-group')
            if group:
                buttons = group.select('div.row.row-seamless.btn-block-group button div.text-bold.mt-5')
                # Expect up to four entries: height (ft), speed (mph), length (ft), laps (int)
                values = [b.get_text(strip=True) for b in buttons]
                def _to_float(num_str: str) -> Optional[float]:
                    try:
                        return float(num_str.replace(',', '').replace(' ', '').replace('ft','').replace('mph','').replace('kph','').replace('km/h',''))
                    except Exception:
                        try:
                            return float(num_str.replace(',', '.'))
                        except Exception:
                            return None
                for val in values:
                    lower = val.lower()
                    if 'ft' in lower and any(ch.isdigit() for ch in lower):
                        f = _to_float(lower)
                        if f is not None:
                            # If length looks very large (> 500), assume track length; else could be height
                            if f >= 500:
                                specs['track_length_m'] = round(f * 0.3048, 2)
                            else:
                                specs['height_m'] = round(f * 0.3048, 2)
                    elif 'mph' in lower:
                        f = _to_float(lower)
                        if f is not None:
                            specs['speed_kmh'] = round(f * 1.60934, 2)
                    elif lower.isdigit():
                        # Laps/inversions button shows an integer
                        try:
                            specs['inversions_count'] = int(lower)
                        except Exception:
                            pass
        except Exception:
            pass

        # If DOM-targeted failed to populate, fallback to text regex
        text = soup.get_text(separator='\n')
        # Normalize text (replace commas in numbers, unify units)
        def _to_float(num_str: str) -> Optional[float]:
            try:
                return float(num_str.replace(',', '.'))
            except Exception:
                return None

        # Only fill missing values via regex fallback

        # Height in meters
        m = re.search(r'(?:Height|Total\s*height|Max\s*height)[^\d]*(\d+(?:[\.,]\d+)?)\s*m', text, flags=re.IGNORECASE)
        if m and 'height_m' not in specs:
            val = _to_float(m.group(1))
            if val is not None:
                specs['height_m'] = val

        # Speed with unit conversion support
        m_kmh = re.search(r'(?:Speed|Top\s*speed)[^\d]*(\d+(?:[\.,]\d+)?)\s*(?:km/?h|kmh|kph)', text, flags=re.IGNORECASE)
        m_mph = re.search(r'(?:Speed|Top\s*speed)[^\d]*(\d+(?:[\.,]\d+)?)\s*mph', text, flags=re.IGNORECASE)
        if m_kmh and 'speed_kmh' not in specs:
            val = _to_float(m_kmh.group(1))
            if val is not None:
                specs['speed_kmh'] = val
        elif m_mph and 'speed_kmh' not in specs:
            val = _to_float(m_mph.group(1))
            if val is not None:
                specs['speed_kmh'] = round(val * 1.60934, 2)

        # Track length in meters
        m_len = re.search(r'(?:Length|Track\s*length|Total\s*length)[^\d]*(\d+(?:[\.,]\d+)?)\s*m', text, flags=re.IGNORECASE)
        if m_len and 'track_length_m' not in specs:
            val = _to_float(m_len.group(1))
            if val is not None:
                specs['track_length_m'] = val

        # Inversions count (loops and similar)
        m_inv = re.search(r'(?:Inversions|Number\s*of\s*inversions)[^\d]*(\d+)', text, flags=re.IGNORECASE)
        if m_inv and 'inversions_count' not in specs:
            try:
                specs['inversions_count'] = int(m_inv.group(1))
            except Exception:
                pass

        return specs
    
    def scrape_single_coaster(self, coaster_id: int) -> Optional[Dict]:
        """
        Scrape rating distribution for a single coaster (convenience method)
        
        Args:
            coaster_id: Captain Coaster coaster ID
            
        Returns:
            Dictionary with rating data
        """
        return self.extract_rating_distribution(coaster_id)

    def scrape_single_url(self, url: str, output_csv: Optional[str] = None) -> pd.DataFrame:
        """Scrape a single Captain Coaster page by full URL without needing IDs/CSV.
        Extracts `coaster_id` and slug from the URL and delegates to `extract_rating_distribution`.
        """
        m = re.search(r'/coasters/(\d+)(?:/([^/]+))?', url)
        if not m:
            raise ValueError(f"Unrecognized Captain Coaster URL format: {url}")
        coaster_id = int(m.group(1))
        slug = m.group(2) if len(m.groups()) >= 2 else None
        data = self.extract_rating_distribution(coaster_id, coaster_slug=slug)
        df = pd.DataFrame([data]) if data else pd.DataFrame()
        if output_csv and not df.empty:
            df.to_csv(output_csv, index=False, encoding='utf-8')
            print(f"\n✓ Saved single URL data to {output_csv}")
        return df

    def scrape_urls_file(self,
                         urls_file: str,
                         output_csv: Optional[str] = None,
                         start_index: int = 0,
                         end_index: Optional[int] = None,
                         save_interval: int = 50,
                         parallel_workers: int = 1) -> pd.DataFrame:
        """Scrape a newline-separated list of Captain Coaster URLs from a text file.
        No CSV dependencies required.
        """
        if not os.path.exists(urls_file):
            raise FileNotFoundError(f"URLs file not found: {urls_file}")
        with open(urls_file, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
        total = len(urls)
        if end_index is None:
            end_index = total
        urls = urls[start_index:end_index]
        total_count = len(urls)
        print(f"Will scrape {total_count} coasters from URLs file [{start_index}:{end_index}]...")

        all_data: List[Dict] = []

        def _scrape_one(url: str) -> Optional[Dict]:
            try:
                m = re.search(r'/coasters/(\d+)(?:/([^/]+))?', url)
                coaster_id = int(m.group(1)) if m else None
                slug = m.group(2) if (m and len(m.groups()) >= 2) else None
                if coaster_id is not None:
                    return self.extract_rating_distribution(coaster_id, coaster_slug=slug)
                soup = self.get_page(url)
                if soup:
                    canon = soup.find('link', rel='canonical')
                    href = canon.get('href') if canon else url
                    m2 = re.search(r'/coasters/(\d+)', href or '')
                    if m2:
                        cid = int(m2.group(1))
                        return self.extract_rating_distribution(cid, coaster_slug=slug)
                return None
            finally:
                if self.delay and parallel_workers == 1:
                    time.sleep(self.delay)

        if parallel_workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=parallel_workers) as ex:
                futures = {ex.submit(_scrape_one, u): i for i, u in enumerate(urls, start=1)}
                processed = 0
                for fut in as_completed(futures):
                    processed += 1
                    data = None
                    try:
                        data = fut.result()
                    except Exception as e:
                        print(f"  ✗ Error (parallel): {e}")
                    if data:
                        all_data.append(data)
                    if processed % save_interval == 0 and all_data:
                        self._save_progress(all_data, processed, output_csv)
                    print(f"[{processed}/{total_count}] processed")
        else:
            for idx, url in enumerate(urls, start=1):
                label = url
                print(f"[{idx}/{total_count}] Scraping: {label}")
                try:
                    data = _scrape_one(url)
                    if data:
                        all_data.append(data)
                        print(f"  ✓ Success - Avg rating: {data.get('avg_rating', 'N/A')}\n")
                    else:
                        print("  ✗ Failed to extract data\n")
                    if idx % save_interval == 0 and all_data:
                        self._save_progress(all_data, idx, output_csv)
                except Exception as e:
                    print(f"  ✗ Error: {e}\n")
                    continue

        df = pd.DataFrame(all_data)
        if output_csv is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_csv = f"rating_distributions_{timestamp}.csv"
        if not df.empty:
            df.to_csv(output_csv, index=False, encoding='utf-8')
            print(f"\n✓ Final data saved to {output_csv}")
            print(f"✓ Total coasters scraped: {len(df)}")
            print(f"✓ Coasters with ratings: {df['avg_rating'].notna().sum()}")
        return df

    def scrape_urls_csv(self,
                        urls_csv: str,
                        url_column: str = 'coaster_url',
                        output_csv: Optional[str] = None,
                        start_index: int = 0,
                        end_index: Optional[int] = None,
                        save_interval: int = 50,
                        parallel_workers: int = 1) -> pd.DataFrame:
        """Scrape URLs listed in a CSV file. You can specify which column contains URLs.
        Useful for files like `ratings_data/complete_coaster_mapping.csv`.
        """
        if not os.path.exists(urls_csv):
            raise FileNotFoundError(f"URLs CSV not found: {urls_csv}")
        df_urls = pd.read_csv(urls_csv)
        if url_column not in df_urls.columns:
            raise ValueError(f"Column '{url_column}' not found in {urls_csv}. Available: {list(df_urls.columns)}")
        urls = [u for u in df_urls[url_column].astype(str).tolist() if isinstance(u, str) and u.strip()]
        total = len(urls)
        if end_index is None:
            end_index = total
        urls = urls[start_index:end_index]
        total_count = len(urls)
        print(f"Will scrape {total_count} coasters from URLs CSV [{start_index}:{end_index}] using column '{url_column}'...")

        all_data: List[Dict] = []

        def _scrape_one(url: str) -> Optional[Dict]:
            try:
                m = re.search(r'/coasters/(\d+)(?:/([^/]+))?', url)
                coaster_id = int(m.group(1)) if m else None
                slug = m.group(2) if (m and len(m.groups()) >= 2) else None
                if coaster_id is not None:
                    return self.extract_rating_distribution(coaster_id, coaster_slug=slug)
                soup = self.get_page(url)
                if soup:
                    canon = soup.find('link', rel='canonical')
                    href = canon.get('href') if canon else url
                    m2 = re.search(r'/coasters/(\d+)', href or '')
                    if m2:
                        cid = int(m2.group(1))
                        return self.extract_rating_distribution(cid, coaster_slug=slug)
                return None
            finally:
                if self.delay and parallel_workers == 1:
                    time.sleep(self.delay)

        if parallel_workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=parallel_workers) as ex:
                futures = {ex.submit(_scrape_one, u): i for i, u in enumerate(urls, start=1)}
                processed = 0
                for fut in as_completed(futures):
                    processed += 1
                    data = None
                    try:
                        data = fut.result()
                    except Exception as e:
                        print(f"  ✗ Error (parallel): {e}")
                    if data:
                        all_data.append(data)
                    if processed % save_interval == 0 and all_data:
                        self._save_progress(all_data, processed, output_csv)
                    print(f"[{processed}/{total_count}] processed")
        else:
            for idx, url in enumerate(urls, start=1):
                label = url
                print(f"[{idx}/{total_count}] Scraping: {label}")
                try:
                    data = _scrape_one(url)
                    if data:
                        all_data.append(data)
                        print(f"  ✓ Success - Avg rating: {data.get('avg_rating', 'N/A')}\n")
                    else:
                        print("  ✗ Failed to extract data\n")
                    if idx % save_interval == 0 and all_data:
                        self._save_progress(all_data, idx, output_csv)
                except Exception as e:
                    print(f"  ✗ Error: {e}\n")
                    continue

        df = pd.DataFrame(all_data)
        if output_csv is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_csv = f"rating_distributions_{timestamp}.csv"
        if not df.empty:
            df.to_csv(output_csv, index=False, encoding='utf-8')
            print(f"\n✓ Final data saved to {output_csv}")
            print(f"✓ Total coasters scraped: {len(df)}")
            print(f"✓ Coasters with ratings: {df['avg_rating'].notna().sum()}")
        return df

    def discover_all_coaster_urls(self,
                                  start_page: int = 1,
                                  max_pages: int = 1000) -> List[str]:
        """Discover all coaster URLs by paging through Captain Coaster listings.
        This scrapes the catalog pages to collect the `href` links to coaster detail pages.
        """
        urls: List[str] = []
        base = "https://captaincoaster.com/en/coasters?sort=rank&page={page}"
        for page in range(start_page, start_page + max_pages):
            url = base.format(page=page)
            print(f"Discovering page {page}: {url}")
            soup = self.get_page(url)
            if not soup:
                print("  ✗ Failed to load listing page; stopping discovery")
                break
            cards = soup.select('a.card')
            # Fallback selector
            if not cards:
                cards = soup.select('a[href^="/en/coasters/"]')
            page_urls = []
            for a in cards:
                href = a.get('href')
                if href and '/en/coasters/' in href:
                    full = href if href.startswith('http') else f"https://captaincoaster.com{href}"
                    page_urls.append(full)
            if not page_urls:
                print("  No more coaster links found; stopping discovery")
                break
            print(f"  Found {len(page_urls)} coaster URLs on page {page}")
            urls.extend(page_urls)
            time.sleep(self.delay)
        # Deduplicate
        urls = list(dict.fromkeys(urls))
        print(f"Discovered {len(urls)} unique coaster URLs in total")
        return urls

    def scrape_discovered(self,
                          output_csv: Optional[str] = None,
                          start_page: int = 1,
                          max_pages: int = 1000,
                          save_interval: int = 50) -> pd.DataFrame:
        """Discover coaster URLs automatically and scrape them, no CSVs required."""
        urls = self.discover_all_coaster_urls(start_page=start_page, max_pages=max_pages)
        # Write a temporary list file for transparency (optional)
        tmp_list = None
        try:
            tmp_list = os.path.join(os.getcwd(), 'discovered_coaster_urls.txt')
            with open(tmp_list, 'w', encoding='utf-8') as f:
                for u in urls:
                    f.write(u + '\n')
            print(f"  💾 Discovered URLs saved to {tmp_list}")
        except Exception:
            pass
        return self.scrape_urls_file(tmp_list or '', output_csv=output_csv, start_index=0, end_index=None, save_interval=save_interval)


def main():
    """Main execution function"""
    print("=" * 60)
    print("Captain Coaster Rating Distribution Crawler")
    print("=" * 60)
    print()
    
    # Initialize crawler
    crawler = RatingDistributionCrawler(delay=1.0)
    
    # Default file paths
    ratings_csv = r"c:\Users\Lenovo\rollercoaster\ratings_data\processed_all_reviews_metadata_20251110_161035.csv"
    url_mapping_csv = r"c:\Users\Lenovo\rollercoaster\coaster_id_to_url_mapping.csv"
    
    print("Options:")
    print("1. Scrape all coasters from ratings data")
    print("2. Scrape a specific range of coasters")
    print("3. Test with first 10 coasters")
    print("4. Scrape a single coaster by ID (requires URL)")
    print()
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == "4":
        # Single coaster test
        coaster_id = int(input("Enter coaster ID: "))
        coaster_slug = input("Enter coaster URL slug (e.g., voltron-europa-park): ").strip()
        
        print(f"\nScraping coaster {coaster_id}...")
        
        data = crawler.extract_rating_distribution(coaster_id, coaster_slug=coaster_slug)
        if data:
            print("\nResult:")
            print(json.dumps(data, indent=2))
            
            # Save to CSV
            df = pd.DataFrame([data])
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"rating_distribution_coaster_{coaster_id}_{timestamp}.csv"
            df.to_csv(filename, index=False, encoding='utf-8')
            print(f"\nSaved to {filename}")
        else:
            print("Failed to scrape coaster")
    
    else:
        # Batch scraping
        if choice == "1":
            start_idx = 0
            end_idx = None
            print(f"\nWill scrape ALL coasters (this may take several hours)")
        elif choice == "2":
            start_idx = int(input("Enter start index (0-based): "))
            end_idx = int(input("Enter end index (exclusive): "))
        else:  # choice == "3" or default
            start_idx = 0
            end_idx = 10
            print(f"\nTest mode: Will scrape first 10 coasters")
        
        # Output file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_csv = f"rating_distributions_{timestamp}.csv"
        
        print(f"\nOutput file: {output_csv}")
        print("Press Ctrl+C to stop and save progress at any time")
        print("-" * 60)
        
        # Start scraping
        df = crawler.scrape_coasters_from_csv(
            ratings_csv=ratings_csv,
            url_mapping_csv=url_mapping_csv,
            output_csv=output_csv,
            start_index=start_idx,
            end_index=end_idx,
            save_interval=50
        )
        
        if not df.empty:
            print("\n" + "=" * 60)
            print("Scraping Summary:")
            print(f"Total coasters processed: {len(df)}")
            print(f"Coasters with avg rating: {df['avg_rating'].notna().sum()}")
            print(f"Coasters with distribution data: {df['pct_5.0_stars'].notna().sum()}")
            
            if df['avg_rating'].notna().any():
                print(f"\nAverage rating statistics:")
                print(f"  Mean: {df['avg_rating'].mean():.2f}")
                print(f"  Median: {df['avg_rating'].median():.2f}")
                print(f"  Min: {df['avg_rating'].min():.2f}")
                print(f"  Max: {df['avg_rating'].max():.2f}")
            
            print("\n" + "=" * 60)
            print(f"✓ Final data saved to: {output_csv}")
            print("=" * 60)
        else:
            print("\n⚠ No data collected")


if __name__ == "__main__":
    main()
