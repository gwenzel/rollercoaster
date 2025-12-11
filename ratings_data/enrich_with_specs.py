"""
Ratings Data: Enrich scraped rating distributions with physical specs

This script loads a rating distributions CSV (from ratings_data workflow),
fetches each coaster page, extracts Height (m), Speed (km/h), and Length (m),
and writes an augmented CSV alongside the original.

Usage:
    python enrich_with_specs.py --input star_ratings_per_rc/rating_distributions_full_YYYYMMDD_HHMMSS.csv

Outputs:
    star_ratings_per_rc/rating_distributions_full_YYYYMMDD_HHMMSS_enriched.csv
"""
import argparse
import os
import re
import sys
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

DEFAULT_DELAY = 0.8
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0 Safari/537.36'
}

def get_page(url: str) -> Optional[BeautifulSoup]:
    try:
        resp = requests.get(url, timeout=25, headers=HEADERS)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, 'html.parser')
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


def _to_float(num_str: str) -> Optional[float]:
    try:
        return float(str(num_str).replace(',', '.'))
    except Exception:
        return None


def extract_specs(soup: BeautifulSoup) -> Dict[str, float]:
    """Extract Height (m), Speed (km/h), Length (m) from page text."""
    text = soup.get_text(separator='\n')
    specs: Dict[str, float] = {}

    # Height (m)
    m = re.search(r'(?:Height|Total\s*height|Max\s*height)[^\d]*(\d+(?:[\.,]\d+)?)\s*m', text, flags=re.IGNORECASE)
    if m:
        val = _to_float(m.group(1))
        if val is not None:
            specs['height_m'] = val

    # Speed (km/h or mph)
    m_kmh = re.search(r'(?:Speed|Top\s*speed)[^\d]*(\d+(?:[\.,]\d+)?)\s*(?:km/?h|kmh|kph)', text, flags=re.IGNORECASE)
    m_mph = re.search(r'(?:Speed|Top\s*speed)[^\d]*(\d+(?:[\.,]\d+)?)\s*mph', text, flags=re.IGNORECASE)
    if m_kmh:
        val = _to_float(m_kmh.group(1))
        if val is not None:
            specs['speed_kmh'] = val
    elif m_mph:
        val = _to_float(m_mph.group(1))
        if val is not None:
            specs['speed_kmh'] = round(val * 1.60934, 2)

    # Length (m)
    m_len = re.search(r'(?:Length|Track\s*length|Total\s*length)[^\d]*(\d+(?:[\.,]\d+)?)\s*m', text, flags=re.IGNORECASE)
    if m_len:
        val = _to_float(m_len.group(1))
        if val is not None:
            specs['track_length_m'] = val

    return specs


def enrich_csv(input_csv: str) -> str:
    df = pd.read_csv(input_csv)
    if 'url' not in df.columns:
        raise ValueError("Input CSV must contain a 'url' column for each coaster page.")

    print(f"Loaded {len(df)} rows from {input_csv}")

    session = requests.Session()
    session.headers.update(HEADERS)

    height_vals = []
    speed_vals = []
    length_vals = []

    for i, row in df.iterrows():
        url = row.get('url')
        if not isinstance(url, str) or not url.startswith('http'):
            height_vals.append(None)
            speed_vals.append(None)
            length_vals.append(None)
            continue
        soup = get_page(url)
        if not soup:
            height_vals.append(None)
            speed_vals.append(None)
            length_vals.append(None)
            continue
        specs = extract_specs(soup)
        height_vals.append(specs.get('height_m'))
        speed_vals.append(specs.get('speed_kmh'))
        length_vals.append(specs.get('track_length_m'))

    df['height_m'] = height_vals
    df['speed_kmh'] = speed_vals
    df['track_length_m'] = length_vals

    base, ext = os.path.splitext(input_csv)
    output_csv = f"{base}_enriched{ext}"
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Saved enriched CSV to {output_csv}")
    return output_csv


def main():
    parser = argparse.ArgumentParser(description='Enrich rating distributions CSV with coaster specs.')
    parser.add_argument('--input', required=True, help='Path to rating distributions CSV (from ratings_data workflow).')
    args = parser.parse_args()

    # Change to ratings_data directory if invoked elsewhere
    if not os.getcwd().endswith('ratings_data'):
        try:
            os.chdir(os.path.join(os.path.dirname(__file__)))
        except Exception:
            pass

    enrich_csv(args.input)


if __name__ == '__main__':
    main()
