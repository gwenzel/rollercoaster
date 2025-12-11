import argparse
import os
import sys

# Ensure workspace root is on sys.path when running directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from crawler.captaincoaster.rating_distribution_crawler import RatingDistributionCrawler
def main():
    parser = argparse.ArgumentParser(description="Run Captain Coaster scraper (simple modes)")
    parser.add_argument("--url", type=str, default=None, help="Scrape a single Captain Coaster page by full URL")
    parser.add_argument("--urls_file", type=str, default=None, help="Scrape newline-separated Captain Coaster URLs from a text file")
    parser.add_argument("--urls_csv", type=str, default=None, help="Scrape URLs from a CSV file (e.g., ratings_data/complete_coaster_mapping.csv)")
    parser.add_argument("--url_column", type=str, default="coaster_url", help="Column name in --urls_csv that contains the full Captain Coaster URL")
    parser.add_argument("--ratings_csv", type=str, default=None, help="Path to ratings CSV with coaster_id/coaster_name (optional)")
    parser.add_argument("--url_mapping_csv", type=str, default=None, help="CSV mapping coaster_id to full URL (optional)")
    parser.add_argument("--output_csv", type=str, default=None, help="Output CSV filename")
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based)")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests in seconds")
    parser.add_argument("--save_interval", type=int, default=50, help="Checkpoint save interval")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel workers for scraping (URLs modes only)")

    args = parser.parse_args()

    crawler = RatingDistributionCrawler(delay=args.delay)

    if args.url:
        df = crawler.scrape_single_url(url=args.url, output_csv=args.output_csv)
    elif args.urls_file:
        df = crawler.scrape_urls_file(
            urls_file=args.urls_file,
            output_csv=args.output_csv,
            start_index=args.start,
            end_index=args.end,
            save_interval=args.save_interval,
            parallel_workers=args.parallel,
        )
    elif args.urls_csv:
        df = crawler.scrape_urls_csv(
            urls_csv=args.urls_csv,
            url_column=args.url_column,
            output_csv=args.output_csv,
            start_index=args.start,
            end_index=args.end,
            save_interval=args.save_interval,
            parallel_workers=args.parallel,
        )
    elif args.ratings_csv:
        df = crawler.scrape_coasters_from_csv(
            ratings_csv=args.ratings_csv,
            url_mapping_csv=args.url_mapping_csv,
            output_csv=args.output_csv,
            start_index=args.start,
            end_index=args.end,
            save_interval=args.save_interval,
        )
    elif args.url_mapping_csv:
        df = crawler.scrape_from_url_mapping(
            url_mapping_csv=args.url_mapping_csv,
            output_csv=args.output_csv,
            start_index=args.start,
            end_index=args.end,
            save_interval=args.save_interval,
        )
    else:
        raise SystemExit("Provide either --url, --urls_file, --url_mapping_csv, or --ratings_csv")

    print(f"Saved rows: {len(df)}")


if __name__ == "__main__":
    main()
