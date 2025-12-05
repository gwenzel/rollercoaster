from crawler.captaincoaster.rating_distribution_crawler import RatingDistributionCrawler
import json

if __name__ == "__main__":
    crawler = RatingDistributionCrawler(delay=0.5)
    # Steel Vengeance: ID 2268, slug 'steel-vengeance-cedar-point'
    data = crawler.extract_rating_distribution(2268, coaster_name="Steel Vengeance", coaster_slug="steel-vengeance-cedar-point")
    print(json.dumps(data, indent=2))
