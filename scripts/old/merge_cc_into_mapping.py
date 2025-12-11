import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Merge scraped Captain Coaster data into enhanced mapping CSV")
    parser.add_argument("--mapping_csv", type=str, required=True, help="Path to enhanced mapping CSV (will be updated)")
    parser.add_argument("--scrape_csv", type=str, required=True, help="Path to scraped ratings CSV to merge")
    parser.add_argument("--output_csv", type=str, default=None, help="Optional output path; if omitted, overwrites mapping")
    parser.add_argument("--on", type=str, default="coaster_id", help="Join key: coaster_id or url")
    args = parser.parse_args()

    mapping_csv = args.mapping_csv
    scrape_csv = args.scrape_csv
    output_csv = args.output_csv or mapping_csv
    join_key = args.on

    if not os.path.exists(mapping_csv):
        raise FileNotFoundError(f"Mapping CSV not found: {mapping_csv}")
    if not os.path.exists(scrape_csv):
        raise FileNotFoundError(f"Scrape CSV not found: {scrape_csv}")

    map_df = pd.read_csv(mapping_csv)
    scr_df = pd.read_csv(scrape_csv)

    # Normalize join key presence
    if join_key == "coaster_id":
        if "coaster_id" not in map_df.columns:
            raise ValueError("mapping_csv missing 'coaster_id' column")
        if "coaster_id" not in scr_df.columns:
            # Try to derive coaster_id from URL when absent
            if "url" in scr_df.columns:
                scr_df = scr_df.copy()
                scr_df["coaster_id"] = scr_df["url"].str.extract(r"/coasters/(\d+)").astype(float).astype("Int64")
            else:
                raise ValueError("scrape_csv missing 'coaster_id' column and 'url' not available to derive it")
    elif join_key == "url":
        if "url" not in map_df.columns:
            raise ValueError("mapping_csv missing 'url' column")
        if "url" not in scr_df.columns:
            # Construct url from id if available
            if "coaster_id" in scr_df.columns:
                scr_df = scr_df.copy()
                scr_df["url"] = scr_df["coaster_id"].apply(lambda cid: f"https://captaincoaster.com/en/coasters/{cid}" if pd.notna(cid) else None)
            else:
                raise ValueError("scrape_csv missing 'url' column and 'coaster_id' not available to construct it")
    else:
        raise ValueError("--on must be either 'coaster_id' or 'url'")

    # Columns to merge from scrape
    rating_cols = [
        "avg_rating","total_ratings",
        "pct_0.5_stars","pct_1.0_stars","pct_1.5_stars","pct_2.0_stars","pct_2.5_stars",
        "pct_3.0_stars","pct_3.5_stars","pct_4.0_stars","pct_4.5_stars","pct_5.0_stars",
        "count_0.5_stars","count_1.0_stars","count_1.5_stars","count_2.0_stars","count_2.5_stars",
        "count_3.0_stars","count_3.5_stars","count_4.0_stars","count_4.5_stars","count_5.0_stars",
        "scraped_at",
    ]
    spec_cols = ["height_m","speed_kmh","track_length_m","inversions_count"]

    available_cols = [c for c in rating_cols + spec_cols if c in scr_df.columns]
    if not available_cols:
        print("No expected rating/spec columns found in scrape CSV; proceeding with available fields.")
        available_cols = [c for c in scr_df.columns if c not in ("coaster_name",)]

    # Prepare right-hand frame with join key and selected columns
    rhs = scr_df[[join_key] + available_cols].drop_duplicates(join_key)

    # Merge (left keep mapping, update/append new columns)
    merged = map_df.merge(rhs, on=join_key, how="left", suffixes=("", ""))

    # Write output
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    merged.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"✓ Merged CSV written to: {output_csv}")
    print(f"  Rows: {len(merged)} | New cols merged: {len(available_cols)}")


if __name__ == "__main__":
    main()
