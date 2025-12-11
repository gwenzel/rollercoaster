"""
Data preparation for training BiGRU model.
Loads data from complete_coaster_mapping.csv and prepares for training.
"""

import pandas as pd
import numpy as np
import glob
from pathlib import Path
from collections import defaultdict


def load_complete_mapping(csv_path='../ratings_data/complete_coaster_mapping.csv'):
    """
    Load the complete coaster mapping CSV.
    
    Returns:
        DataFrame with coaster_name, avg_rating, full_path, match_type, etc.
    """
    df = pd.read_csv(csv_path)
    print(f"\n✓ Loaded {len(df)} coasters from complete mapping")
    return df


def filter_perfect_matches(df):
    """
    Filter for perfect matches only (≥95% similarity).
    
    Args:
        df: Complete mapping DataFrame
        
    Returns:
        Filtered DataFrame with perfect matches only
    """
    perfect = df[df['match_type'] == 'perfect'].copy()
    print(f"✓ Filtered to {len(perfect)} coasters with perfect name matches")
    return perfect


def aggregate_duplicates(df):
    """
    Average ratings for duplicate coaster names.
    Keep the entry with the most CSV files.
    
    Args:
        df: DataFrame with potential duplicates
        
    Returns:
        DataFrame with unique coaster names and averaged ratings
    """
    # Group by coaster_name
    grouped = df.groupby('coaster_name').agg({
        'avg_rating': 'mean',  # Average the ratings
        'total_ratings': 'sum',  # Sum total ratings
        'csv_count': 'max',  # Take max CSV count
        'full_path': 'first',  # Keep first path (will update below)
        'rfdb_park_folder': 'first',
        'rfdb_coaster_folder': 'first',
        'coaster_id': 'first',
        'ratings_coaster': 'first',
        'ratings_park': 'first',
        'match_type': 'first'
    }).reset_index()
    
    # For duplicates, select the path with the most CSV files
    duplicates = df['coaster_name'].value_counts()
    duplicates = duplicates[duplicates > 1].index
    
    if len(duplicates) > 0:
        print(f"\n⚠️  Found {len(duplicates)} duplicate coaster names:")
        for coaster_name in duplicates:
            dupes = df[df['coaster_name'] == coaster_name].sort_values('csv_count', ascending=False)
            best_path = dupes.iloc[0]['full_path']
            best_csv_count = dupes.iloc[0]['csv_count']
            avg_rating = dupes['avg_rating'].mean()
            
            # Update in grouped dataframe
            grouped.loc[grouped['coaster_name'] == coaster_name, 'full_path'] = best_path
            grouped.loc[grouped['coaster_name'] == coaster_name, 'csv_count'] = best_csv_count
            grouped.loc[grouped['coaster_name'] == coaster_name, 'avg_rating'] = avg_rating
            
            print(f"  - {coaster_name}: {len(dupes)} entries → avg rating {avg_rating:.2f}, using path with {best_csv_count} CSVs")
    
    print(f"\n✓ Final dataset: {len(grouped)} unique coasters")
    return grouped


def get_accelerometer_files(full_path):
    """
    Get list of accelerometer CSV files from the path.
    
    Args:
        full_path: Path like "rfdb_csvs/park/coaster"
        
    Returns:
        List of CSV file paths
    """
    csv_files = glob.glob(f"{full_path}/*.csv")
    return sorted(csv_files)  # Sort to ensure consistent ordering


def load_last_accelerometer_file(full_path):
    """
    Load the last (most recent or highest numbered) CSV file from the path.
    
    Args:
        full_path: Path like "rfdb_csvs/park/coaster"
        
    Returns:
        DataFrame with accelerometer data or None if not found
    """
    # Handle both relative paths from scripts folder
    if not full_path.startswith('..'):
        full_path = f"../{full_path}"
    
    csv_files = get_accelerometer_files(full_path)
    
    if not csv_files:
        return None
    
    # Take the last file (assuming they're sorted/numbered)
    last_file = csv_files[-1]
    
    try:
        df = pd.read_csv(last_file)
        return df
    except Exception as e:
        print(f"Error loading {last_file}: {e}")
        return None


def prepare_training_data(mapping_csv='../ratings_data/complete_coaster_mapping.csv', 
                         min_csv_count=1,
                         min_ratings=10):
    """
    Prepare complete training dataset from mapping CSV.
    
    Args:
        mapping_csv: Path to complete_coaster_mapping.csv
        min_csv_count: Minimum number of CSV files required
        min_ratings: Minimum number of ratings required
        
    Returns:
        DataFrame with coaster_name, avg_rating, full_path for training
    """
    print("\n" + "="*70)
    print("PREPARING TRAINING DATA FROM COMPLETE MAPPING")
    print("="*70)
    
    # Load mapping
    df = load_complete_mapping(mapping_csv)
    
    # Filter for perfect matches
    df = filter_perfect_matches(df)
    
    # Filter for minimum requirements
    df = df[df['csv_count'] >= min_csv_count]
    print(f"✓ Filtered to {len(df)} coasters with ≥{min_csv_count} CSV files")
    
    df = df[df['total_ratings'] >= min_ratings]
    print(f"✓ Filtered to {len(df)} coasters with ≥{min_ratings} total ratings")
    
    # Aggregate duplicates
    df = aggregate_duplicates(df)
    
    # Sort by rating for analysis
    df = df.sort_values('avg_rating', ascending=False)
    
    # Show statistics
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    print(f"Total coasters: {len(df)}")
    print(f"Rating range: {df['avg_rating'].min():.2f} - {df['avg_rating'].max():.2f}")
    print(f"Average rating: {df['avg_rating'].mean():.2f} ± {df['avg_rating'].std():.2f}")
    print(f"Total CSV files: {df['csv_count'].sum()}")
    print(f"Average CSVs per coaster: {df['csv_count'].mean():.2f}")
    
    # Show top and bottom rated
    print("\n📊 Top 5 highest rated:")
    for _, row in df.head(5).iterrows():
        print(f"   {row['coaster_name']:30s} {row['avg_rating']:.2f}★ ({row['csv_count']} CSVs)")
    
    print("\n📊 Bottom 5 lowest rated:")
    for _, row in df.tail(5).iterrows():
        print(f"   {row['coaster_name']:30s} {row['avg_rating']:.2f}★ ({row['csv_count']} CSVs)")
    
    return df


def calculate_airtime(accel_df, threshold=-0.1):
    """
    Calculate airtime statistics from accelerometer data.
    Airtime is when vertical acceleration is below threshold (negative g-force).
    
    Args:
        accel_df: DataFrame with accelerometer data (must have 'Vertical' column)
        threshold: G-force threshold for airtime (default -0.1g)
        
    Returns:
        dict with airtime statistics
    """
    if accel_df is None or 'Vertical' not in accel_df.columns:
        return {
            'total_airtime': 0,
            'max_negative_g': 0,
            'airtime_moments': 0,
            'avg_airtime_duration': 0
        }
    
    vertical = accel_df['Vertical'].values
    time_col = 'Time' if 'Time' in accel_df.columns else None
    
    # Find airtime moments
    airtime_mask = vertical < threshold
    
    # Calculate statistics
    total_airtime_samples = np.sum(airtime_mask)
    max_negative_g = np.min(vertical) if len(vertical) > 0 else 0
    
    # Count distinct airtime moments (groups of consecutive True values)
    airtime_moments = 0
    in_airtime = False
    for is_airtime in airtime_mask:
        if is_airtime and not in_airtime:
            airtime_moments += 1
            in_airtime = True
        elif not is_airtime:
            in_airtime = False
    
    # Calculate average duration (assuming 10Hz sampling = 0.1s per sample)
    sampling_rate = 10  # Hz
    total_airtime = total_airtime_samples / sampling_rate
    avg_airtime_duration = total_airtime / airtime_moments if airtime_moments > 0 else 0
    
    return {
        'total_airtime': total_airtime,
        'max_negative_g': max_negative_g,
        'airtime_moments': airtime_moments,
        'avg_airtime_duration': avg_airtime_duration
    }


if __name__ == "__main__":
    # Test the data preparation
    df = prepare_training_data()
    
    # Test loading accelerometer data and calculating airtime
    print("\n" + "="*70)
    print("TESTING ACCELEROMETER LOADING & AIRTIME CALCULATION")
    print("="*70)
    
    sample_coaster = df.iloc[0]
    print(f"\nTesting with: {sample_coaster['coaster_name']}")
    print(f"Rating: {sample_coaster['avg_rating']:.2f}")
    print(f"Path: {sample_coaster['full_path']}")
    
    accel_df = load_last_accelerometer_file(sample_coaster['full_path'])
    if accel_df is not None:
        print(f"✓ Loaded accelerometer data: {len(accel_df)} samples")
        print(f"  Columns: {list(accel_df.columns)}")
        
        airtime_stats = calculate_airtime(accel_df)
        print(f"\n🎢 Airtime Statistics:")
        print(f"  Total airtime: {airtime_stats['total_airtime']:.2f}s")
        print(f"  Max negative G: {airtime_stats['max_negative_g']:.2f}g")
        print(f"  Airtime moments: {airtime_stats['airtime_moments']}")
        print(f"  Avg moment duration: {airtime_stats['avg_airtime_duration']:.2f}s")
    else:
        print("✗ Could not load accelerometer data")
