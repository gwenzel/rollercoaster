"""
Add Steel Vengeance from RFDB to the leaderboard.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.cloud_data_loader import load_rfdb_csv, list_rfdb_csvs
from utils.submission_manager import add_submission_to_leaderboard, load_submissions
from app_builder import check_gforce_safety
from utils.lgbm_predictor import predict_score_lgb


def add_steel_vengeance():
    """
    Add Steel Vengeance from RFDB to the leaderboard.
    """
    print("="*70)
    print("ADDING STEEL VENGEANCE TO LEADERBOARD")
    print("="*70)
    
    park = "cedarpoint"
    coaster = "steelvengeance"
    
    # Check if already in leaderboard
    existing_submissions = load_submissions()
    existing_ids = {s.get('submission_id') for s in existing_submissions}
    
    # Get CSV files for Steel Vengeance
    print(f"\nLoading CSV files for {coaster} ({park})...")
    csv_files = list_rfdb_csvs(park, coaster, use_cloud=True)
    
    if not csv_files:
        # Try local fallback
        rfdb_root = os.path.join(os.path.dirname(__file__), '..', 'rfdb_csvs')
        coaster_path = os.path.join(rfdb_root, park, coaster)
        if os.path.exists(coaster_path):
            csv_files = [f for f in os.listdir(coaster_path) if f.endswith('.csv')]
        else:
            print(f"Error: Could not find {coaster} at {park}")
            return
    
    if not csv_files:
        print(f"Error: No CSV files found for {coaster}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Use the first CSV file (or you could pick the best one)
    csv_file = csv_files[0]
    print(f"Using CSV file: {csv_file}")
    
    submission_id = f"rfdb_{park}_{coaster}_{csv_file}".replace(' ', '_').replace('/', '_')
    submission_id = submission_id.replace('.csv', '').replace(':', '-')
    
    # Check if already exists
    if submission_id in existing_ids:
        print(f"\nSteel Vengeance already in leaderboard with ID: {submission_id}")
        print("Use rerun_rfdb_scores.py to update its scores.")
        return
    
    try:
        # Load CSV
        print(f"\nLoading CSV data...")
        df = load_rfdb_csv(park, coaster, csv_file, use_cloud=True)
        if df is None:
            # Try local fallback
            rfdb_root = os.path.join(os.path.dirname(__file__), '..', 'rfdb_csvs')
            csv_path = os.path.join(rfdb_root, park, coaster, csv_file)
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
            else:
                print(f"Error: Could not load CSV file")
                return
        
        # Resolve columns
        cols_lower = {c.lower(): c for c in df.columns}
        def resolve_any(candidates):
            for cand in candidates:
                if cand.lower() in cols_lower:
                    return cols_lower[cand.lower()]
            return None
        
        time_col = resolve_any(['Time','time','t','timestamp','elapsed','seconds','s'])
        vert_col = resolve_any(['Vertical','vertical','vert','zforce','g_vert','gvertical','gz','accel_z','az'])
        lat_col = resolve_any(['Lateral','lateral','lat','xforce','g_lat','glateral','gx','accel_x','ax'])
        long_col = resolve_any(['Longitudinal','longitudinal','long','yforce','g_long','glongitudinal','gy','accel_y','ay'])
        
        if not all([vert_col, lat_col, long_col]):
            print(f"Error: Could not resolve required columns")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Create accel_df
        accel_df = pd.DataFrame({
            'Time': df[time_col] if time_col else np.arange(len(df)),
            'Vertical': df[vert_col],
            'Lateral': df[lat_col],
            'Longitudinal': df[long_col],
        })
        
        print(f"Loaded {len(accel_df)} data points")
        
        # Calculate safety score
        print("Calculating safety score...")
        safety = check_gforce_safety(accel_df)
        safety_score = safety['safety_score']
        
        # Calculate fun rating with LightGBM
        print("Calculating fun rating with LightGBM...")
        fun_rating = predict_score_lgb(accel_df)
        
        # Create submission name
        submitter_name = f"RFDB: Steel Vengeance (Cedar Point)"
        
        # Additional metadata
        metadata = {
            'source': 'RFDB',
            'park': park,
            'coaster': coaster,
            'csv_file': csv_file
        }
        
        timestamp = datetime.now().isoformat()
        
        # Add to leaderboard
        print(f"\nAdding to leaderboard...")
        success = add_submission_to_leaderboard(
            submitter_name=submitter_name,
            score=fun_rating,
            safety_score=safety_score,
            submission_id=submission_id,
            timestamp=timestamp,
            metadata=metadata
        )
        
        if success:
            print(f"\n{'='*70}")
            print(f"SUCCESS!")
            print(f"  Coaster: Steel Vengeance")
            print(f"  Park: Cedar Point")
            print(f"  Fun Rating: {fun_rating:.2f}")
            print(f"  Safety Score: {safety_score:.2f}")
            print(f"  Submission ID: {submission_id}")
            print(f"{'='*70}")
        else:
            print(f"\nError: Failed to add to leaderboard")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    add_steel_vengeance()

