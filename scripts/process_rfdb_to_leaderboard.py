"""
Process all RFDB CSV files and add them to the leaderboard.
Calculates safety scores and estimates fun ratings from accelerometer data.
"""

import os
import sys
import pandas as pd
import numpy as np
import random
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.cloud_data_loader import list_rfdb_parks, list_rfdb_coasters, list_rfdb_csvs, load_rfdb_csv
from utils.submission_manager import add_submission_to_leaderboard, update_submission_in_leaderboard, load_submissions
from app_builder import check_gforce_safety, compute_airtime_metrics
from utils.lgbm_predictor import predict_score_lgb


def _heuristic_fun_rating(accel_df):
    if accel_df is None or len(accel_df) < 10:
        return 2.5  # Default middle rating
    
    # Calculate metrics
    max_vertical = accel_df['Vertical'].max()
    min_vertical = accel_df['Vertical'].min()
    max_lateral = accel_df['Lateral'].abs().max()
    max_longitudinal = accel_df['Longitudinal'].abs().max()
    
    # Airtime metrics
    airtime_metrics = compute_airtime_metrics(accel_df)
    total_airtime = airtime_metrics.get('total_airtime', 0)
    ejector_time = airtime_metrics.get('ejector', 0)
    
    # Duration
    if 'Time' in accel_df.columns:
        duration = accel_df['Time'].iloc[-1] - accel_df['Time'].iloc[0]
    else:
        duration = len(accel_df) * 0.02  # Assume 20ms sampling
    
    # G-force variety (range)
    vertical_range = max_vertical - min_vertical
    
    # Heuristic scoring (0-5 scale)
    score = 2.5  # Start at middle
    
    # Airtime bonus (up to +1.5)
    if total_airtime > 0:
        airtime_score = min(1.5, total_airtime / 10.0)  # 10s airtime = +1.5
        score += airtime_score
    
    # Ejector airtime bonus (up to +0.5)
    if ejector_time > 0:
        ejector_score = min(0.5, ejector_time / 3.0)  # 3s ejector = +0.5
        score += ejector_score
    
    # Intensity bonus (moderate g-forces are fun, but not too extreme)
    # Positive vertical g (up to +0.8)
    if 2.0 <= max_vertical <= 4.5:
        intensity_bonus = min(0.8, (max_vertical - 2.0) / 2.5)
        score += intensity_bonus
    elif max_vertical > 4.5:
        # Too intense, slight penalty
        score -= 0.3
    
    # G-force variety bonus (up to +0.5)
    if vertical_range > 3.0:
        variety_bonus = min(0.5, (vertical_range - 3.0) / 4.0)
        score += variety_bonus
    
    # Duration bonus (longer rides are generally better, up to +0.3)
    if duration > 60:
        duration_bonus = min(0.3, (duration - 60) / 120.0)
        score += duration_bonus
    
    # Penalty for excessive forces (safety concerns reduce fun)
    if max_vertical > 5.0 or min_vertical < -3.0:
        score -= 0.5
    
    # Clamp to 0-5 range
    score = max(0.0, min(5.0, score))
    
    return score


def estimate_fun_rating_from_accelerations(accel_df):
    """
    Estimate fun rating using the LightGBM extreme model.
    Falls back to the heuristic scorer if the model is unavailable.
    """
    try:
        return predict_score_lgb(accel_df)
    except Exception as e:
        print(f"[fallback] LightGBM prediction failed: {e}")
        return _heuristic_fun_rating(accel_df)


def process_rfdb_tracks(max_rfdb_submissions=100):
    """
    Process RFDB tracks and add to leaderboard.
    Randomly samples max_rfdb_submissions tracks from all available RFDB data.
    
    Args:
        max_rfdb_submissions: Maximum number of RFDB submissions to keep (default: 100)
    """
    print("Loading RFDB parks...")
    parks = list_rfdb_parks(use_cloud=True)
    
    if not parks:
        # Try local fallback
        rfdb_root = os.path.join(os.path.dirname(__file__), '..', 'rfdb_csvs')
        if os.path.exists(rfdb_root):
            parks = [d for d in os.listdir(rfdb_root) if os.path.isdir(os.path.join(rfdb_root, d))]
        else:
            print("No RFDB data found!")
            return
    
    print(f"Found {len(parks)} parks")
    
    # First, collect all available RFDB tracks
    print("\nCollecting all available RFDB tracks...")
    all_rfdb_tracks = []
    
    for park in parks:
        coasters = list_rfdb_coasters(park, use_cloud=True)
        if not coasters:
            # Try local fallback
            rfdb_root = os.path.join(os.path.dirname(__file__), '..', 'rfdb_csvs')
            park_path = os.path.join(rfdb_root, park)
            if os.path.exists(park_path):
                coasters = [d for d in os.listdir(park_path) if os.path.isdir(os.path.join(park_path, d))]
            else:
                continue
        
        for coaster in coasters:
            csv_files = list_rfdb_csvs(park, coaster, use_cloud=True)
            if not csv_files:
                # Try local fallback
                rfdb_root = os.path.join(os.path.dirname(__file__), '..', 'rfdb_csvs')
                coaster_path = os.path.join(rfdb_root, park, coaster)
                if os.path.exists(coaster_path):
                    csv_files = [f for f in os.listdir(coaster_path) if f.endswith('.csv')]
                else:
                    continue
            
            for csv_file in csv_files:
                submission_id = f"rfdb_{park}_{coaster}_{csv_file}".replace(' ', '_').replace('/', '_')
                submission_id = submission_id.replace('.csv', '').replace(':', '-')
                all_rfdb_tracks.append((park, coaster, csv_file, submission_id))
    
    print(f"Found {len(all_rfdb_tracks)} total RFDB tracks")
    
    # Randomly sample max_rfdb_submissions tracks
    if len(all_rfdb_tracks) > max_rfdb_submissions:
        selected_tracks = random.sample(all_rfdb_tracks, max_rfdb_submissions)
        print(f"Randomly selected {max_rfdb_submissions} tracks to process")
    else:
        selected_tracks = all_rfdb_tracks
        print(f"Processing all {len(selected_tracks)} tracks (less than {max_rfdb_submissions})")
    
    # Get IDs of selected tracks
    selected_ids = {track[3] for track in selected_tracks}
    
    # Remove any existing RFDB submissions that aren't in the selected set
    print("\nCleaning up existing RFDB submissions...")
    existing_submissions = load_submissions()
    rfdb_submissions_to_remove = [
        s for s in existing_submissions 
        if s.get('source') == 'RFDB' and s.get('submission_id') not in selected_ids
    ]
    
    if rfdb_submissions_to_remove:
        from utils.submission_manager import _get_submissions_dir
        import json
        from datetime import datetime
        
        submissions_dir = _get_submissions_dir()
        leaderboard_file = os.path.join(submissions_dir, 'leaderboard.json')
        
        # Load leaderboard
        with open(leaderboard_file, 'r', encoding='utf-8') as f:
            leaderboard_data = json.load(f)
            existing_submissions = leaderboard_data.get('submissions', [])
        
        # Remove RFDB submissions not in selected set
        existing_submissions = [
            s for s in existing_submissions 
            if not (s.get('source') == 'RFDB' and s.get('submission_id') not in selected_ids)
        ]
        
        # Save updated leaderboard
        leaderboard_data = {
            'last_updated': datetime.now().isoformat(),
            'submissions': existing_submissions
        }
        
        with open(leaderboard_file, 'w', encoding='utf-8') as f:
            json.dump(leaderboard_data, f, indent=2)
        
        print(f"Removed {len(rfdb_submissions_to_remove)} RFDB submissions not in selected set")
    
    total_processed = 0
    total_added = 0
    errors = 0
    
    # Load existing submissions again (after cleanup)
    existing_submissions = load_submissions()
    existing_ids = {s['submission_id'] for s in existing_submissions}
    
    # Process only the selected tracks
    print(f"\nProcessing {len(selected_tracks)} selected RFDB tracks...")
    for track_idx, (park, coaster, csv_file, submission_id) in enumerate(selected_tracks, 1):
        print(f"[{track_idx}/{len(selected_tracks)}] Processing: {coaster} ({park}) - {csv_file}")
        total_processed += 1
        
        # Check if already exists - we'll update it instead of skipping
        is_update = submission_id in existing_ids
        
        try:
            # Load CSV
            df = load_rfdb_csv(park, coaster, csv_file, use_cloud=True)
            if df is None:
                # Try local fallback
                rfdb_root = os.path.join(os.path.dirname(__file__), '..', 'rfdb_csvs')
                csv_path = os.path.join(rfdb_root, park, coaster, csv_file)
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                else:
                    errors += 1
                    continue
            
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
                errors += 1
                continue
            
            # Create accel_df
            accel_df = pd.DataFrame({
                'Time': df[time_col] if time_col else np.arange(len(df)),
                'Vertical': df[vert_col],
                'Lateral': df[lat_col],
                'Longitudinal': df[long_col],
            })
            
            # Calculate safety score
            safety = check_gforce_safety(accel_df)
            safety_score = safety['safety_score']
            
            # Estimate fun rating
            fun_rating = estimate_fun_rating_from_accelerations(accel_df)
            
            # Create submission name
            submitter_name = f"RFDB: {coaster} ({park})"
            
            # Add directly to leaderboard (no geometry file needed)
            from datetime import datetime
            
            timestamp = datetime.now().isoformat()
            
            # Additional metadata
            metadata = {
                'source': 'RFDB',  # Mark as RFDB source
                'park': park,
                'coaster': coaster,
                'csv_file': csv_file
            }
            
            # Add or update in leaderboard
            if is_update:
                # Update existing submission with new scores
                success = update_submission_in_leaderboard(
                    submission_id=submission_id,
                    score=fun_rating,
                    safety_score=safety_score,
                    metadata=metadata
                )
                if success:
                    total_added += 1
                    if total_added % 10 == 0:
                        print(f"    Updated {total_added} submissions so far...")
            else:
                # Add new submission
                success = add_submission_to_leaderboard(
                    submitter_name=submitter_name,
                    score=fun_rating,
                    safety_score=safety_score,
                    submission_id=submission_id,
                    timestamp=timestamp,
                    metadata=metadata
                )
                
                if success:
                    total_added += 1
                    existing_ids.add(submission_id)
                    if total_added % 10 == 0:
                        print(f"    Added {total_added} submissions so far...")
        
        except Exception as e:
            errors += 1
            print(f"    Error processing {csv_file}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Total processed: {total_processed}")
    print(f"  Successfully added: {total_added}")
    print(f"  Errors: {errors}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import sys
    max_submissions = 500  # Default to 300
    if len(sys.argv) > 1:
        try:
            max_submissions = int(sys.argv[1])
        except ValueError:
            print(f"Invalid argument: {sys.argv[1]}. Using default: {max_submissions}")
    process_rfdb_tracks(max_rfdb_submissions=max_submissions)

