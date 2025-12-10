"""
Rerun scores for all existing submissions in the leaderboard.
This script loads all entries (both RFDB and user submissions), recomputes their 
fun and safety scores using the current LightGBM model, and updates the leaderboard.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.cloud_data_loader import load_rfdb_csv
from utils.submission_manager import load_submissions, update_submission_in_leaderboard, load_submission_geometry
from app_builder import check_gforce_safety
from utils.lgbm_predictor import predict_score_lgb
from utils.accelerometer_transform import track_to_accelerometer_data


def rerun_all_scores():
    """
    Rerun scores for all existing submissions in the leaderboard (both RFDB and user submissions).
    """
    print("="*70)
    print("RERUNNING ALL SCORES WITH LIGHTGBM MODEL")
    print("="*70)
    
    # Load all submissions
    print("\nLoading existing submissions...")
    all_submissions = load_submissions()
    
    # Separate RFDB and non-RFDB submissions
    rfdb_submissions = [
        s for s in all_submissions 
        if s.get('source') == 'RFDB'
    ]
    
    non_rfdb_submissions = [
        s for s in all_submissions 
        if s.get('source') != 'RFDB'
    ]
    
    print(f"Found {len(rfdb_submissions)} RFDB submissions")
    print(f"Found {len(non_rfdb_submissions)} user submissions")
    print(f"Total: {len(all_submissions)} submissions to update")
    
    total_updated = 0
    total_errors = 0
    
    # Process RFDB submissions
    if rfdb_submissions:
        print(f"\n{'='*70}")
        print(f"PROCESSING RFDB SUBMISSIONS ({len(rfdb_submissions)} entries)")
        print(f"{'='*70}")
        
        for idx, submission in enumerate(rfdb_submissions, 1):
            submission_id = submission.get('submission_id')
            # Metadata fields are stored directly in submission, not nested
            park = submission.get('park')
            coaster = submission.get('coaster')
            csv_file = submission.get('csv_file')
            
            if not all([park, coaster, csv_file]):
                print(f"[{idx}/{len(rfdb_submissions)}] Skipping {submission_id}: Missing metadata (park={park}, coaster={coaster}, csv_file={csv_file})")
                total_errors += 1
                continue
            
            print(f"\n[{idx}/{len(rfdb_submissions)}] Processing: {coaster} ({park}) - {csv_file}")
            
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
                        print(f"    [ERROR] Could not load CSV file")
                        total_errors += 1
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
                    print(f"    [ERROR] Could not resolve required columns")
                    total_errors += 1
                    continue
                
                # Create accel_df
                accel_df = pd.DataFrame({
                    'Time': df[time_col] if time_col else np.arange(len(df)),
                    'Vertical': df[vert_col],
                    'Lateral': df[lat_col],
                    'Longitudinal': df[long_col],
                })
                
                # Recalculate safety score
                safety = check_gforce_safety(accel_df)
                safety_score = safety['safety_score']
                
                # Recalculate fun rating with LightGBM
                fun_rating = predict_score_lgb(accel_df)
                
                # Prepare metadata dict for update (keep all existing fields)
                metadata = {
                    'source': 'RFDB',
                    'park': park,
                    'coaster': coaster,
                    'csv_file': csv_file
                }
                
                # Update submission
                success = update_submission_in_leaderboard(
                    submission_id=submission_id,
                    score=fun_rating,
                    safety_score=safety_score,
                    metadata=metadata
                )
                
                if success:
                    total_updated += 1
                    print(f"    [OK] Updated: Fun={fun_rating:.2f}, Safety={safety_score:.2f}")
                else:
                    print(f"    [FAIL] Failed to update submission")
                    total_errors += 1
            
            except Exception as e:
                print(f"    [ERROR] {e}")
                total_errors += 1
                continue
    
    # Process non-RFDB (user) submissions
    if non_rfdb_submissions:
        print(f"\n{'='*70}")
        print(f"PROCESSING USER SUBMISSIONS ({len(non_rfdb_submissions)} entries)")
        print(f"{'='*70}")
        
        for idx, submission in enumerate(non_rfdb_submissions, 1):
            submission_id = submission.get('submission_id')
            submitter_name = submission.get('submitter_name', 'Unknown')
            
            print(f"\n[{idx}/{len(non_rfdb_submissions)}] Processing: {submitter_name} ({submission_id})")
            
            try:
                # Load geometry from submission file
                geometry = load_submission_geometry(submission_id)
                
                if geometry is None:
                    print(f"    [SKIP] Could not load geometry for submission")
                    total_errors += 1
                    continue
                
                # Convert geometry to track DataFrame
                track_df = pd.DataFrame({
                    'x': geometry.get('x', []),
                    'y': geometry.get('y', []),
                    'z': geometry.get('z', [])
                })
                
                if len(track_df) < 10:
                    print(f"    [SKIP] Track too short ({len(track_df)} points)")
                    total_errors += 1
                    continue
                
                # Convert track to accelerometer data
                accel_df = track_to_accelerometer_data(
                    track_df,
                    mass=500.0,  # Default mass
                    rho=1.2,     # Default air density
                    Cd=0.1,      # Default drag coefficient
                    A=2.0,       # Default frontal area
                    mu=0.001     # Default friction
                )
                
                if accel_df is None or len(accel_df) < 10:
                    print(f"    [SKIP] Could not generate accelerometer data")
                    total_errors += 1
                    continue
                
                # Recalculate safety score
                safety = check_gforce_safety(accel_df)
                safety_score = safety['safety_score']
                
                # Compute metadata from track geometry
                x_track = np.array(geometry.get('x', []))
                y_track = np.array(geometry.get('y', []))
                z_track = np.array(geometry.get('z', []))
                
                # Track length: 3D arc length
                dx = np.diff(x_track, prepend=x_track[0] if len(x_track) > 0 else 0)
                dy = np.diff(y_track, prepend=y_track[0] if len(y_track) > 0 else 0)
                dz = np.diff(z_track, prepend=z_track[0] if len(z_track) > 0 else 0)
                track_length_m = float(np.sum(np.sqrt(dx**2 + dy**2 + dz**2)))
                
                # Max height
                height_m = float(np.max(y_track)) if len(y_track) > 0 else 0.0
                
                # Max speed: estimate from energy conservation
                max_height_drop = float(np.max(y_track) - np.min(y_track)) if len(y_track) > 0 else 0.0
                g = 9.81
                energy_efficiency = 0.95
                max_speed_ms = np.sqrt(2 * g * max_height_drop * energy_efficiency) if max_height_drop > 0 else 0.0
                speed_kmh = float(max_speed_ms * 3.6)
                
                metadata = {
                    'height_m': height_m,
                    'speed_kmh': speed_kmh,
                    'track_length_m': track_length_m
                }
                
                # Recalculate fun rating with LightGBM (with metadata)
                fun_rating = predict_score_lgb(accel_df, metadata=metadata)
                
                # Update submission (no metadata needed for user submissions)
                success = update_submission_in_leaderboard(
                    submission_id=submission_id,
                    score=fun_rating,
                    safety_score=safety_score
                )
                
                if success:
                    total_updated += 1
                    print(f"    [OK] Updated: Fun={fun_rating:.2f}, Safety={safety_score:.2f}")
                else:
                    print(f"    [FAIL] Failed to update submission")
                    total_errors += 1
            
            except Exception as e:
                print(f"    [ERROR] {e}")
                import traceback
                traceback.print_exc()
                total_errors += 1
                continue
    
    print(f"\n{'='*70}")
    print(f"RERUN COMPLETE!")
    print(f"  Total submissions: {len(all_submissions)}")
    print(f"    - RFDB: {len(rfdb_submissions)}")
    print(f"    - User: {len(non_rfdb_submissions)}")
    print(f"  Successfully updated: {total_updated}")
    print(f"  Errors: {total_errors}")
    print(f"{'='*70}")


if __name__ == "__main__":
    rerun_all_scores()

