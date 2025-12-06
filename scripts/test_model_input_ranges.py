"""
Test to verify the acceleration ranges the BiGRU model expects.
Compare against what we're actually generating.
"""
import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from utils.track import build_modular_track
from utils.track_blocks import (
    lift_hill_profile, vertical_drop_profile, loop_profile,
    airtime_hill_profile, spiral_profile, bunny_hop_profile
)
from utils.accelerometer_transform import track_to_accelerometer_data
from utils.bigru_predictor import predict_score_bigru

def test_track(name, elements):
    """Test a single track configuration."""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print('='*70)
    
    try:
        track_df = build_modular_track(elements)
        print(f"Track: {len(track_df)} points")
        print(f"Height range: {track_df['y'].min():.1f} to {track_df['y'].max():.1f} m")
        
        # Get acceleration data
        accel_df = track_to_accelerometer_data(track_df)
        
        # Print ranges
        print(f"\nAcceleration Ranges (in g):")
        print(f"  Lateral:      [{accel_df['Lateral'].min():6.2f}, {accel_df['Lateral'].max():6.2f}]")
        print(f"  Vertical:     [{accel_df['Vertical'].min():6.2f}, {accel_df['Vertical'].max():6.2f}]")
        print(f"  Longitudinal: [{accel_df['Longitudinal'].min():6.2f}, {accel_df['Longitudinal'].max():6.2f}]")
        
        # Check for clipping (values exactly at limits)
        clipped_count = 0
        if np.any(np.abs(accel_df['Lateral']) >= 2.99):
            clipped_count += np.sum(np.abs(accel_df['Lateral']) >= 2.99)
            print(f"  ⚠ Lateral CLIPPED at ±3g: {np.sum(np.abs(accel_df['Lateral']) >= 2.99)} points")
        if np.any(accel_df['Vertical'] <= -1.99) or np.any(accel_df['Vertical'] >= 5.99):
            count = np.sum((accel_df['Vertical'] <= -1.99) | (accel_df['Vertical'] >= 5.99))
            clipped_count += count
            print(f"  ⚠ Vertical CLIPPED at -2g/+6g: {count} points")
        if np.any(np.abs(accel_df['Longitudinal']) >= 2.99):
            count = np.sum(np.abs(accel_df['Longitudinal']) >= 2.99)
            clipped_count += count
            print(f"  ⚠ Longitudinal CLIPPED at ±3g: {count} points")
        
        if clipped_count == 0:
            print("  OK: No clipping detected")
        
        # Predict score (pass track_df, not accel_df!)
        score = predict_score_bigru(track_df)
        print(f"\nPredicted Score: {score:.2f} / 5.0")
        
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

# Test cases covering different intensities
test_cases = [
    ("Gentle Family Coaster", [
        {'type': 'climb', 'params': {'length': 30, 'height': 20}},
        {'type': 'drop', 'params': {'length': 40, 'angle': 40}},
        {'type': 'hills', 'params': {'num_hills': 2, 'amplitude': 5, 'wavelength': 25}},
    ]),
    
    ("Medium Thrill Ride", [
        {'type': 'climb', 'params': {'length': 40, 'height': 40}},
        {'type': 'drop', 'params': {'length': 60, 'angle': 60}},
        {'type': 'loop', 'params': {'radius': 12}},
        {'type': 'hills', 'params': {'num_hills': 2, 'amplitude': 10, 'wavelength': 30}},
    ]),
    
    ("Extreme Coaster", [
        {'type': 'climb', 'params': {'length': 50, 'height': 60}},
        {'type': 'drop', 'params': {'length': 80, 'angle': 80}},
        {'type': 'loop', 'params': {'radius': 15}},
        {'type': 'loop', 'params': {'radius': 10}},
        {'type': 'hills', 'params': {'num_hills': 4, 'amplitude': 12, 'wavelength': 35}},
    ]),
]

# Test using track_blocks directly (what the app uses)
def stitch_blocks(blocks):
    """Stitch blocks together like the app does."""
    all_x, all_y = [], []
    current_x, current_y = 0, 0
    
    for func, params in blocks:
        if 'current_height' in params:
            params['current_height'] = current_y
        x_rel, y_rel = func(**params)
        x_abs = x_rel + current_x
        y_abs = y_rel + current_y
        all_x.extend(x_abs)
        all_y.extend(y_abs)
        current_x, current_y = all_x[-1], all_y[-1]
    
    return pd.DataFrame({'x': all_x, 'y': all_y})

blocks_test_cases = [
    ("Lift + Drop + Loop (App Style)", [
        (lift_hill_profile, {'length': 50, 'height': 40}),
        (vertical_drop_profile, {'height': 40, 'steepness': 0.9}),
        (loop_profile, {'diameter': 30}),
    ]),
    
    ("Multiple Elements (App Style)", [
        (lift_hill_profile, {'length': 60, 'height': 50}),
        (vertical_drop_profile, {'height': 45, 'steepness': 0.85}),
        (loop_profile, {'diameter': 25}),
        (airtime_hill_profile, {'length': 40, 'height': 15}),
        (bunny_hop_profile, {'length': 20, 'height': 8}),
    ]),
]

print("\n" + "="*70)
print("BiGRU MODEL INPUT RANGE VALIDATION")
print("="*70)

# Test legacy track builder
print("\n### TESTING: Legacy Track Builder (track.py)")
for name, elements in test_cases:
    test_track(name, elements)

# Test track_blocks (app builder)
print("\n\n### TESTING: Track Blocks (app_builder.py)")
for name, blocks in blocks_test_cases:
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print('='*70)
    
    try:
        track_df = stitch_blocks(blocks)
        print(f"Track: {len(track_df)} points")
        print(f"Height range: {track_df['y'].min():.1f} to {track_df['y'].max():.1f} m")
        
        accel_df = track_to_accelerometer_data(track_df)
        
        print(f"\nAcceleration Ranges (in g):")
        print(f"  Lateral:      [{accel_df['Lateral'].min():6.2f}, {accel_df['Lateral'].max():6.2f}]")
        print(f"  Vertical:     [{accel_df['Vertical'].min():6.2f}, {accel_df['Vertical'].max():6.2f}]")
        print(f"  Longitudinal: [{accel_df['Longitudinal'].min():6.2f}, {accel_df['Longitudinal'].max():6.2f}]")
        
        # Check clipping
        clipped = []
        if np.any(np.abs(accel_df['Lateral']) >= 2.99):
            clipped.append(f"Lateral: {np.sum(np.abs(accel_df['Lateral']) >= 2.99)} pts")
        if np.any((accel_df['Vertical'] <= -1.99) | (accel_df['Vertical'] >= 5.99)):
            clipped.append(f"Vertical: {np.sum((accel_df['Vertical'] <= -1.99) | (accel_df['Vertical'] >= 5.99))} pts")
        if np.any(np.abs(accel_df['Longitudinal']) >= 2.99):
            clipped.append(f"Longitudinal: {np.sum(np.abs(accel_df['Longitudinal']) >= 2.99)} pts")
        
        if clipped:
            print(f"  WARNING - CLIPPING: {', '.join(clipped)}")
        else:
            print("  OK: No clipping")
        
        score = predict_score_bigru(track_df)
        print(f"\nPredicted Score: {score:.2f} / 5.0")
        
    except Exception as e:
        print(f"FAILED: {e}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("If you see significant clipping, the physics engine is producing")
print("unrealistic values that need to be fixed at the source, not just clipped.")
print("="*70)
