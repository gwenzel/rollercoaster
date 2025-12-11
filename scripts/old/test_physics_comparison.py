"""
Compare the realistic physics engine vs energy conservation method
"""
import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from utils.track import build_modular_track
from utils.accelerometer_transform import compute_rider_accelerations
from utils.acceleration import compute_acc_profile

# Create a test track
track_elements = [
    {'type': 'climb', 'params': {'length': 30, 'height': 50}},
    {'type': 'drop', 'params': {'length': 60, 'angle': 70}},
    {'type': 'loop', 'params': {'radius': 10}},
    {'type': 'hills', 'params': {'num_hills': 3, 'amplitude': 8, 'wavelength': 30}},
]

print("Building track...")
track_df = build_modular_track(track_elements)
print(f"Track: {len(track_df)} points")

# Method 1: Energy Conservation (old)
print("\n" + "="*70)
print("METHOD 1: Energy Conservation (Ideal Physics)")
print("="*70)
try:
    accel_old = compute_rider_accelerations(track_df)
    print(f"✓ Success")
    print(f"  Lateral: [{accel_old['Lateral'].min():.2f}, {accel_old['Lateral'].max():.2f}] g")
    print(f"  Vertical: [{accel_old['Vertical'].min():.2f}, {accel_old['Vertical'].max():.2f}] g")
    print(f"  Longitudinal: [{accel_old['Longitudinal'].min():.2f}, {accel_old['Longitudinal'].max():.2f}] g")
except Exception as e:
    print(f"✗ Failed: {e}")

# Method 2: Realistic Physics (new)
print("\n" + "="*70)
print("METHOD 2: Realistic Physics (Friction + Drag)")
print("="*70)
try:
    x = track_df['x'].values
    y = track_df['y'].values
    z = track_df.get('z', pd.Series(np.zeros_like(x))).values
    points = np.column_stack([x, z, y])
    
    acc_result = compute_acc_profile(
        points,
        dt=0.02,
        mass=6000.0,
        rho=1.3,
        Cd=0.6,
        A=4.0,
        mu=0.02,
        v0=1.0
    )
    
    print(f"✓ Success")
    print(f"  Lateral: [{acc_result['f_lat_g'].min():.2f}, {acc_result['f_lat_g'].max():.2f}] g")
    print(f"  Vertical: [{acc_result['f_vert_g'].min():.2f}, {acc_result['f_vert_g'].max():.2f}] g")
    print(f"  Longitudinal: [{acc_result['f_long_g'].min():.2f}, {acc_result['f_long_g'].max():.2f}] g")
    print(f"  Speed range: [{acc_result['v'].min():.1f}, {acc_result['v'].max():.1f}] m/s")
    
    # Check if train stalls
    if np.any(acc_result['v'] < 0.1):
        first_stall = np.where(acc_result['v'] < 0.1)[0][0]
        print(f"  ⚠ Train stalls at point {first_stall}/{len(track_df)}")
    else:
        print(f"  ✓ No stalling detected")
        
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
