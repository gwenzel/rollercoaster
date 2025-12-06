"""
Test the full pipeline: track with banking -> acceleration data -> lateral Gs
"""
import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from utils.track_blocks import banked_turn_profile, spiral_profile
from utils.accelerometer_transform import track_to_accelerometer_data

print("Testing lateral G-force generation with banking\n")
print("="*70)

# Test 1: Banked Turn
print("\nTest 1: Banked Turn")
print("-"*70)
x, y, z = banked_turn_profile(radius=30, angle=90)
print(f"Generated {len(x)} points")
print(f"Z (banking) range: [{z.min():.2f}, {z.max():.2f}] m")

track_df = pd.DataFrame({'x': x, 'y': y, 'z': z})
accel_df = track_to_accelerometer_data(track_df)

print(f"\nAcceleration data:")
print(f"  Lateral:      [{accel_df['Lateral'].min():6.2f}, {accel_df['Lateral'].max():6.2f}] g")
print(f"  Vertical:     [{accel_df['Vertical'].min():6.2f}, {accel_df['Vertical'].max():6.2f}] g")
print(f"  Longitudinal: [{accel_df['Longitudinal'].min():6.2f}, {accel_df['Longitudinal'].max():6.2f}] g")

if np.any(np.abs(accel_df['Lateral']) > 0.01):
    print(f"  ✓ SUCCESS: Non-zero lateral Gs detected!")
else:
    print(f"  ✗ FAILURE: Lateral Gs are zero!")

# Test 2: Spiral
print("\n\nTest 2: Spiral (Corkscrew)")
print("-"*70)
x, y, z = spiral_profile(diameter=25, turns=1.5)
print(f"Generated {len(x)} points")
print(f"Z (banking) range: [{z.min():.2f}, {z.max():.2f}] m")

track_df = pd.DataFrame({'x': x, 'y': y, 'z': z})
accel_df = track_to_accelerometer_data(track_df)

print(f"\nAcceleration data:")
print(f"  Lateral:      [{accel_df['Lateral'].min():6.2f}, {accel_df['Lateral'].max():6.2f}] g")
print(f"  Vertical:     [{accel_df['Vertical'].min():6.2f}, {accel_df['Vertical'].max():6.2f}] g")
print(f"  Longitudinal: [{accel_df['Longitudinal'].min():6.2f}, {accel_df['Longitudinal'].max():6.2f}] g")

if np.any(np.abs(accel_df['Lateral']) > 0.01):
    print(f"  ✓ SUCCESS: Non-zero lateral Gs detected!")
else:
    print(f"  ✗ FAILURE: Lateral Gs are zero!")

print("\n" + "="*70)
