"""
Test smoothing through the full pipeline (track_to_accelerometer_data)
"""
import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from utils.track_blocks import lift_hill_profile, vertical_drop_profile, loop_profile, flat_section_profile
from utils.accelerometer_transform import track_to_accelerometer_data

print("Testing Smoothness with Full Pipeline\n")
print("="*70)

# Build a smooth track
all_x, all_y, all_z = [], [], []
offset_x, offset_y = 0.0, 0.0

blocks = [
    ("Lift", lift_hill_profile(height=30, length=40)),
    ("Drop", vertical_drop_profile(height=25, steepness=0.8)),
    ("Flat", flat_section_profile(length=30)),
    ("Loop", loop_profile(diameter=25)),
]

for name, (x, y, z) in blocks:
    all_x.extend((x + offset_x).tolist())
    all_y.extend((y + offset_y).tolist())
    all_z.extend(z.tolist())
    offset_x, offset_y = all_x[-1], all_y[-1]

track_df = pd.DataFrame({'x': all_x, 'y': all_y, 'z': all_z})
accel_df = track_to_accelerometer_data(track_df)

print(f"Track: Lift → Drop → Flat → Loop")
print(f"Total points: {len(track_df)}")

print(f"\n{'='*70}")
print("Acceleration Results:")
print(f"{'='*70}")

print(f"\nVertical G-forces:")
print(f"  Range: [{accel_df['Vertical'].min():.2f}, {accel_df['Vertical'].max():.2f}] g")
print(f"  Mean: {accel_df['Vertical'].mean():.2f} g")
print(f"  Std dev: {accel_df['Vertical'].std():.3f}")

# Count rapid changes (> 0.5g between consecutive samples)
vert_changes = np.abs(np.diff(accel_df['Vertical']))
large_changes = np.sum(vert_changes > 0.5)
print(f"  Large changes (>0.5g): {large_changes}")
print(f"  Max change: {vert_changes.max():.2f}g")

print(f"\nLateral G-forces:")
print(f"  Range: [{accel_df['Lateral'].min():.2f}, {accel_df['Lateral'].max():.2f}] g")
print(f"  Std dev: {accel_df['Lateral'].std():.3f}")
lat_changes = np.abs(np.diff(accel_df['Lateral']))
print(f"  Large changes (>0.5g): {np.sum(lat_changes > 0.5)}")

print(f"\nLongitudinal G-forces:")
print(f"  Range: [{accel_df['Longitudinal'].min():.2f}, {accel_df['Longitudinal'].max():.2f}] g")
print(f"  Std dev: {accel_df['Longitudinal'].std():.3f}")
long_changes = np.abs(np.diff(accel_df['Longitudinal']))
print(f"  Large changes (>0.5g): {np.sum(long_changes > 0.5)}")

print(f"\n{'='*70}")
if large_changes < 5:
    print("✓ SMOOTH - Few rapid changes, good dynamics")
elif large_changes < 15:
    print("⚠️  MODERATE - Some oscillations present")
else:
    print("⚠️  NOISY - Many rapid changes, needs more smoothing")
print(f"{'='*70}")
