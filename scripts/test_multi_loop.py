"""
Test G-forces with multiple loops to see if forces are too low
"""
import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from utils.track_blocks import lift_hill_profile, vertical_drop_profile, loop_profile
from utils.accelerometer_transform import track_to_accelerometer_data

print("Testing Multi-Loop Track G-Forces\n")
print("="*70)

# Build a track with lift hill, drop, and two loops
all_x, all_y, all_z = [], [], []
offset_x, offset_y = 0.0, 0.0

# 1. Lift hill (50m high)
x, y, z = lift_hill_profile(height=50, length=60)
all_x.extend((x + offset_x).tolist())
all_y.extend((y + offset_y).tolist())
all_z.extend(z.tolist())
offset_x, offset_y = all_x[-1], all_y[-1]

# 2. Steep drop
x, y, z = vertical_drop_profile(height=45, steepness=0.8)
all_x.extend((x + offset_x).tolist())
all_y.extend((y + offset_y).tolist())
all_z.extend(z.tolist())
offset_x, offset_y = all_x[-1], all_y[-1]

# 3. First loop (20m diameter)
x, y, z = loop_profile(diameter=20)
all_x.extend((x + offset_x).tolist())
all_y.extend((y + offset_y).tolist())
all_z.extend(z.tolist())
offset_x, offset_y = all_x[-1], all_y[-1]

# 4. Second loop (18m diameter, tighter)
x, y, z = loop_profile(diameter=18)
all_x.extend((x + offset_x).tolist())
all_y.extend((y + offset_y).tolist())
all_z.extend(z.tolist())
offset_x, offset_y = all_x[-1], all_y[-1]

# 5. Third loop (15m diameter, very tight)
x, y, z = loop_profile(diameter=15)
all_x.extend((x + offset_x).tolist())
all_y.extend((y + offset_y).tolist())
all_z.extend(z.tolist())

# Convert to DataFrame
track_df = pd.DataFrame({
    'x': np.array(all_x),
    'y': np.array(all_y),
    'z': np.array(all_z)
})

print(f"Track: Lift → Drop → 3 Loops (20m, 18m, 15m diameter)")
print(f"Total points: {len(track_df)}")
print(f"Height range: [{track_df['y'].min():.1f}, {track_df['y'].max():.1f}] m")
print(f"Track length: ~{np.sum(np.sqrt(np.diff(track_df['x'])**2 + np.diff(track_df['y'])**2)):.1f} m")

# Get accelerometer data
print("\nComputing G-forces...")
accel_df = track_to_accelerometer_data(track_df)

print(f"\n{'='*70}")
print("RESULTS:")
print(f"{'='*70}")
print(f"\nLateral G-forces:")
print(f"  Range: [{accel_df['Lateral'].min():6.2f}, {accel_df['Lateral'].max():6.2f}] g")
print(f"  Mean: {accel_df['Lateral'].mean():6.2f} g")
print(f"  Std: {accel_df['Lateral'].std():6.2f} g")

print(f"\nVertical G-forces:")
print(f"  Range: [{accel_df['Vertical'].min():6.2f}, {accel_df['Vertical'].max():6.2f}] g")
print(f"  Mean: {accel_df['Vertical'].mean():6.2f} g")
print(f"  Std: {accel_df['Vertical'].std():6.2f} g")

print(f"\nLongitudinal G-forces:")
print(f"  Range: [{accel_df['Longitudinal'].min():6.2f}, {accel_df['Longitudinal'].max():6.2f}] g")
print(f"  Mean: {accel_df['Longitudinal'].mean():6.2f} g")
print(f"  Std: {accel_df['Longitudinal'].std():6.2f} g")

# Check if we're hitting clipping limits
vert_clipped = np.sum((accel_df['Vertical'] <= -10.0) | (accel_df['Vertical'] >= 10.0))
lat_clipped = np.sum((accel_df['Lateral'] <= -10.0) | (accel_df['Lateral'] >= 10.0))
long_clipped = np.sum((accel_df['Longitudinal'] <= -10.0) | (accel_df['Longitudinal'] >= 10.0))

print(f"\n{'='*70}")
print(f"Points at clipping limits (±10g):")
print(f"  Vertical: {vert_clipped}/{len(accel_df)} ({100*vert_clipped/len(accel_df):.1f}%)")
print(f"  Lateral: {lat_clipped}/{len(accel_df)} ({100*lat_clipped/len(accel_df):.1f}%)")
print(f"  Longitudinal: {long_clipped}/{len(accel_df)} ({100*long_clipped/len(accel_df):.1f}%)")

print(f"\n{'='*70}")
print("EXPECTED for tight loops:")
print("  Bottom of loop: +5 to +6g vertical (centripetal + gravity)")
print("  Top of loop: -1 to +1g vertical (centripetal - gravity)")
print("  Fast loops: +4 to +5g sustained")
print(f"{'='*70}")
