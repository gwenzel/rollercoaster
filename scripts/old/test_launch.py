"""
Test the launch block and energy conservation with minimal initial speed
"""
import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from utils.track_blocks import lift_hill_profile, vertical_drop_profile, launch_profile, loop_profile
from utils.accelerometer_transform import track_to_accelerometer_data

print("Testing Launch Block and Energy Conservation\n")
print("="*70)

# Test 1: Track with lift hill only (no launch)
print("\nTest 1: Traditional coaster (lift hill → drop → loop)")
print("-"*70)

all_x, all_y, all_z = [], [], []
offset_x, offset_y = 0.0, 0.0

x, y, z = lift_hill_profile(height=40, length=50)
all_x.extend((x + offset_x).tolist())
all_y.extend((y + offset_y).tolist())
all_z.extend(z.tolist())
offset_x, offset_y = all_x[-1], all_y[-1]

x, y, z = vertical_drop_profile(height=35, steepness=0.8)
all_x.extend((x + offset_x).tolist())
all_y.extend((y + offset_y).tolist())
all_z.extend(z.tolist())
offset_x, offset_y = all_x[-1], all_y[-1]

x, y, z = loop_profile(diameter=25)
all_x.extend((x + offset_x).tolist())
all_y.extend((y + offset_y).tolist())
all_z.extend(z.tolist())

track_df = pd.DataFrame({'x': all_x, 'y': all_y, 'z': all_z})
accel_df = track_to_accelerometer_data(track_df)

print(f"Initial height: 40m")
print(f"Vertical G: [{accel_df['Vertical'].min():.2f}, {accel_df['Vertical'].max():.2f}] g")
print(f"Expected speed at bottom: ~{np.sqrt(2*9.81*40):.1f} m/s = {np.sqrt(2*9.81*40)*3.6:.0f} km/h")

# Test 2: Track with launch (minimal lift)
print("\n\nTest 2: Launched coaster (small lift → drop → LAUNCH → loop)")
print("-"*70)

all_x, all_y, all_z = [], [], []
offset_x, offset_y = 0.0, 0.0

x, y, z = lift_hill_profile(height=15, length=30)  # Small lift
all_x.extend((x + offset_x).tolist())
all_y.extend((y + offset_y).tolist())
all_z.extend(z.tolist())
offset_x, offset_y = all_x[-1], all_y[-1]

x, y, z = vertical_drop_profile(height=10, steepness=0.7)
all_x.extend((x + offset_x).tolist())
all_y.extend((y + offset_y).tolist())
all_z.extend(z.tolist())
offset_x, offset_y = all_x[-1], all_y[-1]

x, y, z = launch_profile(length=40, speed_boost=25)  # 25 m/s = 90 km/h
all_x.extend((x + offset_x).tolist())
all_y.extend((y + offset_y).tolist())
all_z.extend(z.tolist())
offset_x, offset_y = all_x[-1], all_y[-1]

x, y, z = loop_profile(diameter=25)
all_x.extend((x + offset_x).tolist())
all_y.extend((y + offset_y).tolist())
all_z.extend(z.tolist())

track_df2 = pd.DataFrame({'x': all_x, 'y': all_y, 'z': all_z})
accel_df2 = track_to_accelerometer_data(track_df2)

print(f"Initial height: 15m (vs 40m traditional)")
print(f"Launch: 25 m/s = 90 km/h boost")
print(f"Vertical G: [{accel_df2['Vertical'].min():.2f}, {accel_df2['Vertical'].max():.2f}] g")

print(f"\n{'='*70}")
print("COMPARISON:")
print(f"  Traditional (40m lift): {accel_df['Vertical'].max():.2f}g max")
print(f"  Launched (15m + boost):  {accel_df2['Vertical'].max():.2f}g max")
print(f"\n  Launch enables high forces with lower initial lift!")
print(f"{'='*70}")
