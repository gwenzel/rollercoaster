"""
Test what G-forces are generated BEFORE clipping to see if we're distorting results
"""
import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from utils.track_blocks import vertical_drop_profile, loop_profile, banked_turn_profile
from utils.acceleration import compute_acc_profile

print("Testing RAW G-forces (before clipping)\n")
print("="*70)

# Test 1: Vertical Drop
print("\nTest 1: Vertical Drop (90 degrees)")
print("-"*70)
x, y, z = vertical_drop_profile(height=80, steepness=0.9)
points = np.column_stack([x, z, y])  # Map to acceleration.py convention

acc = compute_acc_profile(points, dt=0.02, v0=5.0)
print(f"RAW G-forces (before clipping):")
print(f"  Lateral:      [{acc['f_lat_g'].min():6.2f}, {acc['f_lat_g'].max():6.2f}] g")
print(f"  Vertical:     [{acc['f_vert_g'].min():6.2f}, {acc['f_vert_g'].max():6.2f}] g")
print(f"  Longitudinal: [{acc['f_long_g'].min():6.2f}, {acc['f_long_g'].max():6.2f}] g")

clipped_vert = np.clip(acc['f_vert_g'], -2.0, 6.0)
clipped_lat = np.clip(acc['f_lat_g'], -3.0, 3.0)
clipped_long = np.clip(acc['f_long_g'], -3.0, 3.0)
vert_clipped_count = np.sum((acc['f_vert_g'] < -2.0) | (acc['f_vert_g'] > 6.0))
lat_clipped_count = np.sum(np.abs(acc['f_lat_g']) > 3.0)
long_clipped_count = np.sum(np.abs(acc['f_long_g']) > 3.0)

print(f"\nPoints clipped:")
print(f"  Vertical: {vert_clipped_count}/{len(clipped_vert)} ({100*vert_clipped_count/len(clipped_vert):.1f}%)")
print(f"  Lateral: {lat_clipped_count}/{len(clipped_lat)} ({100*lat_clipped_count/len(clipped_lat):.1f}%)")
print(f"  Longitudinal: {long_clipped_count}/{len(clipped_long)} ({100*long_clipped_count/len(clipped_long):.1f}%)")

# Test 2: Vertical Loop
print("\n\nTest 2: Vertical Loop (tight 15m diameter)")
print("-"*70)
x, y, z = loop_profile(diameter=15)
points = np.column_stack([x, z, y])

acc = compute_acc_profile(points, dt=0.02, v0=25.0)  # High speed entry
print(f"RAW G-forces (before clipping):")
print(f"  Lateral:      [{acc['f_lat_g'].min():6.2f}, {acc['f_lat_g'].max():6.2f}] g")
print(f"  Vertical:     [{acc['f_vert_g'].min():6.2f}, {acc['f_vert_g'].max():6.2f}] g")
print(f"  Longitudinal: [{acc['f_long_g'].min():6.2f}, {acc['f_long_g'].max():6.2f}] g")

vert_clipped_count = np.sum((acc['f_vert_g'] < -2.0) | (acc['f_vert_g'] > 6.0))
lat_clipped_count = np.sum(np.abs(acc['f_lat_g']) > 3.0)
long_clipped_count = np.sum(np.abs(acc['f_long_g']) > 3.0)

print(f"\nPoints clipped:")
print(f"  Vertical: {vert_clipped_count}/{len(acc['f_vert_g'])} ({100*vert_clipped_count/len(acc['f_vert_g']):.1f}%)")
print(f"  Lateral: {lat_clipped_count}/{len(acc['f_lat_g'])} ({100*lat_clipped_count/len(acc['f_lat_g']):.1f}%)")
print(f"  Longitudinal: {long_clipped_count}/{len(acc['f_long_g'])} ({100*long_clipped_count/len(acc['f_long_g']):.1f}%)")

# Test 3: Banked Turn
print("\n\nTest 3: Banked Turn (30m radius)")
print("-"*70)
x, y, z = banked_turn_profile(radius=30, angle=90)
points = np.column_stack([x, z, y])

acc = compute_acc_profile(points, dt=0.02, v0=20.0)
print(f"RAW G-forces (before clipping):")
print(f"  Lateral:      [{acc['f_lat_g'].min():6.2f}, {acc['f_lat_g'].max():6.2f}] g")
print(f"  Vertical:     [{acc['f_vert_g'].min():6.2f}, {acc['f_vert_g'].max():6.2f}] g")
print(f"  Longitudinal: [{acc['f_long_g'].min():6.2f}, {acc['f_long_g'].max():6.2f}] g")

vert_clipped_count = np.sum((acc['f_vert_g'] < -2.0) | (acc['f_vert_g'] > 6.0))
lat_clipped_count = np.sum(np.abs(acc['f_lat_g']) > 3.0)
long_clipped_count = np.sum(np.abs(acc['f_long_g']) > 3.0)

print(f"\nPoints clipped:")
print(f"  Vertical: {vert_clipped_count}/{len(acc['f_vert_g'])} ({100*vert_clipped_count/len(acc['f_vert_g']):.1f}%)")
print(f"  Lateral: {lat_clipped_count}/{len(acc['f_lat_g'])} ({100*lat_clipped_count/len(acc['f_lat_g']):.1f}%)")
print(f"  Longitudinal: {long_clipped_count}/{len(acc['f_long_g'])} ({100*long_clipped_count/len(acc['f_long_g']):.1f}%)")

print("\n" + "="*70)
print("\nREAL COASTER REFERENCE:")
print("  Formula Rossa: +4.8g vertical (fastest coaster)")
print("  Intimidator 305: +5.9g vertical, -1.5g airtime")
print("  Schwarzkopf loops: +6g vertical (rare)")
print("  Most modern coasters: +4g vertical max")
print("  Red Force: -1.5g ejector airtime")
print("="*70)
