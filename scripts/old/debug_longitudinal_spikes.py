"""
Debug longitudinal forces at block junctions to identify spike causes
"""
import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.track_blocks import lift_hill_profile, vertical_drop_profile, loop_profile, flat_section_profile
from utils.accelerometer_transform import track_to_accelerometer_data

print("Debugging Longitudinal Force Spikes at Block Junctions\n")
print("="*70)

# Build a simple track with clear junctions
all_x, all_y, all_z = [], [], []
junction_indices = []  # Track where blocks join
block_names = []
offset_x, offset_y = 0.0, 0.0

blocks = [
    ("Lift", lift_hill_profile(height=30, length=40)),
    ("Drop", vertical_drop_profile(height=25, steepness=0.8)),
    ("Flat", flat_section_profile(length=30)),
    ("Loop", loop_profile(diameter=25)),
]

for name, (x, y, z) in blocks:
    block_names.append(name)
    junction_indices.append(len(all_x))  # Mark start of this block
    all_x.extend((x + offset_x).tolist())
    all_y.extend((y + offset_y).tolist())
    all_z.extend(z.tolist())
    offset_x, offset_y = all_x[-1], all_y[-1]

junction_indices.append(len(all_x))  # Mark end

track_df = pd.DataFrame({'x': all_x, 'y': all_y, 'z': all_z})
accel_df = track_to_accelerometer_data(track_df)

print(f"Track: {' → '.join(block_names)}")
print(f"Total points: {len(track_df)}")
print(f"\nBlock boundaries (indices):")
for i, (idx, name) in enumerate(zip(junction_indices[:-1], block_names)):
    end_idx = junction_indices[i+1] - 1
    print(f"  {name}: {idx} to {end_idx}")

print(f"\n{'='*70}")
print("Longitudinal Forces:")
print(f"  Overall range: [{accel_df['Longitudinal'].min():.2f}, {accel_df['Longitudinal'].max():.2f}] g")

# Find spikes
threshold = 2.0  # g
spikes = np.where(np.abs(accel_df['Longitudinal']) > threshold)[0]
print(f"\n  Points with |longitudinal| > {threshold}g: {len(spikes)}")

# Check if spikes occur near junctions
if len(spikes) > 0:
    print(f"\n  Spike locations:")
    for spike_idx in spikes[:10]:  # Show first 10
        # Find which block this spike is in
        for i, (start, end) in enumerate(zip(junction_indices[:-1], junction_indices[1:])):
            if start <= spike_idx < end:
                dist_to_junction = min(spike_idx - start, end - 1 - spike_idx)
                print(f"    Index {spike_idx}: {accel_df['Longitudinal'].iloc[spike_idx]:6.2f}g " +
                      f"in {block_names[i]} (dist to junction: {dist_to_junction} points)")
                break

# Check velocity discontinuities at junctions
print(f"\n{'='*70}")
print("Checking for velocity discontinuities at junctions:")
for i, (idx, name) in enumerate(zip(junction_indices[1:-1], block_names[1:]), start=1):
    # Check 5 points before and after junction
    window = 5
    before_idx = max(0, idx - window)
    after_idx = min(len(track_df), idx + window)
    
    # Check position continuity
    pos_jump_x = track_df['x'].iloc[idx] - track_df['x'].iloc[idx-1] if idx > 0 else 0
    pos_jump_y = track_df['y'].iloc[idx] - track_df['y'].iloc[idx-1] if idx > 0 else 0
    
    # Check longitudinal force jump
    long_before = accel_df['Longitudinal'].iloc[idx-1] if idx > 0 else 0
    long_after = accel_df['Longitudinal'].iloc[idx]
    long_jump = long_after - long_before
    
    print(f"\n  Junction {i} (before {block_names[i]}):")
    print(f"    Position jump: Δx={pos_jump_x:.4f}m, Δy={pos_jump_y:.4f}m")
    print(f"    Longitudinal jump: {long_before:.2f}g → {long_after:.2f}g (Δ={long_jump:.2f}g)")

print(f"\n{'='*70}")
print("\nPOSSIBLE CAUSES:")
print("  1. Hermite blending changes curvature abruptly")
print("  2. Speed calculation has discontinuities at junctions")
print("  3. Tangent vector estimation is noisy at junctions")
print("="*70)
