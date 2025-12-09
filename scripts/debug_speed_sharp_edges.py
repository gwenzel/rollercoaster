"""
Debug sharp edges in speed plot - check if smoothing is applied correctly
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.acceleration import compute_acc_profile
from utils.track import build_modular_track

# Create a track with multiple blocks to see transitions
track_elements = [
    {'type': 'launch', 'params': {'length': 40, 'speed_boost': 22}},
    {'type': 'lift_hill', 'params': {'length': 40, 'height': 80}},
    {'type': 'drop', 'params': {'height': 80, 'steepness': 0.8}},
    {'type': 'loop', 'params': {'radius': 15}},
]

track_df = build_modular_track(track_elements)
x = track_df['x'].values
y = track_df['y'].values
z = track_df.get('z', pd.Series(np.zeros_like(x))).values
points = np.column_stack([x, z, y])

print("="*80)
print("DEBUGGING SHARP EDGES IN SPEED PLOT")
print("="*80)
print(f"Track points: {len(points)}")

# Check raw geometry for sharp transitions
print(f"\n1. RAW GEOMETRY CHECK:")
print(f"   X range: [{x.min():.1f}, {x.max():.1f}] m")
print(f"   Y range: [{y.min():.1f}, {y.max():.1f}] m")
print(f"   Z range: [{z.min():.1f}, {z.max():.1f}] m")

# Check for sharp transitions in height
dy = np.diff(y)
dx = np.diff(x)
slope = dy / (dx + 1e-9)
slope_change = np.diff(slope)
sharp_indices = np.where(np.abs(slope_change) > 0.5)[0]

print(f"\n2. SHARP TRANSITIONS IN GEOMETRY:")
print(f"   Max slope change: {np.abs(slope_change).max():.3f}")
print(f"   Number of sharp transitions (|d²y/dx²| > 0.5): {len(sharp_indices)}")
if len(sharp_indices) > 0:
    print(f"   Sharp transitions at indices: {sharp_indices[:10]}...")
    print(f"   Corresponding x positions: {x[sharp_indices[:10]]}")

# Test with launch sections
launch_sections = [(0.0, 40.0, 22.0)]
result = compute_acc_profile(
    points,
    dt=0.02,
    mass=6000.0,
    v0=0.0,
    use_energy_conservation=True,
    launch_sections=launch_sections
)

v = result['v']
v_3d = result['v_3d']
v_estimate = v  # This is what's used

# Check speed changes
dv = np.diff(v)
sharp_speed_indices = np.where(np.abs(dv) > 2.0)[0]  # Speed jumps > 2 m/s

print(f"\n3. SHARP TRANSITIONS IN SPEED:")
print(f"   Speed range: [{v.min():.2f}, {v.max():.2f}] m/s")
print(f"   Max speed change per point: {np.abs(dv).max():.2f} m/s")
print(f"   Number of sharp speed changes (|dv| > 2 m/s): {len(sharp_speed_indices)}")
if len(sharp_speed_indices) > 0:
    print(f"   Sharp speed changes at indices: {sharp_speed_indices[:10]}...")
    print(f"   Corresponding speeds: {v[sharp_speed_indices[:10]]}")

# Check if smoothing is applied
from scipy.ndimage import gaussian_filter1d
points_smooth = points.copy()
sigma = 2.0
for i in range(3):
    points_smooth[:, i] = gaussian_filter1d(points[:, i], sigma=sigma, mode='nearest')

# Compare smoothed vs unsmoothed
y_diff = np.abs(points_smooth[:, 2] - points[:, 2])  # Z is vertical (y in original)
max_smooth_diff = y_diff.max()
mean_smooth_diff = y_diff.mean()

print(f"\n4. SMOOTHING CHECK:")
print(f"   Max smoothing difference (height): {max_smooth_diff:.3f} m")
print(f"   Mean smoothing difference (height): {mean_smooth_diff:.3f} m")
print(f"   Smoothing sigma: {sigma}")

# Check velocity calculation
dp_smooth = points_smooth[1:] - points_smooth[:-1]
dp_raw = points[1:] - points[:-1]
ds_smooth = np.linalg.norm(dp_smooth, axis=1)
ds_raw = np.linalg.norm(dp_raw, axis=1)

print(f"\n5. VELOCITY CALCULATION:")
print(f"   Using smoothed points for velocity: YES")
print(f"   Max ds difference (smoothed vs raw): {np.abs(ds_smooth - ds_raw).max():.3f} m")
print(f"   Mean ds difference: {np.abs(ds_smooth - ds_raw).mean():.3f} m")

# Create plot to visualize
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: Geometry
axes[0].plot(x, y, 'b-', linewidth=2, label='Raw geometry')
axes[0].plot(points_smooth[:, 0], points_smooth[:, 2], 'r--', linewidth=1, alpha=0.7, label='Smoothed geometry')
axes[0].set_xlabel('X (m)')
axes[0].set_ylabel('Height (m)')
axes[0].set_title('Track Geometry: Raw vs Smoothed')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Speed
axes[1].plot(x, v*3.6, 'g-', linewidth=2, label='Speed (km/h)')
if len(sharp_speed_indices) > 0:
    axes[1].scatter(x[sharp_speed_indices], v[sharp_speed_indices]*3.6, 
                   color='red', s=50, zorder=5, label='Sharp transitions')
axes[1].set_xlabel('X (m)')
axes[1].set_ylabel('Speed (km/h)')
axes[1].set_title('Speed Profile')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Speed derivative
axes[2].plot(x[1:], np.abs(dv)*3.6, 'orange', linewidth=1)
axes[2].axhline(y=2.0*3.6, color='r', linestyle='--', label='Sharp threshold (2 m/s)')
axes[2].set_xlabel('X (m)')
axes[2].set_ylabel('|dSpeed| (km/h per point)')
axes[2].set_title('Speed Change Rate')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('debug_speed_sharp_edges.png', dpi=150)
print(f"\n[OK] Plot saved to: debug_speed_sharp_edges.png")

