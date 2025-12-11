"""
Test launch acceleration to ensure it's realistic
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from utils.acceleration import compute_acc_profile
from utils.track import build_modular_track

# Create a track with launch - use more points
track_elements = [
    {'type': 'launch', 'params': {'length': 40, 'speed_boost': 22}},  # 22 m/s = 79.2 km/h
]

track_df = build_modular_track(track_elements)
x = track_df['x'].values
y = track_df['y'].values
z = track_df.get('z', pd.Series(np.zeros_like(x))).values
points = np.column_stack([x, z, y])

print("="*80)
print("TESTING LAUNCH ACCELERATION")
print("="*80)
print(f"Track points: {len(points)}")
print(f"Launch: 40m length, target speed: 22 m/s = {22*3.6:.1f} km/h")

# Create launch section
launch_sections = [(0.0, 40.0, 22.0)]  # (start_x, end_x, target_speed)

result = compute_acc_profile(
    points,
    dt=0.02,
    mass=6000.0,
    v0=0.0,
    use_energy_conservation=True,
    launch_sections=launch_sections
)

v = result['v']
v_kmh = v * 3.6

# Calculate cumulative distance
ds = np.linalg.norm(np.diff(points, axis=0, prepend=points[0:1]), axis=1)
s_cumulative = np.cumsum(ds)

print(f"\nSpeed profile during launch (first 40m):")
print(f"{'Distance (m)':>12} {'Speed (km/h)':>12} {'Expected':>12}")
print(f"{'-'*40}")

# Check first 10 points or until we leave launch
for i in range(min(10, len(v))):
    dist = s_cumulative[i]
    speed = v_kmh[i]
    # Expected speed from constant acceleration: v² = 2*a*d
    # a = v_target² / (2*length) = 22² / (2*40) = 6.05 m/s²
    if dist <= 40:
        expected_v = np.sqrt(2 * (22**2 / (2*40)) * dist) * 3.6
        print(f"{dist:>12.1f} {speed:>12.2f} {expected_v:>12.2f}")
    else:
        print(f"{dist:>12.1f} {speed:>12.2f} {'(post-launch)':>12}")

print(f"\nEnergy check:")
print(f"  To accelerate from 0 to 22 m/s over 40m:")
print(f"  Required acceleration: a = v²/(2*d) = {22**2/(2*40):.2f} m/s² = {22**2/(2*40)/9.81:.2f} g")
print(f"  Required energy: KE = 0.5*m*v² = {0.5*6000*22**2:.0f} J")
print(f"  Power (if 2s launch): P = E/t = {0.5*6000*22**2/2:.0f} W = {0.5*6000*22**2/2/1000:.0f} kW")

# Check if speed increases gradually
speed_changes = np.diff(v_kmh)
print(f"\nSpeed changes:")
print(f"  Max jump: {speed_changes.max():.2f} km/h per point")
print(f"  Mean change: {speed_changes.mean():.2f} km/h per point")
print(f"  First 5 changes: {speed_changes[:5]}")

if speed_changes.max() > 50:
    print(f"\n[WARNING] Speed is jumping too fast! Max jump: {speed_changes.max():.2f} km/h")
    print(f"  This suggests instant acceleration, which requires infinite energy")
else:
    print(f"\n[OK] Speed increases gradually")

