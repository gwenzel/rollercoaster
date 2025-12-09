"""
Debug why speed jumps from 0 to 80 instantly
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from utils.acceleration import compute_acc_profile
from utils.track import build_modular_track

# Create a track with launch
track_elements = [
    {'type': 'launch', 'params': {'length': 40, 'speed_boost': 22}},  # 22 m/s = 79.2 km/h
    {'type': 'lift_hill', 'params': {'length': 40, 'height': 80}},
]

track_df = build_modular_track(track_elements)
x = track_df['x'].values
y = track_df['y'].values
z = track_df.get('z', pd.Series(np.zeros_like(x))).values
points = np.column_stack([x, z, y])

print("="*80)
print("DEBUGGING SPEED JUMP FROM 0 TO 80")
print("="*80)

print(f"\nTrack points: {len(points)}")
print(f"Launch block: length=40m, speed_boost=22 m/s = {22*3.6:.1f} km/h")

# Test with v0=0 (should start from rest)
print(f"\n1. TESTING WITH v0=0 (start from rest):")
# Create launch section: first 40m should accelerate from 0 to 22 m/s
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
v_3d = result['v_3d']

print(f"   Speed at start: {v_kmh[0]:.2f} km/h")
if len(v_kmh) > 10:
    print(f"   Speed at point 10: {v_kmh[10]:.2f} km/h")
if len(v_kmh) > 20:
    print(f"   Speed at point 20: {v_kmh[20]:.2f} km/h")
if len(v_kmh) > 30:
    print(f"   Speed at point 30: {v_kmh[30]:.2f} km/h")
print(f"   Max speed: {v_kmh.max():.2f} km/h")
print(f"   First {min(10, len(v_kmh))} speeds: {v_kmh[:min(10, len(v_kmh))]}")

# Check position differences
dp = points[1:] - points[:-1]
ds = np.linalg.norm(dp, axis=1)
print(f"\n2. POSITION DIFFERENCES:")
print(f"   First 10 ds: {ds[:10]}")
print(f"   Max ds: {ds.max():.2f} m")

# Check how velocity is calculated
print(f"\n3. VELOCITY CALCULATION:")
print(f"   v_3d[0]: {v_3d[0]}")
print(f"   ||v_3d[0]||: {np.linalg.norm(v_3d[0]):.2f} m/s = {np.linalg.norm(v_3d[0])*3.6:.2f} km/h")
print(f"   v_3d[1]: {v_3d[1]}")
print(f"   ||v_3d[1]||: {np.linalg.norm(v_3d[1]):.2f} m/s = {np.linalg.norm(v_3d[1])*3.6:.2f} km/h")

# Check if launch is causing instant speed
print(f"\n4. CHECKING LAUNCH SECTION:")
# Launch block is first, so first ~40m should be launch
launch_length = 40
cumulative_x = 0
for i, ds_val in enumerate(ds):
    cumulative_x += ds_val
    if cumulative_x <= launch_length and i < len(v_kmh):
        print(f"   Point {i}: x={cumulative_x:.1f}m, speed={v_kmh[i]:.2f} km/h")
        if i >= min(10, len(v_kmh)-1):
            break

# Check energy
print(f"\n5. ENERGY CHECK:")
h = points[:, 2]  # Z coordinate (vertical)
g = 9.81
h_initial = h[0]
KE = 0.5 * 6000.0 * v**2
PE = 6000.0 * g * (h - h_initial)
E_total = KE + PE

print(f"   Initial energy: {E_total[0]:.2f} J")
if len(E_total) > 10:
    print(f"   Energy at point 10: {E_total[10]:.2f} J")
    print(f"   Energy change: {E_total[10] - E_total[0]:.2f} J")
    cumulative_x = ds[:10].sum() if len(ds) >= 10 else ds.sum()
    print(f"   To go from 0 to {v[10]:.2f} m/s requires: {0.5 * 6000.0 * v[10]**2:.2f} J")
    if cumulative_x > 0:
        print(f"   This would require acceleration: a = v²/(2*d) = {v[10]**2/(2*cumulative_x):.2f} m/s²")
        print(f"   This is {v[10]**2/(2*cumulative_x)/9.81:.2f} g")

