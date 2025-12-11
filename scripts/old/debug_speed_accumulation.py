"""Debug speed accumulation issue"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from utils.acceleration import compute_acc_profile
from utils.track import build_modular_track

# Create a test track
track_elements = [
    {'type': 'lift_hill', 'params': {'length': 40, 'height': 80}},
    {'type': 'drop', 'params': {'length': 50, 'angle': 60}},
]

track_df = build_modular_track(track_elements)
x = track_df['x'].values
y = track_df['y'].values
z = track_df.get('z', pd.Series(np.zeros_like(x))).values
points = np.column_stack([x, z, y])

print("Testing speed calculation...")
result = compute_acc_profile(
    points,
    dt=0.02,
    v0=3.0,
    use_energy_conservation=False,
    use_velocity_verlet=True
)

v = result['v']
v_3d = result['v_3d']
a_tot = result['a_tot']

print(f"\nSpeed statistics:")
print(f"  Min: {v.min():.2f} m/s")
print(f"  Max: {v.max():.2f} m/s")
print(f"  Mean: {v.mean():.2f} m/s")
print(f"  First 10: {v[:10]}")
print(f"  Last 10: {v[-10:]}")

print(f"\n3D Velocity statistics:")
print(f"  v_3d[0]: {v_3d[0]}")
print(f"  v_3d[-1]: {v_3d[-1]}")
print(f"  ||v_3d[0]||: {np.linalg.norm(v_3d[0]):.2f}")
print(f"  ||v_3d[-1]||: {np.linalg.norm(v_3d[-1]):.2f}")

# Check if speed is monotonically increasing (accumulating)
is_increasing = np.all(np.diff(v) >= -1e-6)  # Allow small numerical errors
print(f"\nSpeed is monotonically increasing: {is_increasing}")

# Calculate velocity from position differences (alternative method)
ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
# Use average speed between points
v_from_pos = np.zeros(len(points))
v_from_pos[0] = v[0]
for i in range(1, len(points)):
    # Time step from distance and average speed
    v_avg = 0.5 * (v[i-1] + v[i]) if i > 0 else v[i]
    if v_avg > 0.1:
        dt_actual = ds[i-1] / v_avg
        v_from_pos[i] = ds[i-1] / dt_actual if dt_actual > 1e-6 else v_from_pos[i-1]
    else:
        v_from_pos[i] = v_from_pos[i-1]

print(f"\nVelocity from position differences:")
print(f"  First 10: {v_from_pos[:10]}")
print(f"  Last 10: {v_from_pos[-10:]}")

