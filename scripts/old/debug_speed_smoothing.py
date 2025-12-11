"""
Debug why speed has sharp edges - check if energy conservation creates discontinuities
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.acceleration import compute_acc_profile
from utils.track import build_modular_track
from scipy.ndimage import gaussian_filter1d

# Create a track with multiple blocks
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
print("DEBUGGING SPEED SMOOTHING")
print("="*80)

# Apply same smoothing as in compute_acc_profile
points_smooth = points.copy()
sigma = 2.0
for i in range(3):
    points_smooth[:, i] = gaussian_filter1d(points[:, i], sigma=sigma, mode='nearest')

# Calculate height profile
h = points_smooth[:, 2]  # Z is vertical
h_initial = h[0]
energy_efficiency = 0.95

# Energy conservation speed
v0 = 0.0
v_energy = np.sqrt(np.maximum(0, v0**2 + 2 * 9.81 * (h_initial - h) * energy_efficiency))

# Check for sharp transitions in height
dh = np.diff(h)
d2h = np.diff(dh)
sharp_height_indices = np.where(np.abs(d2h) > 0.1)[0]

# Check for sharp transitions in speed
dv_energy = np.diff(v_energy)
d2v_energy = np.diff(dv_energy)
sharp_speed_indices = np.where(np.abs(d2v_energy) > 0.5)[0]

print(f"\n1. HEIGHT PROFILE:")
print(f"   Height range: [{h.min():.2f}, {h.max():.2f}] m")
print(f"   Max d²h/dx²: {np.abs(d2h).max():.3f} m")
print(f"   Sharp height transitions: {len(sharp_height_indices)}")

print(f"\n2. ENERGY CONSERVATION SPEED:")
print(f"   Speed range: [{v_energy.min():.2f}, {v_energy.max():.2f}] m/s")
print(f"   Max d²v/dx²: {np.abs(d2v_energy).max():.3f} m/s")
print(f"   Sharp speed transitions: {len(sharp_speed_indices)}")

# Now test with actual compute_acc_profile
launch_sections = [(0.0, 40.0, 22.0)]
result = compute_acc_profile(
    points,
    dt=0.02,
    mass=6000.0,
    v0=0.0,
    use_energy_conservation=True,
    launch_sections=launch_sections
)

v_final = result['v']
dv_final = np.diff(v_final)
d2v_final = np.diff(dv_final)
sharp_final_indices = np.where(np.abs(d2v_final) > 0.5)[0]

print(f"\n3. FINAL SPEED (from compute_acc_profile):")
print(f"   Speed range: [{v_final.min():.2f}, {v_final.max():.2f}] m/s")
print(f"   Max d²v/dx²: {np.abs(d2v_final).max():.3f} m/s")
print(f"   Sharp speed transitions: {len(sharp_final_indices)}")

# Compare energy vs final
speed_diff = np.abs(v_final - v_energy)
print(f"\n4. SPEED COMPARISON:")
print(f"   Max difference (final vs energy): {speed_diff.max():.2f} m/s")
print(f"   Mean difference: {speed_diff.mean():.2f} m/s")

# Check if additional smoothing would help
h_smooth2 = gaussian_filter1d(h, sigma=1.0, mode='nearest')
v_energy_smooth2 = np.sqrt(np.maximum(0, v0**2 + 2 * 9.81 * (h_initial - h_smooth2) * energy_efficiency))
d2v_smooth2 = np.diff(np.diff(v_energy_smooth2))
print(f"\n5. ADDITIONAL SMOOTHING TEST:")
print(f"   With extra smoothing (sigma=1.0 on height):")
print(f"   Max d²v/dx²: {np.abs(d2v_smooth2).max():.3f} m/s")
print(f"   Sharp transitions: {len(np.where(np.abs(d2v_smooth2) > 0.5)[0])}")

# Create visualization
fig, axes = plt.subplots(4, 1, figsize=(12, 14))

# Plot 1: Height
axes[0].plot(x, h, 'b-', linewidth=2, label='Smoothed height')
axes[0].set_xlabel('X (m)')
axes[0].set_ylabel('Height (m)')
axes[0].set_title('Height Profile (Smoothed)')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Plot 2: Height second derivative
axes[1].plot(x[2:], d2h, 'r-', linewidth=1)
axes[1].axhline(y=0.1, color='orange', linestyle='--', label='Sharp threshold')
axes[1].axhline(y=-0.1, color='orange', linestyle='--')
axes[1].set_xlabel('X (m)')
axes[1].set_ylabel('d²h/dx²')
axes[1].set_title('Height Curvature (Sharp Transitions)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Speed
axes[2].plot(x, v_energy*3.6, 'g-', linewidth=2, label='Energy conservation')
axes[2].plot(x, v_final*3.6, 'b--', linewidth=1, alpha=0.7, label='Final (with launch)')
if len(sharp_speed_indices) > 0:
    axes[2].scatter(x[sharp_speed_indices], v_energy[sharp_speed_indices]*3.6, 
                   color='red', s=30, zorder=5, label='Sharp transitions')
axes[2].set_xlabel('X (m)')
axes[2].set_ylabel('Speed (km/h)')
axes[2].set_title('Speed Profile')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# Plot 4: Speed second derivative
axes[3].plot(x[2:], np.abs(d2v_energy)*3.6, 'orange', linewidth=1, label='Energy conservation')
axes[3].plot(x[2:], np.abs(d2v_final)*3.6, 'blue', linewidth=1, alpha=0.7, label='Final')
axes[3].axhline(y=0.5*3.6, color='r', linestyle='--', label='Sharp threshold')
axes[3].set_xlabel('X (m)')
axes[3].set_ylabel('|d²v/dx²| (km/h)')
axes[3].set_title('Speed Curvature (Sharp Transitions)')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('debug_speed_smoothing.png', dpi=150)
print(f"\n[OK] Plot saved to: debug_speed_smoothing.png")

