"""Debug speed profile in detail."""
import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
from utils.track_blocks import lift_hill_profile, vertical_drop_profile, loop_profile, airtime_hill_profile, flat_section_profile
from utils.acceleration import compute_acc_profile

# Build default track
x1, y1, z1 = lift_hill_profile(40, 30)
x2, y2, z2 = vertical_drop_profile(30, 0.85)
x2 += x1[-1]; y2 += y1[-1]
x3, y3, z3 = loop_profile(25)
x3 += x2[-1]; y3 += y2[-1]
x4, y4, z4 = airtime_hill_profile(35, 12)
x4 += x3[-1]; y4 += y3[-1]
x5, y5, z5 = flat_section_profile(40)
x5 += x4[-1]; y5 += y4[-1]

x = np.concatenate([x1, x2[1:], x3[1:], x4[1:], x5[1:]])
y = np.concatenate([y1, y2[1:], y3[1:], y4[1:], y5[1:]])
z = np.concatenate([z1, z2[1:], z3[1:], z4[1:], z5[1:]])

points = np.column_stack([x, z, y])

result = compute_acc_profile(
    points,
    dt=0.02,
    mass=6000,
    rho=1.3,
    Cd=0.15,
    A=4.0,
    mu=0.003,
    v0=3.0,
    use_energy_conservation=True
)

v_kmh = result['v'] * 3.6
time = np.arange(len(v_kmh)) * 0.02

# Print some sample speeds
print("Speed profile samples:")
for i in [0, 10, 20, 50, 100, 150, 200, 250, 300]:
    if i < len(v_kmh):
        print(f"  t={time[i]:.1f}s: h={y[i]:.1f}m, v={v_kmh[i]:.1f} km/h")

print(f"\nHeight profile:")
print(f"  Start: {y[0]:.1f}m")
print(f"  Max: {y.max():.1f}m at index {np.argmax(y)}")
print(f"  End: {y[-1]:.1f}m")

# Find where lift ends
h_max = y.max()
lift_end_idx = 0
for i in range(1, min(len(y), int(len(y) * 0.3))):
    if y[i] >= h_max * 0.95:
        lift_end_idx = i
        break
    if y[i] < y[i-1]:
        lift_end_idx = i - 1
        break

print(f"\nLift hill detection:")
print(f"  Lift ends at index {lift_end_idx} (t={time[lift_end_idx]:.1f}s)")
print(f"  Height at lift end: {y[lift_end_idx]:.1f}m")
print(f"  Speed during lift: {v_kmh[:lift_end_idx].mean():.1f} km/h (should be ~11 km/h)")
print(f"  Speed after lift: {v_kmh[lift_end_idx+1:lift_end_idx+10].mean():.1f} km/h")

# Check for constant speed issue
speed_changes = np.abs(np.diff(v_kmh))
constant_sections = speed_changes < 0.1  # Less than 0.1 km/h change
if constant_sections.sum() > len(v_kmh) * 0.5:
    print(f"\n⚠️ WARNING: {constant_sections.sum()}/{len(v_kmh)} points have constant speed!")
    print(f"   Speed variation: min={v_kmh.min():.1f}, max={v_kmh.max():.1f}, std={v_kmh.std():.1f}")
