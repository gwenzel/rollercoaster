"""Test speed calculations."""
import sys
sys.path.insert(0, '.')

import numpy as np
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

print(f"Speed Statistics:")
print(f"  Max speed: {v_kmh.max():.1f} km/h ({result['v'].max():.1f} m/s)")
print(f"  Avg speed: {v_kmh.mean():.1f} km/h ({result['v'].mean():.1f} m/s)")
print(f"  Min speed: {v_kmh.min():.1f} km/h ({result['v'].min():.1f} m/s)")
print(f"\nTrack info:")
print(f"  Max height: {y.max():.1f}m")
print(f"  Min height: {y.min():.1f}m")
print(f"  Height drop: {y.max() - y.min():.1f}m")
print(f"\nTheoretical max speed from {y.max():.1f}m drop:")
print(f"  100% efficiency: {np.sqrt(2*9.81*y.max())*3.6:.1f} km/h")
print(f"  80% efficiency: {np.sqrt(2*9.81*y.max())*3.6*0.8:.1f} km/h")
print(f"\nReal coaster comparison:")
print(f"  Millennium Force: 150 km/h (94 mph) from 94m")
print(f"  Fury 325: 153 km/h (95 mph) from 99m")
print(f"  Steel Vengeance: 119 km/h (74 mph) from 62m")
