"""
Check what speeds are being computed in the multi-loop track
"""
import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from utils.track_blocks import lift_hill_profile, vertical_drop_profile, loop_profile
from utils.acceleration import compute_acc_profile

print("Checking Speed Profile in Multi-Loop Track\n")
print("="*70)

# Build track
all_x, all_y, all_z = [], [], []
offset_x, offset_y = 0.0, 0.0

x, y, z = lift_hill_profile(height=50, length=60)
all_x.extend((x + offset_x).tolist())
all_y.extend((y + offset_y).tolist())
all_z.extend(z.tolist())
offset_x, offset_y = all_x[-1], all_y[-1]

x, y, z = vertical_drop_profile(height=45, steepness=0.8)
all_x.extend((x + offset_x).tolist())
all_y.extend((y + offset_y).tolist())
all_z.extend(z.tolist())
offset_x, offset_y = all_x[-1], all_y[-1]

x, y, z = loop_profile(diameter=20)
all_x.extend((x + offset_x).tolist())
all_y.extend((y + offset_y).tolist())
all_z.extend(z.tolist())

points = np.column_stack([all_x, all_z, all_y])

# Test with current parameters
print("Current parameters:")
print("  mu=0.02, Cd=0.6, A=4.0")
acc = compute_acc_profile(points, dt=0.02, mass=6000.0, rho=1.3, Cd=0.6, A=4.0, mu=0.02, v0=5.0)
print(f"  Speed: [{acc['v'].min():.1f}, {acc['v'].max():.1f}] m/s")
print(f"  Vertical G: [{acc['f_vert_g'].min():.2f}, {acc['f_vert_g'].max():.2f}] g")

# Physics check: at bottom of 20m loop after 45m drop
# v = sqrt(2*g*h) = sqrt(2*9.81*45) = 29.7 m/s
# Centripetal: v²/r = 29.7²/10 = 88 m/s² = 9g
# Plus gravity = 10g total expected
print(f"\nExpected at bottom of loop (no friction):")
print(f"  Speed: ~30 m/s from energy conservation")
print(f"  Vertical G: ~9g (centripetal) + 1g (gravity) = 10g")

# Test with lower friction
print("\n" + "-"*70)
print("Reduced friction (mu=0.005, Cd=0.3):")
acc2 = compute_acc_profile(points, dt=0.02, mass=6000.0, rho=1.3, Cd=0.3, A=4.0, mu=0.005, v0=5.0)
print(f"  Speed: [{acc2['v'].min():.1f}, {acc2['v'].max():.1f}] m/s")
print(f"  Vertical G: [{acc2['f_vert_g'].min():.2f}, {acc2['f_vert_g'].max():.2f}] g")

# Test with minimal friction
print("\n" + "-"*70)
print("Minimal friction (mu=0.001, Cd=0.1):")
acc3 = compute_acc_profile(points, dt=0.02, mass=6000.0, rho=1.3, Cd=0.1, A=4.0, mu=0.001, v0=5.0)
print(f"  Speed: [{acc3['v'].min():.1f}, {acc3['v'].max():.1f}] m/s")
print(f"  Vertical G: [{acc3['f_vert_g'].min():.2f}, {acc3['f_vert_g'].max():.2f}] g")

print("\n" + "="*70)
