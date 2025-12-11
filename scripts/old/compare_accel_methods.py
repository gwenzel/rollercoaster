"""
Compare acceleration calculations: energy conservation vs simple method
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
from utils.acceleration import compute_acc_profile

print("Comparing Acceleration Methods on Smooth Track\n")
print("="*70)

# Build a smooth track
all_x, all_y, all_z = [], [], []
offset_x, offset_y = 0.0, 0.0

blocks = [
    ("Lift", lift_hill_profile(height=30, length=40)),
    ("Drop", vertical_drop_profile(height=25, steepness=0.8)),
    ("Flat", flat_section_profile(length=30)),
]

for name, (x, y, z) in blocks:
    all_x.extend((x + offset_x).tolist())
    all_y.extend((y + offset_y).tolist())
    all_z.extend(z.tolist())
    offset_x, offset_y = all_x[-1], all_y[-1]

points = np.column_stack([all_x, all_z, all_y])

print(f"Track: Lift → Drop → Flat")
print(f"Total points: {len(points)}")

# Method 1: Energy conservation (current)
print("\n" + "-"*70)
print("Method 1: Energy Conservation")
print("-"*70)
acc_energy = compute_acc_profile(points, dt=0.02, v0=3.0, use_energy_conservation=True)
print(f"Speed range: [{acc_energy['v'].min():.1f}, {acc_energy['v'].max():.1f}] m/s")
print(f"Vertical G: [{acc_energy['f_vert_g'].min():.2f}, {acc_energy['f_vert_g'].max():.2f}] g")
print(f"Lateral G: [{acc_energy['f_lat_g'].min():.2f}, {acc_energy['f_lat_g'].max():.2f}] g")
print(f"Longitudinal G: [{acc_energy['f_long_g'].min():.2f}, {acc_energy['f_long_g'].max():.2f}] g")

# Check for oscillations
vert_std = np.std(acc_energy['f_vert_g'])
vert_changes = np.sum(np.abs(np.diff(acc_energy['f_vert_g'])) > 0.5)
print(f"\nVertical G std dev: {vert_std:.3f}")
print(f"Large changes (>0.5g): {vert_changes}")

# Method 2: Physics integration (friction/drag)
print("\n" + "-"*70)
print("Method 2: Physics Integration (with friction/drag)")
print("-"*70)
acc_physics = compute_acc_profile(points, dt=0.02, v0=3.0, 
                                   mu=0.003, Cd=0.15, A=4.0, 
                                   use_energy_conservation=False)
print(f"Speed range: [{acc_physics['v'].min():.1f}, {acc_physics['v'].max():.1f}] m/s")
print(f"Vertical G: [{acc_physics['f_vert_g'].min():.2f}, {acc_physics['f_vert_g'].max():.2f}] g")
print(f"Lateral G: [{acc_physics['f_lat_g'].min():.2f}, {acc_physics['f_lat_g'].max():.2f}] g")
print(f"Longitudinal G: [{acc_physics['f_long_g'].min():.2f}, {acc_physics['f_long_g'].max():.2f}] g")

vert_std_phys = np.std(acc_physics['f_vert_g'])
vert_changes_phys = np.sum(np.abs(np.diff(acc_physics['f_vert_g'])) > 0.5)
print(f"\nVertical G std dev: {vert_std_phys:.3f}")
print(f"Large changes (>0.5g): {vert_changes_phys}")

# Visual comparison
print("\n" + "="*70)
print("Plotting comparison...")
fig, axes = plt.subplots(4, 1, figsize=(12, 10))

# Track profile
axes[0].plot(all_x, all_y, 'k-', linewidth=2)
axes[0].set_ylabel('Height (m)')
axes[0].set_title('Track Profile')
axes[0].grid(True, alpha=0.3)

# Speed comparison
axes[1].plot(acc_energy['v'], label='Energy Conservation', linewidth=2)
axes[1].plot(acc_physics['v'], label='Physics Integration', linewidth=2, alpha=0.7)
axes[1].set_ylabel('Speed (m/s)')
axes[1].set_title('Speed Profile')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Vertical G comparison
axes[2].plot(acc_energy['f_vert_g'], label='Energy Conservation', linewidth=2)
axes[2].plot(acc_physics['f_vert_g'], label='Physics Integration', linewidth=2, alpha=0.7)
axes[2].set_ylabel('Vertical G (g)')
axes[2].set_title('Vertical G-Forces')
axes[2].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# Longitudinal G comparison
axes[3].plot(acc_energy['f_long_g'], label='Energy Conservation', linewidth=2)
axes[3].plot(acc_physics['f_long_g'], label='Physics Integration', linewidth=2, alpha=0.7)
axes[3].set_ylabel('Longitudinal G (g)')
axes[3].set_xlabel('Sample Index')
axes[3].set_title('Longitudinal G-Forces')
axes[3].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('scripts/accel_method_comparison.png', dpi=150, bbox_inches='tight')
print(f"Saved comparison plot to: scripts/accel_method_comparison.png")

print("\n" + "="*70)
print("ANALYSIS:")
print("-"*70)
if vert_std > vert_std_phys * 1.5:
    print("⚠️  Energy conservation has MORE oscillations than physics integration")
    print("   This suggests the curvature calculation is noisy")
elif vert_std_phys > vert_std * 1.5:
    print("⚠️  Physics integration has MORE oscillations than energy conservation")
    print("   This suggests the speed integration is unstable")
else:
    print("✓ Both methods have similar smoothness")

print("\nPossible causes of oscillations:")
print("  1. Curvature calculation from discrete points is noisy")
print("  2. Tangent vector estimation amplifies discretization errors")
print("  3. Need smoothing/filtering on track geometry before physics")
print("="*70)
