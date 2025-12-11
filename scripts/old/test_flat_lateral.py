"""Test why flat designs have lateral forces."""
import numpy as np
import pandas as pd
from utils.track_blocks import lift_hill_profile, flat_section_profile, vertical_drop_profile
from utils.accelerometer_transform import compute_rider_accelerations

# Create a simple flat track: lift → flat → drop → flat
print("Generating simple flat track...")

# Block 1: Lift hill
x1, y1, z1 = lift_hill_profile(length=50, height=30)
print(f"Lift hill: z range [{z1.min():.9f}, {z1.max():.9f}]")

# Block 2: Flat section
x2, y2, z2 = flat_section_profile(length=30)
x2 = x2 + x1[-1]
y2 = y2 + y1[-1]
z2 = z2 + z1[-1]
print(f"Flat 1: z range [{z2.min():.9f}, {z2.max():.9f}]")

# Block 3: Drop
x3, y3, z3 = vertical_drop_profile(height=25, steepness=0.85)
x3 = x3 + x2[-1]
y3 = y3 + y2[-1]
z3 = z3 + z2[-1]
print(f"Drop: z range [{z3.min():.9f}, {z3.max():.9f}]")

# Block 4: Final flat
x4, y4, z4 = flat_section_profile(length=40)
x4 = x4 + x3[-1]
y4 = y4 + y3[-1]
z4 = z4 + z3[-1]
print(f"Flat 2: z range [{z4.min():.9f}, {z4.max():.9f}]")

# Concatenate (no blending - just raw blocks)
x = np.concatenate([x1, x2[1:], x3[1:], x4[1:]])
y = np.concatenate([y1, y2[1:], y3[1:], y4[1:]])
z = np.concatenate([z1, z2[1:], z3[1:], z4[1:]])

print(f"\nFull track z range: [{z.min():.9f}, {z.max():.9f}]")
print(f"z all zero: {np.allclose(z, 0)}")
print(f"Max |z|: {np.abs(z).max():.9f}")

# Compute accelerations
print("\nComputing accelerations...")
track_df = pd.DataFrame({'x': x, 'y': y, 'z': z})
accel_df = compute_rider_accelerations(track_df)

# Check lateral forces
lateral_g = accel_df['Lateral'].values
print(f"\nLateral G-forces:")
print(f"  Range: [{lateral_g.min():.3f}, {lateral_g.max():.3f}] g")
print(f"  Mean: {lateral_g.mean():.6f} g")
print(f"  Std: {lateral_g.std():.6f} g")
print(f"  Max |lateral|: {np.abs(lateral_g).max():.6f} g")

# Find where lateral forces are largest
max_lat_idx = np.argmax(np.abs(lateral_g))
print(f"\nLargest lateral force at index {max_lat_idx}:")
print(f"  Position: x={x[max_lat_idx]:.1f}m, y={y[max_lat_idx]:.1f}m, z={z[max_lat_idx]:.9f}m")
print(f"  Lateral G: {lateral_g[max_lat_idx]:.6f} g")

# Also check vertical and longitudinal for reference
print(f"\nFor comparison:")
print(f"  Vertical G range: [{accel_df['Vertical'].min():.3f}, {accel_df['Vertical'].max():.3f}] g")
print(f"  Longitudinal G range: [{accel_df['Longitudinal'].min():.3f}, {accel_df['Longitudinal'].max():.3f}] g")

# Check z-coordinate changes around that point
if max_lat_idx > 5 and max_lat_idx < len(z) - 5:
    window = slice(max_lat_idx - 5, max_lat_idx + 6)
    print(f"  z values around this point: {z[window]}")
    print(f"  dz/dx: {np.diff(z[window]) / np.diff(x[window])}")
