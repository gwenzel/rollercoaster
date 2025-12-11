"""
Check what radii the physics engine is actually detecting
"""
import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from utils.track_blocks import banked_turn_profile
from utils.acceleration import _curvature_radius_vectorized

print("Checking Detected Curvature Radii\n")
print("="*70)

# Generate banked turn with 30m radius
x, y, z = banked_turn_profile(radius=30, angle=90)
points = np.column_stack([x, z, y])

print(f"Banked Turn: radius=30m, angle=90°")
print(f"Generated {len(points)} points")
print(f"Arc length: {np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)):.1f}m")
print(f"Expected arc: {2*np.pi*30*90/360:.1f}m")

# Compute radii
R = _curvature_radius_vectorized(points)

# Filter out infinities
finite_R = R[np.isfinite(R)]

print(f"\nDetected radii:")
print(f"  Mean: {np.mean(finite_R):.1f}m (expected: ~30m)")
print(f"  Median: {np.median(finite_R):.1f}m")
print(f"  Min: {np.min(finite_R):.1f}m")
print(f"  Max: {np.max(finite_R):.1f}m")
print(f"  Std: {np.std(finite_R):.1f}m")

# Show distribution
print(f"\nRadius distribution:")
bins = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 100), (100, float('inf'))]
for low, high in bins:
    count = np.sum((finite_R >= low) & (finite_R < high))
    pct = 100 * count / len(finite_R)
    print(f"  {low:3d}-{high:3.0f}m: {count:3d} points ({pct:5.1f}%)")

# Check for clamped values
clamped_min = np.sum(finite_R == 1.0)
clamped_max = np.sum(finite_R == 1000.0)
print(f"\nClamped values:")
print(f"  At 1m (min): {clamped_min}")
print(f"  At 1000m (max): {clamped_max}")

print("\n" + "="*70)
