"""Test 3D velocity calculation"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from utils.acceleration import compute_acc_profile

# Simple test track
points = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [2, 0, -1],
    [3, 0, -2],
    [4, 0, -3]
])

print("Testing 3D velocity calculation...")
result = compute_acc_profile(
    points,
    v0=3.0,
    use_energy_conservation=False,
    use_velocity_verlet=True
)

print(f"\nSpeed (Euclidean magnitude): {result['v'][:3]}")
print(f"3D velocity shape: {result['v_3d'].shape}")
print(f"First 3D velocity vector: {result['v_3d'][0]}")
print(f"Second 3D velocity vector: {result['v_3d'][1]}")
print(f"\nVerification:")
print(f"  Speed[0] = ||v_3d[0]|| = {np.linalg.norm(result['v_3d'][0]):.6f}")
print(f"  Speed[1] = ||v_3d[1]|| = {np.linalg.norm(result['v_3d'][1]):.6f}")
print(f"  Match: {np.allclose(result['v'], np.linalg.norm(result['v_3d'], axis=1))}")

