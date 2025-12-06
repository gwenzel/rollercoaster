"""
Debug the banked turn physics to see why we get 10g lateral
"""
import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from utils.track_blocks import banked_turn_profile
from utils.acceleration import compute_acc_profile

print("Debugging Banked Turn Physics\n")
print("="*70)

# Generate banked turn
x, y, z = banked_turn_profile(radius=30, angle=90)
points = np.column_stack([x, z, y])

# Test with different initial speeds
for v0 in [10.0, 15.0, 20.0, 25.0]:
    print(f"\nInitial speed: {v0:.1f} m/s")
    print("-"*70)
    
    acc = compute_acc_profile(points, dt=0.02, v0=v0)
    
    print(f"Speed range: [{acc['v'].min():.1f}, {acc['v'].max():.1f}] m/s")
    print(f"Lateral G: [{acc['f_lat_g'].min():.2f}, {acc['f_lat_g'].max():.2f}] g")
    print(f"Vertical G: [{acc['f_vert_g'].min():.2f}, {acc['f_vert_g'].max():.2f}] g")
    
    # Calculate expected centripetal force for 30m radius turn
    avg_speed = np.mean(acc['v'])
    expected_cent_g = (avg_speed**2 / 30.0) / 9.81
    print(f"Expected centripetal for {avg_speed:.1f}m/s at R=30m: {expected_cent_g:.2f}g")

print("\n" + "="*70)
print("\nREFERENCE:")
print("  At 20 m/s in 30m radius turn: (20²/30)/9.81 = 1.36g")
print("  At 25 m/s in 30m radius turn: (25²/30)/9.81 = 2.12g")
print("  Banking helps riders feel less lateral force")
print("="*70)
