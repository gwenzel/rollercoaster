"""
Test if track blocks are generating proper 3D (z) coordinates
"""
import numpy as np
import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.track_blocks import (
    lift_hill_profile, vertical_drop_profile, loop_profile,
    airtime_hill_profile, spiral_profile, banked_turn_profile,
    bunny_hop_profile, flat_section_profile
)

print("Testing 3D coordinate generation for all track blocks\n")
print("="*70)

# Test each block
blocks_to_test = [
    ("Lift Hill", lift_hill_profile, {'length': 50, 'height': 40}),
    ("Vertical Drop", vertical_drop_profile, {'height': 40, 'steepness': 0.9}),
    ("Loop", loop_profile, {'diameter': 30}),
    ("Airtime Hill", airtime_hill_profile, {'length': 40, 'height': 15}),
    ("Spiral", spiral_profile, {'diameter': 25, 'turns': 1.5}),
    ("Banked Turn", banked_turn_profile, {'radius': 30, 'angle': 90}),
    ("Bunny Hop", bunny_hop_profile, {'length': 20, 'height': 8}),
    ("Flat Section", flat_section_profile, {'length': 30}),
]

for name, func, params in blocks_to_test:
    print(f"\n{name}:")
    try:
        x, y, z = func(**params)
        print(f"  Points: {len(x)}")
        print(f"  X range: [{x.min():.2f}, {x.max():.2f}]")
        print(f"  Y range: [{y.min():.2f}, {y.max():.2f}]")
        print(f"  Z range: [{z.min():.2f}, {z.max():.2f}]")
        
        if np.all(z == 0):
            print(f"  Status: No banking (all zeros)")
        else:
            print(f"  Status: HAS BANKING! (non-zero z values)")
            
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n" + "="*70)
print("\nBlocks WITH banking should be: Spiral, Banked Turn")
print("All others should have z=0")
