"""
Analyze vertical G-forces in detail to see if they're realistic
"""
import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from utils.track_blocks import lift_hill_profile, vertical_drop_profile, loop_profile
from utils.accelerometer_transform import track_to_accelerometer_data

print("Analyzing Vertical G-Forces vs Real Coasters\n")
print("="*70)

# Test different loop sizes
for diameter in [15, 20, 25, 30, 40]:
    print(f"\nLoop diameter: {diameter}m (radius: {diameter/2}m)")
    print("-"*70)
    
    # Build track: lift → drop → loop
    all_x, all_y, all_z = [], [], []
    offset_x, offset_y = 0.0, 0.0
    
    # Lift to appropriate height for this loop
    # Need enough height to complete the loop: h >= loop_diameter + safety_margin
    lift_height = diameter * 1.5  # 1.5x loop height for good speed
    
    x, y, z = lift_hill_profile(height=lift_height, length=lift_height*1.2)
    all_x.extend((x + offset_x).tolist())
    all_y.extend((y + offset_y).tolist())
    all_z.extend(z.tolist())
    offset_x, offset_y = all_x[-1], all_y[-1]
    
    x, y, z = vertical_drop_profile(height=lift_height*0.9, steepness=0.8)
    all_x.extend((x + offset_x).tolist())
    all_y.extend((y + offset_y).tolist())
    all_z.extend(z.tolist())
    offset_x, offset_y = all_x[-1], all_y[-1]
    
    x, y, z = loop_profile(diameter=diameter)
    all_x.extend((x + offset_x).tolist())
    all_y.extend((y + offset_y).tolist())
    all_z.extend(z.tolist())
    
    track_df = pd.DataFrame({'x': all_x, 'y': all_y, 'z': all_z})
    accel_df = track_to_accelerometer_data(track_df)
    
    # Calculate expected G-force at bottom of loop
    # v = sqrt(2*g*h) where h is the drop height
    v_bottom = np.sqrt(2 * 9.81 * lift_height * 0.9)
    radius = diameter / 2
    centripetal_g = (v_bottom**2 / radius) / 9.81
    total_g_expected = centripetal_g + 1.0  # Add 1g from gravity
    
    print(f"  Lift height: {lift_height:.1f}m")
    print(f"  Speed at bottom: {v_bottom:.1f} m/s = {v_bottom*3.6:.0f} km/h")
    print(f"  Expected centripetal: {centripetal_g:.1f}g")
    print(f"  Expected total (bottom): {total_g_expected:.1f}g")
    print(f"  Actual max vertical: {accel_df['Vertical'].max():.1f}g")
    print(f"  Actual min vertical: {accel_df['Vertical'].min():.1f}g")
    
    # Check if too high
    if accel_df['Vertical'].max() > 7:
        print(f"  ⚠️  TOO HIGH! Real coasters max ~6-7g")
    elif accel_df['Vertical'].max() > 5:
        print(f"  ⚠️  High (intense)")
    else:
        print(f"  ✓ Reasonable")

print(f"\n{'='*70}")
print("REAL COASTER REFERENCE DATA:")
print("-"*70)
print("  Intimidator 305:    +5.9g (one of highest, caused blackouts)")
print("  Formula Rossa:       +4.8g (fastest coaster)")
print("  Millennium Force:    +4.5g")
print("  Schwarzkopf loops:   +6.0g (rare, vintage)")
print("  Most modern coasters: +3.5-4.5g")
print("  Top of loop:         -1.0 to +1.5g")
print(f"{'='*70}")

# Test if clipping or physics is the issue
print("\n" + "="*70)
print("Checking if ±10g clipping is being hit:")
print("-"*70)

all_x, all_y, all_z = [], [], []
offset_x, offset_y = 0.0, 0.0

x, y, z = lift_hill_profile(height=50, length=60)
all_x.extend((x + offset_x).tolist())
all_y.extend((y + offset_y).tolist())
all_z.extend(z.tolist())
offset_x, offset_y = all_x[-1], all_y[-1]

x, y, z = vertical_drop_profile(height=45, steepness=0.85)
all_x.extend((x + offset_x).tolist())
all_y.extend((y + offset_y).tolist())
all_z.extend(z.tolist())
offset_x, offset_y = all_x[-1], all_y[-1]

x, y, z = loop_profile(diameter=20)  # Tight loop
all_x.extend((x + offset_x).tolist())
all_y.extend((y + offset_y).tolist())
all_z.extend(z.tolist())

track_df = pd.DataFrame({'x': all_x, 'y': all_y, 'z': all_z})
accel_df = track_to_accelerometer_data(track_df)

clipped_high = np.sum(accel_df['Vertical'] >= 10.0)
clipped_low = np.sum(accel_df['Vertical'] <= -10.0)
max_val = accel_df['Vertical'].max()
min_val = accel_df['Vertical'].min()

print(f"Extreme scenario: 50m lift → 45m drop → 20m loop")
print(f"  Max vertical: {max_val:.1f}g")
print(f"  Min vertical: {min_val:.1f}g")
print(f"  Points at +10g limit: {clipped_high}")
print(f"  Points at -10g limit: {clipped_low}")

if clipped_high > 0 or clipped_low > 0:
    print(f"  ⚠️  CLIPPING DETECTED - Physics engine generating unrealistic values")
else:
    print(f"  ✓ No clipping - Values are from physics calculation")

print(f"{'='*70}")
