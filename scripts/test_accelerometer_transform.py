"""
Test the accelerometer transformation pipeline.
"""


#add utils to path
import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from utils.accelerometer_transform import track_to_accelerometer_data
from utils.track import build_modular_track

print("=" * 70)
print("TESTING ACCELEROMETER TRANSFORMATION")
print("=" * 70)

# Build a test track
track_elements = [
    {'type': 'climb', 'params': {'length': 30, 'height': 50}},
    {'type': 'drop', 'params': {'length': 60, 'angle': 70}},
    {'type': 'loop', 'params': {'radius': 10}},
]

print("\n[1] Building track...")
track = build_modular_track(track_elements)
print(f"Track has {len(track)} points")
print(f"Track columns: {track.columns.tolist()}")

print("\n[2] Converting to accelerometer data...")
accel = track_to_accelerometer_data(track)

print(f"\nAccelerometer data shape: {accel.shape}")
print(f"Columns: {accel.columns.tolist()}")

print("\n[3] Sample data (first 10 rows):")
print(accel.head(10))

print("\n[4] Acceleration ranges (in g-forces):")
print(f"  Lateral:      [{accel['Lateral'].min():.2f}, {accel['Lateral'].max():.2f}]")
print(f"  Vertical:     [{accel['Vertical'].min():.2f}, {accel['Vertical'].max():.2f}]")
print(f"  Longitudinal: [{accel['Longitudinal'].min():.2f}, {accel['Longitudinal'].max():.2f}]")

print("\n[5] Statistics:")
print(accel.describe())

print("\n" + "=" * 70)
print("✓ Accelerometer transformation successful!")
print("=" * 70)
