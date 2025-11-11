"""
Test the full pipeline: track -> accelerometer -> BiGRU prediction
"""

from utils.track import build_modular_track
from utils.accelerometer_transform import track_to_accelerometer_data
from utils.bigru_predictor import predict_score_bigru

print("=" * 70)
print("TESTING FULL BIGRU PREDICTION PIPELINE")
print("=" * 70)

# Build a test track
track_elements = [
    {'type': 'climb', 'params': {'length': 30, 'height': 50}},
    {'type': 'drop', 'params': {'length': 60, 'angle': 70}},
    {'type': 'loop', 'params': {'radius': 10}},
    {'type': 'hills', 'params': {'num_hills': 3, 'amplitude': 8, 'wavelength': 30}},
]

print("\n[1] Building track...")
track_df = build_modular_track(track_elements)
print(f"✓ Track has {len(track_df)} points")

print("\n[2] Converting to accelerometer data...")
accel_df = track_to_accelerometer_data(track_df)
print(f"✓ Accelerometer data shape: {accel_df.shape}")
print(f"   Lateral range: [{accel_df['Lateral'].min():.2f}, {accel_df['Lateral'].max():.2f}]g")
print(f"   Vertical range: [{accel_df['Vertical'].min():.2f}, {accel_df['Vertical'].max():.2f}]g")
print(f"   Longitudinal range: [{accel_df['Longitudinal'].min():.2f}, {accel_df['Longitudinal'].max():.2f}]g")

print("\n[3] Predicting score with BiGRU...")
predicted_score = predict_score_bigru(track_df)

print("\n" + "=" * 70)
print(f"✓ PREDICTED SCORE: {predicted_score:.2f} / 5.0")
print("=" * 70)
print("\nPipeline test complete!")
