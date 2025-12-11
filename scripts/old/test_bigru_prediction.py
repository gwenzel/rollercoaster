"""
Test script to verify BiGRU prediction works with track data.
"""

#add utils to path
import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from utils.track import build_modular_track, compute_acceleration
from utils.bigru_predictor import predict_score_bigru

# Create a sample track
track_elements = [
    {'type': 'climb', 'params': {'length': 30, 'height': 50}},
    {'type': 'drop', 'params': {'length': 60, 'angle': 70}},
    {'type': 'loop', 'params': {'radius': 10}},
    {'type': 'hills', 'params': {'num_hills': 3, 'amplitude': 8, 'wavelength': 30}},
]

print("Building track...")
track_df = build_modular_track(track_elements)

print(f"Track has {len(track_df)} points")
print(f"Track columns: {track_df.columns.tolist()}")

print("\nComputing acceleration...")
max_height = track_df['y'].max()
track_df = compute_acceleration(track_df, max_height)

print(f"Acceleration range: [{track_df['acceleration'].min():.2f}, {track_df['acceleration'].max():.2f}]")

print("\nPredicting score with BiGRU...")
predicted_score = predict_score_bigru(track_df)

print("\n" + "=" * 70)
print(f"✓ Predicted Score: {predicted_score:.2f} / 5.0")
print("=" * 70)
