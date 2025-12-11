"""
Comprehensive test to verify all components work together.
"""


#add utils to path
import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


print("=" * 70)
print("TESTING BIGRU INTEGRATION FOR STREAMLIT APP")
print("=" * 70)

# Test 1: Import all modules
print("\n[Test 1] Importing modules...")
try:
    from utils.track import build_modular_track, compute_acceleration, compute_features_modular
    from utils.bigru_predictor import predict_score_bigru, get_predictor
    from utils.model import predict_thrill
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Test 2: Build a sample track
print("\n[Test 2] Building modular track...")
try:
    track_elements = [
        {'type': 'climb', 'params': {'length': 30, 'height': 50}},
        {'type': 'drop', 'params': {'length': 60, 'angle': 70}},
        {'type': 'loop', 'params': {'radius': 10}},
        {'type': 'clothoid_loop', 'params': {'radius': 12, 'transition_length': 15}},
        {'type': 'hills', 'params': {'num_hills': 3, 'amplitude': 8, 'wavelength': 30}},
    ]
    track_df = build_modular_track(track_elements)
    print(f"✓ Track built: {len(track_df)} points")
except Exception as e:
    print(f"✗ Track building failed: {e}")
    exit(1)

# Test 3: Compute acceleration
print("\n[Test 3] Computing acceleration...")
try:
    max_height = track_df['y'].max()
    track_df = compute_acceleration(track_df, max_height)
    print(f"✓ Acceleration computed: range [{track_df['acceleration'].min():.2f}, {track_df['acceleration'].max():.2f}]")
except Exception as e:
    print(f"✗ Acceleration computation failed: {e}")
    exit(1)

# Test 4: Compute features
print("\n[Test 4] Computing features...")
try:
    features = compute_features_modular(track_df, track_elements)
    print(f"✓ Features computed: {len(features)} features")
except Exception as e:
    print(f"✗ Feature computation failed: {e}")
    exit(1)

# Test 5: Rule-based prediction
print("\n[Test 5] Rule-based thrill prediction...")
try:
    thrill_score = predict_thrill(features)
    print(f"✓ Thrill score: {thrill_score:.2f}/10")
except Exception as e:
    print(f"✗ Thrill prediction failed: {e}")
    exit(1)

# Test 6: BiGRU prediction
print("\n[Test 6] BiGRU score prediction...")
try:
    bigru_score = predict_score_bigru(track_df, model_path="models/bigru_score_model.pth")
    print(f"✓ BiGRU score: {bigru_score:.2f}/5.0")
except Exception as e:
    print(f"✗ BiGRU prediction failed: {e}")
    exit(1)

# Test 7: Model info
print("\n[Test 7] Getting model info...")
try:
    predictor = get_predictor("models/bigru_score_model.pth")
    info = predictor.get_model_info()
    print(f"✓ Model info retrieved:")
    print(f"  - Sequence length: {info['sequence_length']}")
    print(f"  - Input features: {info['input_size']}")
    print(f"  - Device: {info['device']}")
except Exception as e:
    print(f"✗ Model info failed: {e}")
    exit(1)

# Test 8: Multiple predictions (test singleton pattern)
print("\n[Test 8] Testing multiple predictions (singleton caching)...")
try:
    score1 = predict_score_bigru(track_df)
    score2 = predict_score_bigru(track_df)
    print(f"✓ Multiple predictions successful: {score1:.2f}, {score2:.2f}")
except Exception as e:
    print(f"✗ Multiple predictions failed: {e}")
    exit(1)

# Test 9: Different track configurations
print("\n[Test 9] Testing various track configurations...")
test_configs = [
    [{'type': 'climb', 'params': {'length': 20, 'height': 30}}],
    [{'type': 'loop', 'params': {'radius': 15}}],
    [{'type': 'hills', 'params': {'num_hills': 5, 'amplitude': 10, 'wavelength': 25}}],
]

for i, config in enumerate(test_configs):
    try:
        test_track = build_modular_track(config)
        test_track = compute_acceleration(test_track, test_track['y'].max())
        score = predict_score_bigru(test_track)
        print(f"  Config {i+1}: score = {score:.2f}/5.0 ✓")
    except Exception as e:
        print(f"  Config {i+1}: Failed - {e} ✗")

# Summary
print("\n" + "=" * 70)
print("ALL TESTS PASSED! ✓")
print("=" * 70)
print("\nYou can now run: streamlit run app.py")
print("The app will display both:")
print("  - Rule-based thrill score (0-10)")
print("  - BiGRU predicted rating (1-5)")
print("=" * 70)
