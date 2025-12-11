"""
LightGBM Extreme model integration for Streamlit app.

Replicates the engineered 26-feature pipeline from the
`Dec6_Feature_Only_Cleaned (1).ipynb` notebook and loads the trained
extreme (thrill) model saved as `models/lightgbm/lgb_extreme_model.txt`.

Features come directly from rider-frame, gravity-subtracted acceleration
data (Vertical, Lateral, Longitudinal) plus optional metadata.
"""

from functools import lru_cache
import os
from typing import Dict, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

# Ordered feature names, matching `extreme_model_config.pkl`
FEATURE_NAMES = [
    # Dynamics (20)
    "Num Positive G (>3.0g)",
    "Max Negative Vertical G",
    "Max Positive Vertical G",
    "Max Lateral G",
    "Max Longitudinal G",
    "Vertical Variance",
    "Lateral Variance",
    "Vertical Jerk",
    "Avg Total G",
    "Airtime×G-Force Interaction",
    "G-Force Range",
    "Lateral Jerk",
    "G-Force Skewness",
    "Intensity Pacing",
    "Force Transitions",
    "Peak Density",
    "Rhythm Score",
    "Lateral Vibration",
    "Vertical Vibration",
    "Longitudinal Vibration",
    # Airtime (3)
    "Total Length (log-sec)",
    "Floater Airtime %",
    "Flojector Airtime %",
    # Metadata (3)
    "Height (m)",
    "Speed (km/h)",
    "Track Length (m)",
]

# Fallback metadata (approx typical mid-intensity coaster values)
DEFAULT_METADATA = {
    "height_m": 30.0,
    "speed_kmh": 80.0,
    "track_length_m": 1200.0,
}


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean with minimal dependencies."""
    if window <= 1:
        return arr
    # Use pandas for convenience; center=True matches notebook behavior
    return (
        pd.Series(arr)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def _safe_dt(times: np.ndarray) -> float:
    """Infer sampling interval; default to 0.1s (10Hz) if unknown/invalid."""
    if times is None or len(times) < 2:
        return 0.1
    diffs = np.diff(times.astype(float))
    dt = np.nanmedian(diffs) if len(diffs) else 0.1
    if not np.isfinite(dt) or dt <= 0:
        return 0.1
    return float(dt)


def _prepare_arrays(accel_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Extract raw arrays (lateral, vertical, longitudinal) and dt."""
    vertical = np.nan_to_num(accel_df["Vertical"].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    lateral = np.nan_to_num(accel_df["Lateral"].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    longitudinal = np.nan_to_num(accel_df["Longitudinal"].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    times = accel_df["Time"].to_numpy(dtype=float) if "Time" in accel_df.columns else None
    dt = _safe_dt(times)
    return lateral, vertical, longitudinal, dt


def compute_lightgbm_features(accel_df: pd.DataFrame, metadata: Dict[str, float] = None, return_dict: bool = False):
    """
    Compute the 26-feature vector used by the LightGBM extreme model.

    Args:
        accel_df: DataFrame with columns ['Vertical', 'Lateral', 'Longitudinal'] in rider frame (g units), optional 'Time'.
        metadata: Optional dict with keys height_m, speed_kmh, track_length_m.
        return_dict: If True, also return a {feature_name: value} mapping.

    Returns:
        features: np.ndarray shape (26,)
        features_dict (optional): mapping feature -> value
    """
    if accel_df is None or len(accel_df) == 0:
        features = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        return (features, dict(zip(FEATURE_NAMES, features))) if return_dict else features

    lateral, vertical, longitudinal, dt = _prepare_arrays(accel_df)
    total_g = np.sqrt(vertical ** 2 + lateral ** 2 + longitudinal ** 2)

    # Rolling window size: 1 second equivalent (approx) -> round(1/dt)
    window = max(1, int(round(1.0 / dt)))
    smoothed_lateral = _rolling_mean(lateral, window)
    smoothed_vertical = _rolling_mean(vertical, window)
    smoothed_longitudinal = _rolling_mean(longitudinal, window)

    # === Original 9 ===
    num_positive_g = float(np.sum(vertical > 3.0))
    max_neg_vert = float(np.min(vertical)) if len(vertical) else 0.0
    max_pos_vert = float(np.max(vertical)) if len(vertical) else 0.0
    max_lateral = float(np.max(np.abs(lateral))) if len(lateral) else 0.0
    max_longitudinal = float(np.max(np.abs(longitudinal))) if len(longitudinal) else 0.0
    vert_variance = float(np.var(vertical)) if len(vertical) else 0.0
    lat_variance = float(np.var(lateral)) if len(lateral) else 0.0
    vert_jerk = float(np.mean(np.abs(np.diff(vertical)))) if len(vertical) > 1 else 0.0
    avg_total_g = float(np.mean(total_g)) if len(total_g) else 0.0

    # === Advanced 8 ===
    airtime_ratio = float(np.sum(vertical < 0) / len(vertical)) if len(vertical) else 0.0
    positive_g_ratio = float(np.sum(vertical > 2.0) / len(vertical)) if len(vertical) else 0.0
    airtime_gforce_interaction = airtime_ratio * positive_g_ratio * 10.0

    g_force_range = (max_pos_vert - max_neg_vert) if (max_pos_vert > 0 or max_neg_vert < 0) else 0.0
    lateral_jerk = float(np.mean(np.abs(np.diff(lateral)))) if len(lateral) > 1 else 0.0
    try:
        g_skewness = float(pd.Series(vertical).skew()) if len(vertical) > 3 else 0.0
    except Exception:
        g_skewness = 0.0

    mid_point = len(total_g) // 2
    if mid_point > 0:
        first_half_intensity = float(np.mean(total_g[:mid_point]))
        second_half_intensity = float(np.mean(total_g[mid_point:]))
        intensity_pacing = first_half_intensity / (second_half_intensity + 0.1)
    else:
        intensity_pacing = 1.0

    g_transitions = float(np.std(np.diff(total_g))) if len(total_g) > 1 else 0.0

    high_g_threshold = float(np.percentile(total_g, 90)) if len(total_g) > 10 else 3.0
    high_g_moments = total_g > high_g_threshold
    peak_density = float(np.sum(high_g_moments) / (len(total_g) / 100.0)) if np.sum(high_g_moments) > 0 else 0.0

    if len(total_g) > 20:
        rhythm_score = float(np.corrcoef(total_g[:-10], total_g[10:])[0, 1])
        if not np.isfinite(rhythm_score) or rhythm_score < 0:
            rhythm_score = 0.0
    else:
        rhythm_score = 0.0

    # === Vibration (3) ===
    lateral_vibration = float(np.sqrt(np.mean((lateral - smoothed_lateral) ** 2)))
    vertical_vibration = float(np.sqrt(np.mean((vertical - smoothed_vertical) ** 2)))
    longitudinal_vibration = float(np.sqrt(np.mean((longitudinal - smoothed_longitudinal) ** 2)))

    # === Airtime (3) ===
    total_samples = len(vertical)
    total_seconds = total_samples * dt
    total_length_seconds = float(np.log1p(total_seconds))  # log(1 + seconds)
    floater_airtime_mask = (vertical < 0.0) & (vertical >= -0.5)
    flojector_airtime_mask = (vertical < -0.5) & (vertical >= -1.0)
    floater_airtime_proportion = float(np.sum(floater_airtime_mask) / total_samples) if total_samples > 0 else 0.0
    flojector_airtime_proportion = float(np.sum(flojector_airtime_mask) / total_samples) if total_samples > 0 else 0.0

    # === Metadata (3) ===
    meta = DEFAULT_METADATA.copy()
    if metadata:
        meta.update({k: v for k, v in metadata.items() if v is not None})
    height_m = float(meta.get("height_m", DEFAULT_METADATA["height_m"]))
    speed_kmh = float(meta.get("speed_kmh", DEFAULT_METADATA["speed_kmh"]))
    track_length_m = float(meta.get("track_length_m", DEFAULT_METADATA["track_length_m"]))

    features = np.array(
        [
            num_positive_g,
            max_neg_vert,
            max_pos_vert,
            max_lateral,
            max_longitudinal,
            vert_variance,
            lat_variance,
            vert_jerk,
            avg_total_g,
            airtime_gforce_interaction,
            g_force_range,
            lateral_jerk,
            g_skewness,
            intensity_pacing,
            g_transitions,
            peak_density,
            rhythm_score,
            lateral_vibration,
            vertical_vibration,
            longitudinal_vibration,
            total_length_seconds,
            floater_airtime_proportion,
            flojector_airtime_proportion,
            height_m,
            speed_kmh,
            track_length_m,
        ],
        dtype=np.float32,
    )

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    if return_dict:
        return features, dict(zip(FEATURE_NAMES, features))
    return features


@lru_cache(maxsize=1)
def _load_booster(model_path: str) -> lgb.Booster:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LightGBM model not found at {model_path}")
    return lgb.Booster(model_file=model_path)


def predict_score_lgb(accel_df: pd.DataFrame, metadata: Dict[str, float] = None, model_path: str = "models/lightgbm/lgb_extreme_model.txt") -> float:
    """
    Predict fun rating using the LightGBM extreme model.

    Args:
        accel_df: DataFrame with rider-frame acceleration (Vertical, Lateral, Longitudinal) and optional Time.
        metadata: Optional dict with keys height_m, speed_kmh, track_length_m.
        model_path: Path to the LightGBM model file.
    """
    features = compute_lightgbm_features(accel_df, metadata=metadata, return_dict=False)
    booster = _load_booster(model_path)
    raw_pred = float(booster.predict(features.reshape(1, -1))[0])
    return float(np.clip(raw_pred, 1.0, 5.0))


