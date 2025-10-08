# utils/track.py
import numpy as np
import pandas as pd


def generate_track(height_peak, drop_angle, loop_radius, num_hills, hill_amplitude):
    height_start = 0
    drop_length = 60
    hill_wavelength = 30
    final_length = 40


    # --- SECTION 1: Climb ---
    x1 = np.linspace(0, 30, 100)
    y1 = np.linspace(height_start, height_peak, 100)


    # --- SECTION 2: Drop ---
    x2 = np.linspace(x1[-1], x1[-1] + drop_length, 100)
    y2 = height_peak - (x2 - x1[-1]) * np.tan(np.radians(drop_angle))


    # --- SECTION 3: Loop ---
    loop_length = 2 * np.pi * loop_radius
    x3 = np.linspace(x2[-1], x2[-1] + loop_length, 200)
    theta = np.linspace(0, 2*np.pi, 200)
    loop_center_y = y2[-1] - loop_radius
    y3 = loop_center_y + loop_radius * np.cos(theta)


    # --- SECTION 4: Hills ---
    x4 = np.linspace(x3[-1], x3[-1] + num_hills * hill_wavelength, 300)
    y4 = y3[-1] + hill_amplitude * np.sin(2 * np.pi * (x4 - x3[-1]) / hill_wavelength)


    # --- SECTION 5: Final Flat ---
    x5 = np.linspace(x4[-1], x4[-1] + final_length, 50)
    y5 = np.ones_like(x5) * y4[-1]


    x = np.concatenate([x1, x2, x3, x4, x5])
    y = np.concatenate([y1, y2, y3, y4, y5])


    return pd.DataFrame({"x": x, "y": y})

def compute_features(track_df, loop_radius, num_hills):
    x, y = track_df["x"].values, track_df["y"].values
    slope = np.gradient(y, x)
    curvature = np.gradient(slope, x)
    total_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))


    return {
    "total_length": float(total_length),
    "max_height": float(np.max(y)),
    "max_slope": float(np.max(np.abs(slope))),
    "mean_curvature": float(np.mean(np.abs(curvature))),
    "loop_radius": loop_radius,
    "num_hills": num_hills,
    }


def compute_acceleration(track_df, height_peak, g=9.81):
    # Approximate particle acceleration assuming conservation of energy from the top
    y = track_df["y"].values
    v = np.sqrt(2 * g * (height_peak - y).clip(min=0)) # velocity based on drop height
    x = track_df["x"].values
    a_tangent = np.gradient(v, x, edge_order=2)
    track_df["velocity"] = v
    track_df["acceleration"] = a_tangent
    return track_df