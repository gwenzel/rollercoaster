"""
Test time calculation for speed visualization
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.track import build_modular_track
from utils.acceleration import compute_acc_profile

def test_time_calculation():
    """Test if time calculation fixes the 'stuck' speed visualization"""
    track_elements = [
        {'type': 'lift_hill', 'params': {'length': 40, 'height': 80}},
        {'type': 'drop', 'params': {'length': 50, 'angle': 60}},
    ]
    
    track_df = build_modular_track(track_elements)
    x = track_df['x'].values
    y = track_df['y'].values
    z = track_df.get('z', pd.Series(np.zeros_like(x))).values
    points = np.column_stack([x, z, y])
    
    acc_result = compute_acc_profile(
        points,
        dt=0.02,
        v0=3.0,
        use_energy_conservation=True
    )
    
    velocity = acc_result['v']
    velocity_kmh = velocity * 3.6
    
    # Old method (fixed dt)
    time_old = np.linspace(0, len(velocity) * 0.02, len(velocity))
    
    # New method (actual travel time)
    ds = np.linalg.norm(np.diff(points, axis=0, prepend=points[0:1]), axis=1)
    v_avg = np.zeros_like(velocity)
    v_avg[0] = max(velocity[0], 0.1)
    for i in range(1, len(velocity)):
        v_avg[i] = 0.5 * (velocity[i-1] + velocity[i])
        v_avg[i] = max(v_avg[i], 0.1)
    dt_actual = ds / v_avg
    time_new = np.cumsum(dt_actual)
    
    print("="*80)
    print("TIME CALCULATION COMPARISON")
    print("="*80)
    print(f"\nOld method (fixed dt=0.02s):")
    print(f"   Time range: [0, {time_old[-1]:.2f}] s")
    print(f"   Total time: {time_old[-1]:.2f} s")
    print(f"   Speed at t=1s: {velocity_kmh[int(1.0/0.02)]:.2f} km/h")
    
    print(f"\nNew method (actual travel time):")
    print(f"   Time range: [0, {time_new[-1]:.2f}] s")
    print(f"   Total time: {time_new[-1]:.2f} s")
    idx_1s = np.argmin(np.abs(time_new - 1.0))
    print(f"   Speed at t=1s: {velocity_kmh[idx_1s]:.2f} km/h")
    
    print(f"\nComparison:")
    print(f"   Time difference: {time_new[-1] - time_old[-1]:.2f} s")
    print(f"   Speed variation (old time): {velocity_kmh.max() - velocity_kmh.min():.2f} km/h")
    print(f"   Speed variation (new time): {velocity_kmh.max() - velocity_kmh.min():.2f} km/h")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    axes[0].plot(time_old, velocity_kmh, 'b-', linewidth=2, label='Speed (old time axis)')
    axes[0].set_xlabel('Time (s) - Fixed dt=0.02s')
    axes[0].set_ylabel('Speed (km/h)')
    axes[0].set_title('Speed Profile with Fixed Time Step (OLD - May Look Stuck)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(time_new, velocity_kmh, 'r-', linewidth=2, label='Speed (actual time)')
    axes[1].set_xlabel('Time (s) - Actual Travel Time')
    axes[1].set_ylabel('Speed (km/h)')
    axes[1].set_title('Speed Profile with Actual Travel Time (NEW - Correct)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    output_path = Path(__file__).parent.parent / 'time_calculation_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Plot saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    test_time_calculation()

