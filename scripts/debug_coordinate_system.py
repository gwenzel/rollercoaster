"""
Debug coordinate system mappings and sign conventions
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.track import build_modular_track
from utils.accelerometer_transform import compute_rider_accelerations, compute_track_derivatives
from utils.acceleration import compute_acc_profile, G_VEC, G_NORM

def create_test_track():
    """Create a simple test track"""
    track_elements = [
        {'type': 'lift_hill', 'params': {'length': 40, 'height': 80}},
        {'type': 'drop', 'params': {'length': 50, 'angle': 60}},
        {'type': 'straight', 'params': {'length': 20}},
    ]
    track_df = build_modular_track(track_elements)
    return track_df

def debug_coordinate_system():
    """Debug coordinate mappings and conventions"""
    print("="*80)
    print("COORDINATE SYSTEM DEBUG")
    print("="*80)
    
    track_df = create_test_track()
    
    # Original track coordinates
    x = track_df['x'].values
    y = track_df['y'].values
    z = track_df.get('z', pd.Series(np.zeros_like(x))).values
    
    print("\n1. ORIGINAL TRACK COORDINATES:")
    print(f"   x (forward): [{x.min():.2f}, {x.max():.2f}] m")
    print(f"   y (vertical/height): [{y.min():.2f}, {y.max():.2f}] m")
    print(f"   z (lateral): [{z.min():.2f}, {z.max():.2f}] m")
    print(f"   First point: x={x[0]:.2f}, y={y[0]:.2f}, z={z[0]:.2f}")
    print(f"   Last point:  x={x[-1]:.2f}, y={y[-1]:.2f}, z={z[-1]:.2f}")
    
    # Remapped for acceleration.py
    points = np.column_stack([x, z, y])  # (forward, lateral, vertical)
    
    print("\n2. REMAPPED FOR acceleration.py:")
    print(f"   points[:, 0] (forward/x): [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}] m")
    print(f"   points[:, 1] (lateral/y): [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}] m")
    print(f"   points[:, 2] (vertical/z): [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}] m")
    print(f"   First point: [{points[0, 0]:.2f}, {points[0, 1]:.2f}, {points[0, 2]:.2f}]")
    print(f"   Last point:  [{points[-1, 0]:.2f}, {points[-1, 1]:.2f}, {points[-1, 2]:.2f}]")
    
    print("\n3. GRAVITY VECTOR (acceleration.py):")
    print(f"   G_VEC = {G_VEC}")
    print(f"   G_NORM = {G_NORM:.2f} m/s²")
    print(f"   Convention: Z-down (negative Z = downward)")
    
    # Check height calculations
    print("\n4. HEIGHT CALCULATIONS:")
    h_simple = y  # Simple model uses y directly
    h_advanced = points[:, 2]  # Advanced model uses z (which is y after remapping)
    
    print(f"   Simple model height (y): [{h_simple.min():.2f}, {h_simple.max():.2f}] m")
    print(f"   Advanced model height (points[:,2]): [{h_advanced.min():.2f}, {h_advanced.max():.2f}] m")
    print(f"   Heights match: {np.allclose(h_simple, h_advanced)}")
    
    # Check speed calculations
    print("\n5. SPEED CALCULATIONS:")
    
    # Simple model
    df_simple = compute_track_derivatives(track_df)
    v_simple = df_simple['velocity'].values
    h_initial_simple = y[0]
    print(f"   Simple model:")
    print(f"     h_initial = {h_initial_simple:.2f} m")
    print(f"     v = sqrt(2*g*(h_initial - y) * 0.95)")
    print(f"     Speed range: [{v_simple.min():.2f}, {v_simple.max():.2f}] m/s")
    
    # Advanced model (energy conservation)
    acc_energy = compute_acc_profile(
        points,
        dt=0.02,
        v0=0.0,
        use_energy_conservation=True
    )
    v_energy = acc_energy['v']
    h_initial_advanced = points[0, 2]
    print(f"   Advanced model (energy):")
    print(f"     h_initial = {h_initial_advanced:.2f} m")
    print(f"     v = sqrt(v0² + 2*g*(h_initial - h) * 0.95)")
    print(f"     Speed range: [{v_energy.min():.2f}, {v_energy.max():.2f}] m/s")
    print(f"     Speeds match: {np.allclose(v_simple, v_energy, rtol=0.01)}")
    
    # Advanced model (integration)
    acc_integration = compute_acc_profile(
        points,
        dt=0.02,
        v0=1.0,
        use_energy_conservation=False,
        use_velocity_verlet=True
    )
    v_integration = acc_integration['v']
    print(f"   Advanced model (integration, v0=1.0):")
    print(f"     Speed range: [{v_integration.min():.2f}, {v_integration.max():.2f}] m/s")
    
    # Check gravity component calculations
    print("\n6. GRAVITY COMPONENT ANALYSIS:")
    e_tan = acc_energy['e_tan']
    g_par_mag = e_tan @ G_VEC
    print(f"   Gravity parallel to tangent (g_par_mag):")
    print(f"     Range: [{g_par_mag.min():.2f}, {g_par_mag.max():.2f}] m/s²")
    print(f"     First 5 values: {g_par_mag[:5]}")
    print(f"     Last 5 values: {g_par_mag[-5:]}")
    
    # Check tangent vectors
    print("\n7. TANGENT VECTOR ANALYSIS:")
    print(f"   Tangent magnitude range: [{np.linalg.norm(e_tan, axis=1).min():.4f}, {np.linalg.norm(e_tan, axis=1).max():.4f}]")
    print(f"   First tangent: {e_tan[0]}")
    print(f"   Last tangent: {e_tan[-1]}")
    print(f"   Tangent Z component (vertical): [{e_tan[:, 2].min():.2f}, {e_tan[:, 2].max():.2f}]")
    
    # Check specific force
    print("\n8. SPECIFIC FORCE ANALYSIS:")
    f_spec = acc_energy['f_spec']
    print(f"   f_spec = a_tot - G_VEC")
    print(f"   f_spec range (X): [{f_spec[:, 0].min():.2f}, {f_spec[:, 0].max():.2f}] m/s²")
    print(f"   f_spec range (Y): [{f_spec[:, 1].min():.2f}, {f_spec[:, 1].max():.2f}] m/s²")
    print(f"   f_spec range (Z): [{f_spec[:, 2].min():.2f}, {f_spec[:, 2].max():.2f}] m/s²")
    
    # Check G-force projections
    print("\n9. G-FORCE PROJECTIONS:")
    print(f"   f_long_g (longitudinal): [{acc_energy['f_long_g'].min():.2f}, {acc_energy['f_long_g'].max():.2f}] g")
    print(f"   f_lat_g (lateral): [{acc_energy['f_lat_g'].min():.2f}, {acc_energy['f_lat_g'].max():.2f}] g")
    print(f"   f_vert_g (vertical): [{acc_energy['f_vert_g'].min():.2f}, {acc_energy['f_vert_g'].max():.2f}] g")
    
    # Compare with simple model
    print("\n10. COMPARISON WITH SIMPLE MODEL:")
    result_simple = compute_rider_accelerations(track_df)
    print(f"   Simple model G-forces:")
    print(f"     Lateral: [{result_simple['Lateral'].min():.2f}, {result_simple['Lateral'].max():.2f}] g")
    print(f"     Vertical: [{result_simple['Vertical'].min():.2f}, {result_simple['Vertical'].max():.2f}] g")
    print(f"     Longitudinal: [{result_simple['Longitudinal'].min():.2f}, {result_simple['Longitudinal'].max():.2f}] g")
    
    print(f"   Advanced model G-forces:")
    print(f"     Lateral (f_lat_g): [{acc_energy['f_lat_g'].min():.2f}, {acc_energy['f_lat_g'].max():.2f}] g")
    print(f"     Vertical (f_vert_g): [{acc_energy['f_vert_g'].min():.2f}, {acc_energy['f_vert_g'].max():.2f}] g")
    print(f"     Longitudinal (f_long_g): [{acc_energy['f_long_g'].min():.2f}, {acc_energy['f_long_g'].max():.2f}] g")
    
    # Create comparison plot
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    time = np.linspace(0, len(track_df) * 0.02, len(track_df))
    
    # Speed comparison
    axes[0, 0].plot(time, v_simple*3.6, label='Simple (energy)', linewidth=2)
    axes[0, 0].plot(time, v_energy*3.6, label='Advanced (energy)', linewidth=2, alpha=0.7)
    axes[0, 0].plot(time, v_integration*3.6, label='Advanced (integration)', linewidth=2, alpha=0.7)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Speed (km/h)')
    axes[0, 0].set_title('Speed Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Height comparison
    axes[0, 1].plot(time, h_simple, label='Simple (y)', linewidth=2)
    axes[0, 1].plot(time, h_advanced, label='Advanced (points[:,2])', linewidth=2, alpha=0.7)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Height (m)')
    axes[0, 1].set_title('Height Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Lateral G comparison
    axes[1, 0].plot(time, result_simple['Lateral'], label='Simple', linewidth=2)
    axes[1, 0].plot(time, acc_energy['f_lat_g'], label='Advanced', linewidth=2, alpha=0.7)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Lateral G-Force (g)')
    axes[1, 0].set_title('Lateral G-Forces')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Vertical G comparison
    axes[1, 1].plot(time, result_simple['Vertical'], label='Simple (normal to track)', linewidth=2)
    axes[1, 1].plot(time, acc_energy['f_vert_g'], label='Advanced (global Z)', linewidth=2, alpha=0.7)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Vertical G-Force (g)')
    axes[1, 1].set_title('Vertical G-Forces (Different Definitions!)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Longitudinal G comparison
    axes[2, 0].plot(time, result_simple['Longitudinal'], label='Simple', linewidth=2)
    axes[2, 0].plot(time, acc_energy['f_long_g'], label='Advanced', linewidth=2, alpha=0.7)
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Longitudinal G-Force (g)')
    axes[2, 0].set_title('Longitudinal G-Forces')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Differences
    axes[2, 1].plot(time, np.abs(result_simple['Lateral'] - acc_energy['f_lat_g']), 
                    label='Lateral diff', linewidth=2)
    axes[2, 1].plot(time, np.abs(result_simple['Vertical'] - acc_energy['f_vert_g']), 
                    label='Vertical diff', linewidth=2)
    axes[2, 1].plot(time, np.abs(result_simple['Longitudinal'] - acc_energy['f_long_g']), 
                    label='Longitudinal diff', linewidth=2)
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Absolute Difference (g)')
    axes[2, 1].set_title('G-Force Differences')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(__file__).parent.parent / 'coordinate_system_debug.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    debug_coordinate_system()

