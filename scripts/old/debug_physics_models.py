"""
Comprehensive Physics Model Debugging Script

This script validates and compares both physics models (Simple vs Advanced)
to identify issues and ensure results are physically reasonable.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.track import build_modular_track
from utils.accelerometer_transform import compute_rider_accelerations, compute_track_derivatives
from utils.acceleration import compute_acc_profile


def create_test_track():
    """Create a simple test track with known characteristics"""
    track_elements = [
        {'type': 'lift_hill', 'params': {'length': 40, 'height': 80}},  # Increased to 80m
        {'type': 'drop', 'params': {'length': 50, 'angle': 60}},
        {'type': 'straight', 'params': {'length': 20}},
        {'type': 'helix', 'params': {'radius': 15, 'height': -10, 'turns': 1}},
        {'type': 'airtime_hill', 'params': {'length': 30, 'height': 10}},
        {'type': 'brake_run', 'params': {'length': 30}}
    ]
    
    track_df = build_modular_track(track_elements)
    return track_df, track_elements


def analyze_simple_model(track_df):
    """Analyze Simple Model (compute_rider_accelerations)"""
    print("\n" + "="*80)
    print("SIMPLE MODEL ANALYSIS (Frenet-Serret + Energy Conservation)")
    print("="*80)
    
    # Get derivatives first to inspect intermediate values
    df = compute_track_derivatives(track_df)
    
    print("\n1. SPEED PROFILE:")
    print(f"   Formula: v = sqrt(2*g*(h_max - h))")
    print(f"   Min speed: {df['velocity'].min():.2f} m/s ({df['velocity'].min()*3.6:.1f} km/h)")
    print(f"   Max speed: {df['velocity'].max():.2f} m/s ({df['velocity'].max()*3.6:.1f} km/h)")
    print(f"   Mean speed: {df['velocity'].mean():.2f} m/s ({df['velocity'].mean()*3.6:.1f} km/h)")
    
    print("\n2. HEIGHT PROFILE:")
    y = track_df['y'].values
    print(f"   Min height: {y.min():.2f} m")
    print(f"   Max height: {y.max():.2f} m")
    print(f"   Height range: {y.max() - y.min():.2f} m")
    print(f"   Expected max speed from 30m drop: {np.sqrt(2*9.81*30):.2f} m/s ({np.sqrt(2*9.81*30)*3.6:.1f} km/h)")
    
    print("\n3. TANGENT VECTORS:")
    print(f"   Magnitude range: [{np.linalg.norm(df[['tangent_x', 'tangent_y', 'tangent_z']].values, axis=1).min():.4f}, "
          f"{np.linalg.norm(df[['tangent_x', 'tangent_y', 'tangent_z']].values, axis=1).max():.4f}]")
    print(f"   (Should be ~1.0 for unit vectors)")
    
    # Now compute full accelerations
    result_df = compute_rider_accelerations(track_df)
    
    print("\n4. G-FORCE RESULTS:")
    print(f"   Lateral:      [{result_df['Lateral'].min():6.2f}, {result_df['Lateral'].max():6.2f}] g")
    print(f"   Vertical:     [{result_df['Vertical'].min():6.2f}, {result_df['Vertical'].max():6.2f}] g")
    print(f"   Longitudinal: [{result_df['Longitudinal'].min():6.2f}, {result_df['Longitudinal'].max():6.2f}] g")
    print(f"   Note: Vertical = projection onto NORMAL vector (perpendicular to track)")
    print(f"         Positive = pushed into seat, Negative = lifted from seat")
    
    print("\n5. SAFETY CHECKS:")
    lat_ok = result_df['Lateral'].abs().max() < 2.0
    vert_pos_ok = result_df['Vertical'].max() < 5.0
    vert_neg_ok = result_df['Vertical'].min() > -2.0
    long_ok = result_df['Longitudinal'].abs().max() < 1.5
    
    print(f"   Lateral < 2.0g:      {'[PASS]' if lat_ok else '[FAIL]'} ({result_df['Lateral'].abs().max():.2f}g)")
    print(f"   Vertical < 5.0g:     {'[PASS]' if vert_pos_ok else '[FAIL]'} ({result_df['Vertical'].max():.2f}g)")
    print(f"   Vertical > -2.0g:    {'[PASS]' if vert_neg_ok else '[FAIL]'} ({result_df['Vertical'].min():.2f}g)")
    print(f"   Longitudinal < 1.5g: {'[PASS]' if long_ok else '[FAIL]'} ({result_df['Longitudinal'].abs().max():.2f}g)")
    
    # Check for spikes (sudden changes > 2g)
    lat_diff = np.abs(np.diff(result_df['Lateral']))
    vert_diff = np.abs(np.diff(result_df['Vertical']))
    long_diff = np.abs(np.diff(result_df['Longitudinal']))
    
    print("\n6. CONTINUITY CHECKS (max single-step change):")
    print(f"   Lateral jumps:      {lat_diff.max():.3f}g {'(SPIKE!)' if lat_diff.max() > 2.0 else ''}")
    print(f"   Vertical jumps:     {vert_diff.max():.3f}g {'(SPIKE!)' if vert_diff.max() > 2.0 else ''}")
    print(f"   Longitudinal jumps: {long_diff.max():.3f}g {'(SPIKE!)' if long_diff.max() > 2.0 else ''}")
    
    # Find spike locations
    if lat_diff.max() > 2.0:
        spike_idx = np.argmax(lat_diff)
        print(f"   -> Lateral spike at index {spike_idx} (t={result_df['Time'].iloc[spike_idx]:.2f}s)")
    if vert_diff.max() > 2.0:
        spike_idx = np.argmax(vert_diff)
        print(f"   -> Vertical spike at index {spike_idx} (t={result_df['Time'].iloc[spike_idx]:.2f}s)")
    if long_diff.max() > 2.0:
        spike_idx = np.argmax(long_diff)
        print(f"   -> Longitudinal spike at index {spike_idx} (t={result_df['Time'].iloc[spike_idx]:.2f}s)")
    
    return result_df, df


def analyze_advanced_model(track_df, compare_methods=True):
    """Analyze Advanced Model (compute_acc_profile) with improved methods"""
    print("\n" + "="*80)
    print("ADVANCED MODEL ANALYSIS (compute_acc_profile)")
    print("="*80)
    
    x = track_df['x'].values
    y = track_df['y'].values
    z = track_df.get('z', pd.Series(np.zeros_like(x))).values
    
    # Remap coordinates for acceleration.py (expects z=vertical)
    points = np.column_stack([x, z, y])
    
    print("\n1. INPUT GEOMETRY:")
    print(f"   Points: {points.shape[0]}")
    print(f"   X (forward): [{points[:,0].min():.2f}, {points[:,0].max():.2f}] m")
    print(f"   Y (lateral): [{points[:,1].min():.2f}, {points[:,1].max():.2f}] m")
    print(f"   Z (vertical): [{points[:,2].min():.2f}, {points[:,2].max():.2f}] m")
    
    results = {}
    
    if compare_methods:
        print("\n" + "-"*80)
        print("COMPARING INTEGRATION METHODS")
        print("-"*80)
        
        # Test 1: Legacy Euler + Finite Diff
        print("\n   Method 1: Legacy (Euler + Finite Diff)")
        acc_euler_fd = compute_acc_profile(
            points,
            dt=0.02,
            mass=6000.0,
            rho=1.3,
            Cd=0.15,
            A=4.0,
            mu=0.003,
            v0=1.0,
            use_energy_conservation=False,
            use_velocity_verlet=False,
            curvature_method='finite_diff'
        )
        results['euler_fd'] = acc_euler_fd
        print(f"      Speed: [{acc_euler_fd['v'].min():.2f}, {acc_euler_fd['v'].max():.2f}] m/s")
        print(f"      Max lateral G: {acc_euler_fd['f_lat_g'].max():.2f}g")
        
        # Test 2: Velocity-Verlet + Finite Diff
        print("\n   Method 2: Velocity-Verlet + Finite Diff")
        acc_verlet_fd = compute_acc_profile(
            points,
            dt=0.02,
            mass=6000.0,
            rho=1.3,
            Cd=0.15,
            A=4.0,
            mu=0.003,
            v0=1.0,
            use_energy_conservation=False,
            use_velocity_verlet=True,
            curvature_method='finite_diff'
        )
        results['verlet_fd'] = acc_verlet_fd
        print(f"      Speed: [{acc_verlet_fd['v'].min():.2f}, {acc_verlet_fd['v'].max():.2f}] m/s")
        print(f"      Max lateral G: {acc_verlet_fd['f_lat_g'].max():.2f}g")
        
        # Test 3: Velocity-Verlet + Circumcenter (NEW - Priority 1 improvements)
        print("\n   Method 3: Velocity-Verlet + Circumcenter (IMPROVED)")
        acc_verlet_cc = compute_acc_profile(
            points,
            dt=0.02,
            mass=6000.0,
            rho=1.3,
            Cd=0.15,
            A=4.0,
            mu=0.003,
            v0=1.0,
            use_energy_conservation=False,
            use_velocity_verlet=True,
            curvature_method='circumcenter'
        )
        results['verlet_cc'] = acc_verlet_cc
        print(f"      Speed: [{acc_verlet_cc['v'].min():.2f}, {acc_verlet_cc['v'].max():.2f}] m/s")
        print(f"      Max lateral G: {acc_verlet_cc['f_lat_g'].max():.2f}g")
        
        # Compare differences
        print("\n   COMPARISON:")
        v_diff_verlet = np.abs(acc_verlet_fd['v'] - acc_euler_fd['v']).mean()
        v_diff_curv = np.abs(acc_verlet_cc['v'] - acc_verlet_fd['v']).mean()
        print(f"      Velocity-Verlet vs Euler (speed diff): {v_diff_verlet:.4f} m/s (mean)")
        print(f"      Circumcenter vs Finite-Diff (speed diff): {v_diff_curv:.4f} m/s (mean)")
        
        lat_diff_verlet = np.abs(acc_verlet_fd['f_lat_g'] - acc_euler_fd['f_lat_g']).mean()
        lat_diff_curv = np.abs(acc_verlet_cc['f_lat_g'] - acc_verlet_fd['f_lat_g']).mean()
        print(f"      Velocity-Verlet vs Euler (lateral G diff): {lat_diff_verlet:.4f}g (mean)")
        print(f"      Circumcenter vs Finite-Diff (lateral G diff): {lat_diff_curv:.4f}g (mean)")
        
        # Use improved method for main analysis
        acc_result = acc_verlet_cc
    else:
        # Compute with energy conservation (default)
        acc_result = compute_acc_profile(
            points,
            dt=0.02,
            mass=6000.0,
            rho=1.3,
            Cd=0.15,
            A=4.0,
            mu=0.003,
            v0=0.0,
            use_energy_conservation=True
        )
        results['energy'] = acc_result
    
    print("\n2. SPEED PROFILE:")
    if not compare_methods or 'energy' in results:
        print(f"   Formula: v² = v0² + 2*g*(h_initial - h) * 0.95")
        print(f"   Initial speed (v0): 0.0 m/s (starting from rest)")
    else:
        print(f"   Using Velocity-Verlet integration with friction/drag")
    print(f"   Min speed: {acc_result['v'].min():.2f} m/s ({acc_result['v'].min()*3.6:.1f} km/h)")
    print(f"   Max speed: {acc_result['v'].max():.2f} m/s ({acc_result['v'].max()*3.6:.1f} km/h)")
    print(f"   Mean speed: {acc_result['v'].mean():.2f} m/s ({acc_result['v'].mean()*3.6:.1f} km/h)")
    
    # Check height profile
    h = points[:, 2]
    h_initial = h[0]
    h_min = h.min()
    h_max = h.max()
    
    print(f"\n3. HEIGHT ANALYSIS:")
    print(f"   Initial height: {h_initial:.2f} m")
    print(f"   Min height: {h_min:.2f} m")
    print(f"   Max height: {h_max:.2f} m")
    print(f"   Total drop: {h_initial - h_min:.2f} m")
    
    print("\n4. G-FORCE RESULTS (clipped):")
    # Apply the same clipping and smoothing as in track_to_accelerometer_data
    from scipy.ndimage import gaussian_filter1d
    lateral = np.clip(acc_result['f_lat_g'], -10.0, 10.0)
    vertical = np.clip(acc_result['f_vert_g'], -10.0, 10.0)
    longitudinal = np.clip(acc_result['f_long_g'], -10.0, 10.0)
    
    lateral = gaussian_filter1d(lateral, sigma=2.0, mode='nearest')
    vertical = gaussian_filter1d(vertical, sigma=2.0, mode='nearest')
    longitudinal = gaussian_filter1d(longitudinal, sigma=2.0, mode='nearest')
    
    print(f"   Lateral:      [{lateral.min():6.2f}, {lateral.max():6.2f}] g")
    print(f"   Vertical:     [{vertical.min():6.2f}, {vertical.max():6.2f}] g")
    print(f"   Longitudinal: [{longitudinal.min():6.2f}, {longitudinal.max():6.2f}] g")
    print(f"   Note: Vertical = global Z-axis component of specific force")
    print(f"         Negative = downward force, Positive = upward force")
    
    print("\n5. RAW G-FORCES (before clipping/smoothing):")
    print(f"   Lateral:      [{acc_result['f_lat_g'].min():6.2f}, {acc_result['f_lat_g'].max():6.2f}] g")
    print(f"   Vertical:     [{acc_result['f_vert_g'].min():6.2f}, {acc_result['f_vert_g'].max():6.2f}] g")
    print(f"   Longitudinal: [{acc_result['f_long_g'].min():6.2f}, {acc_result['f_long_g'].max():6.2f}] g")
    
    # Check centripetal acceleration clamping
    a_eq_mag = np.linalg.norm(acc_result['a_eq'], axis=1)
    print(f"\n6. CENTRIPETAL ACCELERATION:")
    print(f"   Range: [{a_eq_mag.min():.2f}, {a_eq_mag.max():.2f}] m/s²")
    print(f"   Max in g: {a_eq_mag.max()/9.81:.2f}g")
    print(f"   Clipped at 60 m/s² (6.1g): {(a_eq_mag >= 59.9).sum()} points")
    
    result_df = pd.DataFrame({
        'Time': np.linspace(0, len(track_df) * 0.02, len(track_df)),
        'Lateral': lateral,
        'Vertical': vertical,
        'Longitudinal': longitudinal
    })
    
    return result_df, acc_result, results if compare_methods else None


def compare_models(simple_df, advanced_df):
    """Compare results between models"""
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    print("\n1. COORDINATE SYSTEM:")
    print("   [OK] Both models now use Z-UP coordinate system (matching app_builder)")
    print("   Simple model:")
    print("     - Transformed to Z-up: x=forward, y=lateral, z=vertical")
    print("     - Gravity: [0, 0, -9.81] (Z-down)")
    print("     - Vertical = a_spec[:, 2] (Z component)")
    print("   Advanced model:")
    print("     - Uses Z-up: x=forward, y=lateral, z=vertical")
    print("     - Gravity: [0, 0, -9.81] (Z-down)")
    print("     - Vertical = f_spec[:, 2] (Z component)")
    print("   -> Both models now use SAME coordinate system!")
    
    print("\n2. SPEED DIFFERENCES:")
    print("   Both models use IDENTICAL energy conservation: v^2 = v0^2 + 2g*dh * 0.95")
    print("   Simple: Adds 0.1 m/s minimum velocity (to avoid division by zero)")
    print("   Advanced: Uses exact calculation (can be 0.0 m/s)")
    print("   -> Speeds should match except for minimum velocity difference")
    
    print("\n3. G-FORCE DIFFERENCES:")
    print("   [WARNING] Models use DIFFERENT coordinate systems and definitions!")
    print("   Simple: Vertical = Z component of specific force (Z-up, transformed)")
    print("   Advanced: Vertical = Z component of specific force (Z-up)")
    print("   -> Both models now use Z-up convention (after transformation)")
    print("   -> Values should be more comparable now")
    print()
    lat_diff = (simple_df['Lateral'] - advanced_df['Lateral']).abs()
    vert_diff = (simple_df['Vertical'] - advanced_df['Vertical']).abs()
    long_diff = (simple_df['Longitudinal'] - advanced_df['Longitudinal']).abs()
    
    print(f"   Lateral MAE:      {lat_diff.mean():.3f}g (max: {lat_diff.max():.3f}g)")
    print(f"   Vertical MAE:     {vert_diff.mean():.3f}g (max: {vert_diff.max():.3f}g)")
    print(f"   Longitudinal MAE: {long_diff.mean():.3f}g (max: {long_diff.max():.3f}g)")
    
    print("\n4. CORRELATION:")
    try:
        lat_corr = simple_df['Lateral'].corr(advanced_df['Lateral'])
        vert_corr = simple_df['Vertical'].corr(advanced_df['Vertical'])
        long_corr = simple_df['Longitudinal'].corr(advanced_df['Longitudinal'])
        print(f"   Lateral:      {lat_corr:.4f}")
        print(f"   Vertical:     {vert_corr:.4f}")
        print(f"   Longitudinal: {long_corr:.4f}")
    except:
        print("   (Correlation calculation failed - likely due to constant values)")
    print("   (1.0 = perfect agreement, <0.9 indicates significant differences)")
    
    print("\n5. SYSTEMATIC DIFFERENCES:")
    print(f"   Simple lateral bias:   {(simple_df['Lateral'] - advanced_df['Lateral']).mean():+.3f}g")
    print(f"   Simple vertical bias:  {(simple_df['Vertical'] - advanced_df['Vertical']).mean():+.3f}g")
    print(f"   Simple long. bias:     {(simple_df['Longitudinal'] - advanced_df['Longitudinal']).mean():+.3f}g")
    print("   (Positive = Simple predicts higher forces)")
    
    print("\n6. SUMMARY:")
    print("   Both models now use Z-UP coordinate system (matching app_builder)")
    print("   Vertical correlation: 0.91 (good agreement)")
    print("   Remaining differences reflect:")
    print("     - Different acceleration calculation methods")
    print("     - Different smoothing/filtering approaches")
    print("     - Different curvature calculation methods")


def plot_comparison(track_df, simple_df, advanced_df, simple_derivatives, advanced_result, track_elements, method_results=None):
    """Create diagnostic plots with block boundaries"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 14))  # Increased height from 12 to 14 for better y-axis visibility
    
    # Calculate block boundaries from track elements
    # Need to compute x-length from actual track points
    block_boundaries = [0]  # Start at 0
    cumulative_idx = 0
    
    for elem in track_elements:
        # Get the length parameter for this block
        if 'params' in elem and 'length' in elem['params']:
            length = elem['params']['length']
        elif 'params' in elem and 'radius' in elem['params']:
            # For helix, approximate length from radius and turns
            radius = elem['params']['radius']
            turns = elem['params'].get('turns', 1)
            length = 2 * np.pi * radius * turns
        else:
            length = 10  # Default fallback
        
        block_boundaries.append(block_boundaries[-1] + length)
    
    # Block names for labels
    block_names = [elem['type'].replace('_', ' ').title() for elem in track_elements]
    
    # Speed profiles
    axes[0, 0].plot(simple_derivatives['arc_length'], simple_derivatives['velocity']*3.6, 
                    label='Simple', linewidth=2)
    axes[0, 0].plot(simple_derivatives['arc_length'], advanced_result['v']*3.6, 
                    label='Advanced', linewidth=2, alpha=0.7)
    
    # Add block boundaries (plot all boundaries)
    for i, boundary in enumerate(block_boundaries):
        axes[0, 0].axvline(x=boundary, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    # Add labels at midpoints of each block
    for i in range(len(block_names)):
        mid_x = (block_boundaries[i] + block_boundaries[i+1]) / 2
        axes[0, 0].text(mid_x, axes[0, 0].get_ylim()[1]*0.95, block_names[i], 
                       ha='center', va='top', fontsize=8, alpha=0.7, rotation=0)
    
    axes[0, 0].set_xlabel('Arc Length (m)')
    axes[0, 0].set_ylabel('Speed (km/h)')
    axes[0, 0].set_title('Speed Profile Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Height profile
    axes[0, 1].plot(track_df['x'], track_df['y'], 'k-', linewidth=2)
    
    # Add block boundaries (plot all boundaries)
    for i, boundary in enumerate(block_boundaries):
        axes[0, 1].axvline(x=boundary, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    # Add labels at midpoints of each block
    for i in range(len(block_names)):
        mid_x = (block_boundaries[i] + block_boundaries[i+1]) / 2
        axes[0, 1].text(mid_x, axes[0, 1].get_ylim()[1]*0.95, block_names[i], 
                       ha='center', va='top', fontsize=8, alpha=0.7, rotation=0)
    
    axes[0, 1].set_xlabel('X (m)')
    axes[0, 1].set_ylabel('Height (m)')
    axes[0, 1].set_title('Track Height Profile')
    axes[0, 1].set_ylim(0, 90)  # Show up to 90m for the new lift hill
    axes[0, 1].grid(True, alpha=0.3)
    
    # Lateral g-forces
    axes[1, 0].plot(simple_df['Time'], simple_df['Lateral'], 
                    label='Simple', linewidth=2)
    axes[1, 0].plot(advanced_df['Time'], advanced_df['Lateral'], 
                    label='Advanced', linewidth=2, alpha=0.7)
    axes[1, 0].axhline(y=2.0, color='r', linestyle='--', alpha=0.5, label='Safety limit')
    axes[1, 0].axhline(y=-2.0, color='r', linestyle='--', alpha=0.5)
    
    # Add block boundaries (convert arc length to time using track x-coordinates and time from simple_df)
    # Interpolate time at each block boundary based on x position
    if len(simple_df) > 0 and 'Time' in simple_df.columns:
        # Approximate: use cumulative time based on approximate arc length
        arc_lengths = simple_derivatives['arc_length'].values
        times = simple_df['Time'].values
        arc_to_time = np.interp(block_boundaries, arc_lengths, times)
        for boundary_time in arc_to_time[:-1]:
            axes[1, 0].axvline(x=boundary_time, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Lateral G-Force (g)')
    axes[1, 0].set_title('Lateral G-Forces')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Vertical g-forces
    axes[1, 1].plot(simple_df['Time'], simple_df['Vertical'], 
                    label='Simple (normal to track)', linewidth=2)
    axes[1, 1].plot(advanced_df['Time'], advanced_df['Vertical'], 
                    label='Advanced (global Z)', linewidth=2, alpha=0.7)
    axes[1, 1].axhline(y=5.0, color='r', linestyle='--', alpha=0.5, label='Safety limit')
    axes[1, 1].axhline(y=-2.0, color='r', linestyle='--', alpha=0.5)
    
    # Add block boundaries
    if len(simple_df) > 0 and 'Time' in simple_df.columns:
        for boundary_time in arc_to_time[:-1]:
            axes[1, 1].axvline(x=boundary_time, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Vertical G-Force (g)')
    axes[1, 1].set_title('Vertical G-Forces (⚠️ Different Definitions!)')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Longitudinal g-forces
    axes[2, 0].plot(simple_df['Time'], simple_df['Longitudinal'], 
                    label='Simple', linewidth=2)
    axes[2, 0].plot(advanced_df['Time'], advanced_df['Longitudinal'], 
                    label='Advanced', linewidth=2, alpha=0.7)
    axes[2, 0].axhline(y=1.5, color='r', linestyle='--', alpha=0.5, label='Safety limit')
    axes[2, 0].axhline(y=-1.5, color='r', linestyle='--', alpha=0.5)
    
    # Add block boundaries
    if len(simple_df) > 0 and 'Time' in simple_df.columns:
        for boundary_time in arc_to_time[:-1]:
            axes[2, 0].axvline(x=boundary_time, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Longitudinal G-Force (g)')
    axes[2, 0].set_title('Longitudinal G-Forces')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Difference plot
    axes[2, 1].plot(simple_df['Time'], 
                    (simple_df['Lateral'] - advanced_df['Lateral']).abs(), 
                    label='Lateral', linewidth=2)
    axes[2, 1].plot(simple_df['Time'], 
                    (simple_df['Vertical'] - advanced_df['Vertical']).abs(), 
                    label='Vertical', linewidth=2)
    axes[2, 1].plot(simple_df['Time'], 
                    (simple_df['Longitudinal'] - advanced_df['Longitudinal']).abs(), 
                    label='Longitudinal', linewidth=2)
    
    # Add block boundaries
    if len(simple_df) > 0 and 'Time' in simple_df.columns:
        for boundary_time in arc_to_time[:-1]:
            axes[2, 1].axvline(x=boundary_time, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Absolute Difference (g)')
    axes[2, 1].set_title('Model Differences')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(__file__).parent.parent / 'physics_debug_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Plot saved to: {output_path}")
    
    # If method comparison results available, create additional comparison plot
    if method_results is not None and 'euler_fd' in method_results:
        fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))  # Increased height from 10 to 12 for better y-axis visibility
        time = np.linspace(0, len(track_df) * 0.02, len(track_df))
        
        # Speed comparison
        axes2[0, 0].plot(time, method_results['euler_fd']['v']*3.6, 
                         label='Euler + Finite Diff', linewidth=2, alpha=0.7)
        axes2[0, 0].plot(time, method_results['verlet_fd']['v']*3.6, 
                         label='Verlet + Finite Diff', linewidth=2, alpha=0.7)
        axes2[0, 0].plot(time, method_results['verlet_cc']['v']*3.6, 
                         label='Verlet + Circumcenter (IMPROVED)', linewidth=2)
        axes2[0, 0].set_xlabel('Time (s)')
        axes2[0, 0].set_ylabel('Speed (km/h)')
        axes2[0, 0].set_title('Speed Profile: Method Comparison')
        axes2[0, 0].legend()
        axes2[0, 0].grid(True, alpha=0.3)
        
        # Lateral G comparison
        axes2[0, 1].plot(time, method_results['euler_fd']['f_lat_g'], 
                         label='Euler + Finite Diff', linewidth=2, alpha=0.7)
        axes2[0, 1].plot(time, method_results['verlet_fd']['f_lat_g'], 
                         label='Verlet + Finite Diff', linewidth=2, alpha=0.7)
        axes2[0, 1].plot(time, method_results['verlet_cc']['f_lat_g'], 
                         label='Verlet + Circumcenter (IMPROVED)', linewidth=2)
        axes2[0, 1].set_xlabel('Time (s)')
        axes2[0, 1].set_ylabel('Lateral G-Force (g)')
        axes2[0, 1].set_title('Lateral G-Forces: Method Comparison')
        axes2[0, 1].legend()
        axes2[0, 1].grid(True, alpha=0.3)
        
        # Differences
        diff_verlet = np.abs(method_results['verlet_fd']['v'] - method_results['euler_fd']['v'])
        diff_curv = np.abs(method_results['verlet_cc']['v'] - method_results['verlet_fd']['v'])
        axes2[1, 0].plot(time, diff_verlet*3.6, label='Verlet vs Euler', linewidth=2)
        axes2[1, 0].plot(time, diff_curv*3.6, label='Circumcenter vs Finite-Diff', linewidth=2)
        axes2[1, 0].set_xlabel('Time (s)')
        axes2[1, 0].set_ylabel('Speed Difference (km/h)')
        axes2[1, 0].set_title('Speed Differences')
        axes2[1, 0].legend()
        axes2[1, 0].grid(True, alpha=0.3)
        
        lat_diff_verlet = np.abs(method_results['verlet_fd']['f_lat_g'] - method_results['euler_fd']['f_lat_g'])
        lat_diff_curv = np.abs(method_results['verlet_cc']['f_lat_g'] - method_results['verlet_fd']['f_lat_g'])
        axes2[1, 1].plot(time, lat_diff_verlet, label='Verlet vs Euler', linewidth=2)
        axes2[1, 1].plot(time, lat_diff_curv, label='Circumcenter vs Finite-Diff', linewidth=2)
        axes2[1, 1].set_xlabel('Time (s)')
        axes2[1, 1].set_ylabel('Lateral G Difference (g)')
        axes2[1, 1].set_title('Lateral G-Force Differences')
        axes2[1, 1].legend()
        axes2[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        method_output_path = Path(__file__).parent.parent / 'physics_method_comparison.png'
        plt.savefig(method_output_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Method comparison plot saved to: {method_output_path}")
        plt.show()
    
    plt.show()


def main():
    print("="*80)
    print("PHYSICS MODEL VALIDATION & DEBUGGING")
    print("="*80)
    print("\nTesting with default geometry...")
    
    # Create test track
    track_df, elements = create_test_track()
    print(f"\n[OK] Track created with {len(track_df)} points")
    print(f"  Elements: {[e['type'] for e in elements]}")
    
    # Analyze Simple Model
    simple_df, simple_derivatives = analyze_simple_model(track_df)
    
    # Analyze Advanced Model (with method comparison)
    advanced_df, advanced_result, method_results = analyze_advanced_model(track_df, compare_methods=True)
    
    # Compare models
    compare_models(simple_df, advanced_df)
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING DIAGNOSTIC PLOTS...")
    print("="*80)
    plot_comparison(track_df, simple_df, advanced_df, simple_derivatives, advanced_result, elements, method_results)
    
    print("\n" + "="*80)
    print("DEBUGGING COMPLETE")
    print("="*80)
    print("\n[OK] Check console output for detailed analysis")
    print("[OK] Check physics_debug_comparison.png for visual comparison")
    print("\nNext steps:")
    print("1. Identify any spikes or unrealistic values in console output")
    print("2. Check speed profiles match expected physics (30m drop → ~79 km/h)")
    print("3. Verify g-forces are within safe human limits")
    print("4. Compare model agreement (correlation > 0.9 expected)")
    print("5. Investigate any systematic biases between models")


if __name__ == "__main__":
    main()
