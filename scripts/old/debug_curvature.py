"""
Curvature Analysis and Peak Detection Script

This script specifically analyzes curvature calculations to identify
where peaks/spikes occur and diagnose their causes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.track import build_modular_track
from utils.accelerometer_transform import compute_track_derivatives
from scipy.ndimage import gaussian_filter1d


def compute_curvature_detailed(track_df, smoothing_sigma=1.5):
    """Compute curvature with detailed intermediate steps"""
    x = track_df['x'].values
    y = track_df['y'].values
    z = track_df.get('z', pd.Series(np.zeros_like(x))).values
    
    # Step 1: Smooth coordinates
    x_smooth = gaussian_filter1d(x, sigma=2, mode='nearest')
    y_smooth = gaussian_filter1d(y, sigma=2, mode='nearest')
    z_smooth = gaussian_filter1d(z, sigma=2, mode='nearest')
    
    # Step 2: Compute tangent vectors
    dx_ds = np.gradient(x_smooth)
    dy_ds = np.gradient(y_smooth)
    dz_ds = np.gradient(z_smooth)
    
    tangent_mag = np.sqrt(dx_ds**2 + dy_ds**2 + dz_ds**2)
    tangent_mag = np.maximum(tangent_mag, 1e-10)
    
    tangent_x = dx_ds / tangent_mag
    tangent_y = dy_ds / tangent_mag
    tangent_z = dz_ds / tangent_mag
    
    # Step 3: Smooth tangent vectors
    tangent_x_smooth = gaussian_filter1d(tangent_x, sigma=smoothing_sigma, mode='nearest')
    tangent_y_smooth = gaussian_filter1d(tangent_y, sigma=smoothing_sigma, mode='nearest')
    tangent_z_smooth = gaussian_filter1d(tangent_z, sigma=smoothing_sigma, mode='nearest')
    
    # Step 4: Compute curvature (before smoothing)
    dt_x = np.gradient(tangent_x_smooth)
    dt_y = np.gradient(tangent_y_smooth)
    dt_z = np.gradient(tangent_z_smooth)
    
    curvature_raw = np.sqrt(dt_x**2 + dt_y**2 + dt_z**2)
    
    # Step 5: Smooth curvature
    curvature_smooth = gaussian_filter1d(curvature_raw, sigma=1.0, mode='nearest')
    
    return {
        'x_raw': x,
        'y_raw': y,
        'z_raw': z,
        'x_smooth': x_smooth,
        'y_smooth': y_smooth,
        'z_smooth': z_smooth,
        'tangent_x': tangent_x,
        'tangent_y': tangent_y,
        'tangent_z': tangent_z,
        'tangent_x_smooth': tangent_x_smooth,
        'tangent_y_smooth': tangent_y_smooth,
        'tangent_z_smooth': tangent_z_smooth,
        'curvature_raw': curvature_raw,
        'curvature_smooth': curvature_smooth
    }


def detect_block_boundaries(track_df, track_elements):
    """Estimate where block boundaries are in the track"""
    boundaries = [0]
    cumulative_points = 0
    
    # Rough estimate: each block gets proportional points
    total_length = sum([e['params'].get('length', 20) for e in track_elements])
    
    for element in track_elements[:-1]:
        length = element['params'].get('length', 20)
        points_per_meter = len(track_df) / total_length
        cumulative_points += int(length * points_per_meter)
        boundaries.append(cumulative_points)
    
    boundaries.append(len(track_df))
    return boundaries, [e['type'] for e in track_elements]


def find_curvature_spikes(curvature, threshold_percentile=95):
    """Find indices where curvature spikes occur"""
    threshold = np.percentile(curvature, threshold_percentile)
    spike_indices = np.where(curvature > threshold)[0]
    
    # Group consecutive spikes
    spike_groups = []
    if len(spike_indices) > 0:
        current_group = [spike_indices[0]]
        for idx in spike_indices[1:]:
            if idx - current_group[-1] <= 3:  # Within 3 points = same spike
                current_group.append(idx)
            else:
                spike_groups.append(current_group)
                current_group = [idx]
        spike_groups.append(current_group)
    
    return spike_groups, threshold


def analyze_curvature(track_df, track_elements):
    """Comprehensive curvature analysis"""
    print("\n" + "="*80)
    print("CURVATURE ANALYSIS")
    print("="*80)
    
    # Compute curvature with details
    curv_data = compute_curvature_detailed(track_df)
    
    print("\n1. CURVATURE STATISTICS:")
    print(f"   Raw curvature:")
    print(f"     Min:    {curv_data['curvature_raw'].min():.6f}")
    print(f"     Max:    {curv_data['curvature_raw'].max():.6f}")
    print(f"     Mean:   {curv_data['curvature_raw'].mean():.6f}")
    print(f"     Std:    {curv_data['curvature_raw'].std():.6f}")
    print(f"   Smoothed curvature:")
    print(f"     Min:    {curv_data['curvature_smooth'].min():.6f}")
    print(f"     Max:    {curv_data['curvature_smooth'].max():.6f}")
    print(f"     Mean:   {curv_data['curvature_smooth'].mean():.6f}")
    print(f"     Std:    {curv_data['curvature_smooth'].std():.6f}")
    
    # Convert to radius of curvature
    radius_raw = 1.0 / np.maximum(curv_data['curvature_raw'], 1e-10)
    radius_smooth = 1.0 / np.maximum(curv_data['curvature_smooth'], 1e-10)
    
    print("\n2. RADIUS OF CURVATURE:")
    print(f"   Raw radius (where curvature > 0.01):")
    valid_raw = radius_raw[curv_data['curvature_raw'] > 0.01]
    if len(valid_raw) > 0:
        print(f"     Min:    {valid_raw.min():.2f} m")
        print(f"     Max:    {valid_raw.max():.2f} m")
        print(f"     Mean:   {valid_raw.mean():.2f} m")
    
    print(f"   Smoothed radius (where curvature > 0.01):")
    valid_smooth = radius_smooth[curv_data['curvature_smooth'] > 0.01]
    if len(valid_smooth) > 0:
        print(f"     Min:    {valid_smooth.min():.2f} m")
        print(f"     Max:    {valid_smooth.max():.2f} m")
        print(f"     Mean:   {valid_smooth.mean():.2f} m")
    
    # Detect spikes
    spike_groups, threshold = find_curvature_spikes(curv_data['curvature_smooth'])
    
    print(f"\n3. CURVATURE SPIKES (>{threshold:.6f}, top 5%):")
    print(f"   Number of spike groups: {len(spike_groups)}")
    
    if len(spike_groups) > 0:
        # Get block boundaries
        boundaries, block_types = detect_block_boundaries(track_df, track_elements)
        
        print(f"\n   Spike locations:")
        for i, group in enumerate(spike_groups[:10]):  # Show first 10
            center_idx = group[len(group)//2]
            max_curv = curv_data['curvature_smooth'][group].max()
            
            # Find which block this is in
            block_idx = 0
            for j, boundary in enumerate(boundaries[1:]):
                if center_idx < boundary:
                    block_idx = j
                    break
            
            # Distance from block boundary
            dist_from_start = center_idx - boundaries[block_idx]
            dist_from_end = boundaries[block_idx + 1] - center_idx
            near_boundary = min(dist_from_start, dist_from_end) < 5
            
            print(f"   #{i+1}: Index {center_idx:4d} "
                  f"(Block: {block_types[block_idx]:15s}) "
                  f"κ={max_curv:.6f} "
                  f"R={1/max_curv:.1f}m "
                  f"{'⚠️ NEAR JOINT' if near_boundary else ''}")
    
    # Check tangent vector smoothness
    tangent_jumps = np.sqrt(
        np.diff(curv_data['tangent_x'])**2 + 
        np.diff(curv_data['tangent_y'])**2 + 
        np.diff(curv_data['tangent_z'])**2
    )
    
    print(f"\n4. TANGENT VECTOR CONTINUITY:")
    print(f"   Max single-step change: {tangent_jumps.max():.6f}")
    print(f"   Mean single-step change: {tangent_jumps.mean():.6f}")
    print(f"   Large jumps (>0.1): {(tangent_jumps > 0.1).sum()}")
    
    if (tangent_jumps > 0.1).any():
        large_jump_idx = np.where(tangent_jumps > 0.1)[0]
        print(f"   Large jump locations: {large_jump_idx[:5]}")  # First 5
    
    return curv_data, boundaries, block_types


def plot_curvature_analysis(track_df, curv_data, boundaries, block_types):
    """Create detailed curvature plots"""
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    
    arc_length = np.cumsum(np.sqrt(
        np.diff(curv_data['x_raw'], prepend=curv_data['x_raw'][0])**2 +
        np.diff(curv_data['y_raw'], prepend=curv_data['y_raw'][0])**2 +
        np.diff(curv_data['z_raw'], prepend=curv_data['z_raw'][0])**2
    ))
    
    # Plot 1: Track geometry
    axes[0].plot(curv_data['x_raw'], curv_data['y_raw'], 'b-', linewidth=2, label='Track')
    
    # Mark block boundaries
    for i, (boundary, block_type) in enumerate(zip(boundaries[:-1], block_types)):
        x_bound = curv_data['x_raw'][boundary]
        y_bound = curv_data['y_raw'][boundary]
        axes[0].plot(x_bound, y_bound, 'ro', markersize=8)
        axes[0].annotate(block_type, (x_bound, y_bound), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Height (m)')
    axes[0].set_title('Track Geometry with Block Boundaries')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Curvature comparison
    axes[1].plot(arc_length, curv_data['curvature_raw'], 
                label='Raw κ', linewidth=1, alpha=0.5)
    axes[1].plot(arc_length, curv_data['curvature_smooth'], 
                label='Smoothed κ', linewidth=2)
    
    # Mark block boundaries
    for boundary in boundaries[1:-1]:
        s = arc_length[boundary]
        axes[1].axvline(x=s, color='r', linestyle='--', alpha=0.3)
    
    axes[1].set_xlabel('Arc Length (m)')
    axes[1].set_ylabel('Curvature κ (1/m)')
    axes[1].set_title('Curvature Profile')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Radius of curvature (capped at 100m for visibility)
    radius_smooth = 1.0 / np.maximum(curv_data['curvature_smooth'], 1e-10)
    radius_capped = np.minimum(radius_smooth, 100)
    
    axes[2].plot(arc_length, radius_capped, linewidth=2)
    axes[2].axhline(y=15, color='g', linestyle='--', alpha=0.5, label='Typical helix (15m)')
    axes[2].axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='Tight loop (5m)')
    
    # Mark block boundaries
    for boundary in boundaries[1:-1]:
        s = arc_length[boundary]
        axes[2].axvline(x=s, color='r', linestyle='--', alpha=0.3)
    
    axes[2].set_xlabel('Arc Length (m)')
    axes[2].set_ylabel('Radius of Curvature (m, capped at 100m)')
    axes[2].set_title('Radius of Curvature Profile')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Tangent vector changes (shows discontinuities)
    tangent_changes = np.sqrt(
        np.diff(curv_data['tangent_x_smooth'], prepend=0)**2 +
        np.diff(curv_data['tangent_y_smooth'], prepend=0)**2 +
        np.diff(curv_data['tangent_z_smooth'], prepend=0)**2
    )
    
    axes[3].plot(arc_length, tangent_changes, linewidth=1)
    axes[3].axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Large jump threshold')
    
    # Mark block boundaries
    for boundary in boundaries[1:-1]:
        s = arc_length[boundary]
        axes[3].axvline(x=s, color='r', linestyle='--', alpha=0.3)
    
    axes[3].set_xlabel('Arc Length (m)')
    axes[3].set_ylabel('Tangent Vector Change')
    axes[3].set_title('Tangent Vector Continuity (Large values = discontinuities)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(__file__).parent.parent / 'curvature_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {output_path}")
    
    plt.show()


def main():
    print("="*80)
    print("CURVATURE DEBUGGING AND SPIKE DETECTION")
    print("="*80)
    
    # Create test track
    track_elements = [
        {'type': 'lift_hill', 'params': {'length': 40, 'height': 30}},
        {'type': 'drop', 'params': {'length': 50, 'angle': 60}},
        {'type': 'straight', 'params': {'length': 20}},
        {'type': 'helix', 'params': {'radius': 15, 'height': -10, 'turns': 1}},
        {'type': 'airtime_hill', 'params': {'length': 30, 'height': 10}},
        {'type': 'brake_run', 'params': {'length': 30}}
    ]
    
    track_df = build_modular_track(track_elements)
    print(f"\n✓ Track created with {len(track_df)} points")
    
    # Analyze curvature
    curv_data, boundaries, block_types = analyze_curvature(track_df, track_elements)
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING CURVATURE PLOTS...")
    print("="*80)
    plot_curvature_analysis(track_df, curv_data, boundaries, block_types)
    
    print("\n" + "="*80)
    print("CURVATURE DEBUGGING COMPLETE")
    print("="*80)
    print("\n✓ Check console output for spike locations")
    print("✓ Check curvature_analysis.png for visual analysis")
    print("\nDiagnostic checklist:")
    print("1. Are spikes concentrated near block boundaries? (⚠️ NEAR JOINT)")
    print("2. Is raw curvature much noisier than smoothed? (needs more smoothing)")
    print("3. Are tangent vector jumps >0.1? (C1 discontinuity at joints)")
    print("4. Is radius < 5m anywhere? (physically unrealistic tight curves)")
    print("5. Do spike patterns correlate with specific block types?")


if __name__ == "__main__":
    main()
