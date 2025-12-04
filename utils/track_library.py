import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from .acceleration import compute_acc_profile
from .track import build_modular_track

LIB_DIR = Path('data') / 'tracks'
META_FILE = LIB_DIR / 'library.json'

def _to_points(track_df: 'np.ndarray') -> np.ndarray:
    # Ensure z exists; default to zeros if not provided
    if 'z' not in track_df.columns:
        track_df = track_df.copy()
        track_df['z'] = 0.0
    return track_df[['x','y','z']].to_numpy(dtype=float)

def _save_npz(path: Path, arrs: Dict[str, np.ndarray]):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **arrs)

def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(str(path), allow_pickle=False)
    return {k: data[k] for k in data.files}

def _default_library_specs() -> List[Dict]:
    return [
        {
            'name': 'simple_loop_1',
            'elements': [
                {'type': 'drop', 'params': {'height': 30, 'steepness': 0.85}},
                {'type': 'flat_section', 'params': {'length': 30}},
                {'type': 'clothoid_loop', 'params': {'radius': 12, 'transition_length': 15}},
                {'type': 'parabolic_curve', 'params': {'amplitude': 6, 'length': 40}},
            ],
        },
        {
            'name': 'family_coaster_hills',
            'elements': [
                {'type': 'drop', 'params': {'height': 28, 'steepness': 0.8}},
                {'type': 'flat_section', 'params': {'length': 25}},
                {'type': 'hills', 'params': {'num_hills': 3, 'amplitude': 5, 'wavelength': 35}},
                {'type': 'parabolic_curve', 'params': {'amplitude': 4, 'length': 50}},
            ],
        },
        {
            'name': 'spiral_turn_safe',
            'elements': [
                {'type': 'drop', 'params': {'height': 26, 'steepness': 0.78}},
                {'type': 'flat_section', 'params': {'length': 25}},
                {'type': 'rotation', 'params': {'angle': 180, 'radius': 10, 'axis': 'roll'}},
                {'type': 'parabolic_curve', 'params': {'amplitude': 5, 'length': 35}},
            ],
        },
        {
            'name': 'drop_flat_s_curves',
            'elements': [
                {'type': 'drop', 'params': {'height': 30, 'steepness': 0.82}},
                {'type': 'flat_section', 'params': {'length': 30}},
                {'type': 'parabolic_curve', 'params': {'amplitude': 5, 'length': 30}},
                {'type': 'parabolic_curve', 'params': {'amplitude': -5, 'length': 30}},
                {'type': 'flat_section', 'params': {'length': 20}},
            ],
        },
        {
            'name': 'mild_hills_combo',
            'elements': [
                {'type': 'drop', 'params': {'height': 24, 'steepness': 0.75}},
                {'type': 'flat_section', 'params': {'length': 25}},
                {'type': 'hills', 'params': {'num_hills': 4, 'amplitude': 3.5, 'wavelength': 28}},
                {'type': 'flat_section', 'params': {'length': 20}},
            ],
        },
        {
            'name': 'double_loop_safe',
            'elements': [
                {'type': 'drop', 'params': {'height': 32, 'steepness': 0.83}},
                {'type': 'flat_section', 'params': {'length': 35}},
                {'type': 'clothoid_loop', 'params': {'radius': 11, 'transition_length': 16}},
                {'type': 'flat_section', 'params': {'length': 20}},
                {'type': 'clothoid_loop', 'params': {'radius': 10, 'transition_length': 15}},
            ],
        },
        {
            'name': 'gentle_spiral_turns',
            'elements': [
                {'type': 'drop', 'params': {'height': 26, 'steepness': 0.78}},
                {'type': 'flat_section', 'params': {'length': 25}},
                {'type': 'rotation', 'params': {'angle': 120, 'radius': 12, 'axis': 'roll'}},
                {'type': 'flat_section', 'params': {'length': 25}},
            ],
        },
        {
            'name': 'airtime_hills',
            'elements': [
                {'type': 'drop', 'params': {'height': 34, 'steepness': 0.86}},
                {'type': 'flat_section', 'params': {'length': 25}},
                {'type': 'hills', 'params': {'num_hills': 3, 'amplitude': 6, 'wavelength': 38}},
                {'type': 'flat_section', 'params': {'length': 25}},
            ],
        },
        {
            'name': 'compact_family',
            'elements': [
                {'type': 'drop', 'params': {'height': 22, 'steepness': 0.72}},
                {'type': 'flat_section', 'params': {'length': 20}},
                {'type': 'parabolic_curve', 'params': {'amplitude': 4, 'length': 28}},
                {'type': 'hills', 'params': {'num_hills': 2, 'amplitude': 3, 'wavelength': 30}},
            ],
        },
        {
            'name': 'extended_turns_combo',
            'elements': [
                {'type': 'drop', 'params': {'height': 28, 'steepness': 0.8}},
                {'type': 'flat_section', 'params': {'length': 30}},
                {'type': 'rotation', 'params': {'angle': 200, 'radius': 14, 'axis': 'roll'}},
                {'type': 'parabolic_curve', 'params': {'amplitude': 5, 'length': 40}},
            ],
        },
        {
            'name': 'loop_hills_mix',
            'elements': [
                {'type': 'drop', 'params': {'height': 30, 'steepness': 0.82}},
                {'type': 'flat_section', 'params': {'length': 30}},
                {'type': 'clothoid_loop', 'params': {'radius': 12, 'transition_length': 15}},
                {'type': 'hills', 'params': {'num_hills': 2, 'amplitude': 4, 'wavelength': 32}},
            ],
        },
        {
            'name': 'sweeping_parabolas',
            'elements': [
                {'type': 'drop', 'params': {'height': 27, 'steepness': 0.79}},
                {'type': 'flat_section', 'params': {'length': 25}},
                {'type': 'parabolic_curve', 'params': {'amplitude': 6, 'length': 45}},
                {'type': 'parabolic_curve', 'params': {'amplitude': -4, 'length': 35}},
            ],
        },
        {
            'name': 'balanced_family_mix',
            'elements': [
                {'type': 'drop', 'params': {'height': 25, 'steepness': 0.76}},
                {'type': 'flat_section', 'params': {'length': 25}},
                {'type': 'hills', 'params': {'num_hills': 3, 'amplitude': 4, 'wavelength': 30}},
                {'type': 'parabolic_curve', 'params': {'amplitude': 4, 'length': 30}},
            ],
        },
        {
            'name': 'twist_and_sweep',
            'elements': [
                {'type': 'drop', 'params': {'height': 29, 'steepness': 0.81}},
                {'type': 'flat_section', 'params': {'length': 28}},
                {'type': 'rotation', 'params': {'angle': 160, 'radius': 11, 'axis': 'roll'}},
                {'type': 'parabolic_curve', 'params': {'amplitude': 5, 'length': 36}},
            ],
        },
        {
            'name': 'gentle_double_s',
            'elements': [
                {'type': 'drop', 'params': {'height': 23, 'steepness': 0.73}},
                {'type': 'flat_section', 'params': {'length': 22}},
                {'type': 'parabolic_curve', 'params': {'amplitude': 3.5, 'length': 26}},
                {'type': 'parabolic_curve', 'params': {'amplitude': -3.5, 'length': 26}},
            ],
        },
        {
            'name': 'classic_out_and_back',
            'elements': [
                {'type': 'drop', 'params': {'height': 31, 'steepness': 0.84}},
                {'type': 'flat_section', 'params': {'length': 35}},
                {'type': 'hills', 'params': {'num_hills': 4, 'amplitude': 4.5, 'wavelength': 34}},
                {'type': 'flat_section', 'params': {'length': 30}},
            ],
        },
        {
            'name': 'compact_loop_twist',
            'elements': [
                {'type': 'drop', 'params': {'height': 28, 'steepness': 0.8}},
                {'type': 'flat_section', 'params': {'length': 28}},
                {'type': 'clothoid_loop', 'params': {'radius': 11, 'transition_length': 14}},
                {'type': 'rotation', 'params': {'angle': 90, 'radius': 10, 'axis': 'roll'}},
            ],
        },
    ]

def ensure_library(dt: float = 0.02) -> List[Dict]:
    """Create the precomputed track library if missing and return metadata entries.
    Each entry contains name and file paths to geometry and physics arrays.
    """
    LIB_DIR.mkdir(parents=True, exist_ok=True)
    if META_FILE.exists():
        try:
            with open(META_FILE, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                # validate files exist
                valid = []
                for m in meta:
                    geo = Path(m['geometry_npz'])
                    phys = Path(m['physics_npz'])
                    if geo.exists() and phys.exists():
                        valid.append(m)
                if valid:
                    return valid
        except Exception:
            pass

    specs = _default_library_specs()
    entries: List[Dict] = []
    for spec in specs:
        name = spec['name']
        track_df = build_modular_track(spec['elements'])
        pts = _to_points(track_df)
        acc = compute_acc_profile(pts, dt=dt)
        # save arrays
        geo_path = LIB_DIR / f"{name}_geometry.npz"
        phys_path = LIB_DIR / f"{name}_physics.npz"
        _save_npz(geo_path, {'points': pts})
        _save_npz(phys_path, {
            'f_lat_g': acc['f_lat_g'],
            'f_vert_g': acc['f_vert_g'],
            'f_long_g': acc['f_long_g'],
            'f_lat': acc['f_lat'],
            'f_vert': acc['f_vert'],
            'f_long': acc['f_long'],
        })
        entries.append({
            'name': name,
            'geometry_npz': str(geo_path),
            'physics_npz': str(phys_path),
            'elements': spec['elements'],
        })

    with open(META_FILE, 'w', encoding='utf-8') as f:
        json.dump(entries, f, indent=2)
    return entries

def pick_random_entry(entries: List[Dict]) -> Dict:
    idx = np.random.randint(0, len(entries))
    return entries[idx]

def load_entry(entry: Dict):
    geo = _load_npz(Path(entry['geometry_npz']))
    phys = _load_npz(Path(entry['physics_npz']))
    return geo, phys

def add_entry(name: str, elements: List[Dict], track_df: 'np.ndarray', dt: float = 0.02) -> Optional[Dict]:
    """Add a new precomputed design to the library. Returns metadata entry or None on failure."""
    try:
        LIB_DIR.mkdir(parents=True, exist_ok=True)
        pts = _to_points(track_df)
        acc = compute_acc_profile(pts, dt=dt)
        geo_path = LIB_DIR / f"{name}_geometry.npz"
        phys_path = LIB_DIR / f"{name}_physics.npz"
        _save_npz(geo_path, {'points': pts})
        _save_npz(phys_path, {
            'f_lat_g': acc['f_lat_g'],
            'f_vert_g': acc['f_vert_g'],
            'f_long_g': acc['f_long_g'],
            'f_lat': acc['f_lat'],
            'f_vert': acc['f_vert'],
            'f_long': acc['f_long'],
        })
        # Load existing meta
        entries = []
        if META_FILE.exists():
            with open(META_FILE, 'r', encoding='utf-8') as f:
                entries = json.load(f)
        # Append or replace by name
        meta = {
            'name': name,
            'geometry_npz': str(geo_path),
            'physics_npz': str(phys_path),
            'elements': elements,
        }
        entries = [e for e in entries if e.get('name') != name]
        entries.append(meta)
        with open(META_FILE, 'w', encoding='utf-8') as f:
            json.dump(entries, f, indent=2)
        return meta
    except Exception:
        return None
