import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List

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
                {'type': 'climb', 'params': {'length': 40, 'height': 35}},
                {'type': 'drop', 'params': {'length': 50, 'angle': 65}},
                {'type': 'clothoid_loop', 'params': {'radius': 12, 'transition_length': 15}},
                {'type': 'parabolic_curve', 'params': {'amplitude': 6, 'length': 40}},
            ],
        },
        {
            'name': 'family_coaster_hills',
            'elements': [
                {'type': 'climb', 'params': {'length': 50, 'height': 25}},
                {'type': 'drop', 'params': {'length': 60, 'angle': 50}},
                {'type': 'hills', 'params': {'num_hills': 3, 'amplitude': 5, 'wavelength': 35}},
                {'type': 'parabolic_curve', 'params': {'amplitude': 4, 'length': 50}},
            ],
        },
        {
            'name': 'spiral_turn_safe',
            'elements': [
                {'type': 'climb', 'params': {'length': 35, 'height': 20}},
                {'type': 'drop', 'params': {'length': 45, 'angle': 55}},
                {'type': 'rotation', 'params': {'angle': 180, 'radius': 10, 'axis': 'roll'}},
                {'type': 'parabolic_curve', 'params': {'amplitude': 5, 'length': 35}},
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
