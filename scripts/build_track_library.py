"""
Build the precomputed track library (geometry + physics) to accelerate the app.
Run with: python scripts/build_track_library.py
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure project root is on sys.path so `utils` can be imported
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.track_library import ensure_library

def main():
    entries = ensure_library(dt=0.02)
    print(f"Built/validated {len(entries)} precomputed tracks:")
    for e in entries:
        print(f"- {e['name']}:\n  geometry: {e['geometry_npz']}\n  physics:  {e['physics_npz']}")
    print("Done. You can now load random precomputed tracks from the app sidebar.")

if __name__ == "__main__":
    main()
