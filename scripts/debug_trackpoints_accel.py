# Debug script to compute acceleration profile from trackpoints using advanced physics model
import sys
import os
import pandas as pd
import numpy as np
import glob
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
import pandas as pd
from utils.acceleration import compute_acc_profile

# Load trackpoints.txt (assume comma-separated, no header)
track_path = 'physics/trackpoints.txt'
def load_trackpoints_with_bom_handling(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    # Remove BOM if present in first line
    if lines and lines[0].startswith('\ufeff'):
        lines[0] = lines[0].replace('\ufeff', '')
    from io import StringIO
    return np.loadtxt(StringIO(''.join(lines)), delimiter=',')

try:
    track_data = load_trackpoints_with_bom_handling(track_path)
except Exception as e:
    print(f"Error loading trackpoints: {e}")
    raise

# If units are not meters, convert here (e.g., mm to m)
# track_data = track_data / 1000.0

# Ensure shape is Nx3
if track_data.ndim != 2 or track_data.shape[1] != 3:
    raise ValueError('Trackpoints file must have Nx3 columns (x, y, z)')

# Run advanced physics model
acc_result = compute_acc_profile(track_data)

# Build DataFrame of outputs
df = pd.DataFrame({
    'index': np.arange(len(track_data)),
    'x': track_data[:, 0],
    'y': track_data[:, 1],
    'z': track_data[:, 2],
    'speed_mps': acc_result['v'],
    'long_g': acc_result['f_long_g'],
    'lat_g': acc_result['f_lat_g'],
    'vert_g': acc_result['f_vert_g'],
})

# Print summary
print(df.head())
print(df.describe())

# Optionally, save to CSV for further analysis
# df.to_csv('physics/trackpoints_accel_output.csv', index=False)


# Plot acceleration profiles
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plt.plot(df['index'], df['long_g'], label='Longitudinal g', color='blue')
plt.plot(df['index'], df['lat_g'], label='Lateral g', color='orange')
plt.plot(df['index'], df['vert_g'], label='Vertical g', color='green')
plt.xlabel('Track Point Index')
plt.ylabel('Acceleration (g)')
plt.title('Acceleration Profiles Along Track')
plt.legend()
plt.grid(True)
plt.show()