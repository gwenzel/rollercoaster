# utils/model.py
import numpy as np

def predict_thrill(features):
    # Placeholder model: weighted sum heuristic (replace with real ML model later)
    thrill = 0.02 * features["max_height"] + 0.5 * features["max_slope"] + \
             0.1 * features["num_hills"] + 0.3 * (1/features["loop_radius"])
    return np.clip(thrill, 0, 10)
