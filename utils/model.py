# utils/model.py
import numpy as np

def predict_thrill(features):
    # Placeholder model: weighted sum heuristic (replace with real ML model later)
    thrill = 0.02 * features["max_height"] + 0.5 * features["max_slope"]
    
    # Add contributions from element types
    element_counts = features.get("element_types", {})
    
    # Weight different element types
    thrill += 1.5 * element_counts.get("loop", 0)
    thrill += 2.0 * element_counts.get("clothoid_loop", 0)
    thrill += 0.5 * element_counts.get("hills", 0)
    thrill += 1.2 * element_counts.get("drop", 0)
    thrill += 1.0 * element_counts.get("gaussian_curve", 0)
    thrill += 0.8 * element_counts.get("parabolic_curve", 0)
    thrill += 1.5 * element_counts.get("rotation", 0)
    
    # Add variety bonus
    thrill += 0.3 * len(element_counts)
    
    return np.clip(thrill, 0, 10)
