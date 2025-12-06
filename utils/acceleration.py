import numpy as np
from typing import Dict, Tuple

# Gravity (global Z-up convention)
G_VEC = np.array([0.0, 0.0, -9.81], dtype=float)
G_NORM = float(np.linalg.norm(G_VEC))

# Optional Numba acceleration
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False


def _safe_norm(v: np.ndarray, eps: float = 1e-9) -> Tuple[float, np.ndarray]:
    n = float(np.linalg.norm(v))
    if n < eps:
        return 0.0, v * 0.0
    return n, v / n


def _tangents(points: np.ndarray) -> np.ndarray:
    n = points.shape[0]
    tangents = np.zeros_like(points)
    # central differences for interior
    t_mid = (points[2:] - points[:-2]) * 0.5
    # normalize
    norms = np.linalg.norm(t_mid, axis=1, keepdims=True)
    safe = np.where(norms < 1e-9, 1.0, norms)
    t_unit_mid = t_mid / safe
    tangents[1:-1] = t_unit_mid
    # endpoints via forward/backward difference
    t0 = points[1] - points[0]
    tn = points[-1] - points[-2]
    n0 = np.linalg.norm(t0)
    nn = np.linalg.norm(tn)
    tangents[0] = t0 / (n0 if n0 > 1e-9 else 1.0)
    tangents[-1] = tn / (nn if nn > 1e-9 else 1.0)
    # fallback for degenerate
    deg = np.where(np.linalg.norm(tangents, axis=1) < 1e-12)[0]
    if deg.size:
        tangents[deg] = np.array([1.0, 0.0, 0.0])
    return tangents


def _curvature_radius_vectorized(points: np.ndarray) -> np.ndarray:
    """Compute curvature radii using finite differences on smoothed tangent vectors.
    More robust than 3-point circumcenter for discretized curves.
    Returns an array of length N with large values where curvature is near-zero."""
    n = points.shape[0]
    R = np.full(n, 1000.0, dtype=float)  # Default to large radius (nearly straight)
    if n < 5:
        return R
    
    # Compute tangent vectors using central differences with wider stencil
    tangents = np.zeros_like(points)
    # Use 5-point stencil for interior points (more stable)
    for i in range(2, n-2):
        # Central difference with spacing 2
        tangents[i] = points[i+2] - points[i-2]
    # Endpoints use narrower stencils
    tangents[0] = points[1] - points[0]
    tangents[1] = points[2] - points[0]
    tangents[-2] = points[-1] - points[-3]
    tangents[-1] = points[-1] - points[-2]
    
    # Normalize tangents
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.where(norms < 1e-9, 1.0, norms)
    
    # Curvature from rate of change of tangent direction
    # κ = |dT/ds| where T is unit tangent
    dt = np.zeros_like(tangents)
    dt[1:-1] = tangents[2:] - tangents[:-2]  # Central diff
    dt[0] = tangents[1] - tangents[0]
    dt[-1] = tangents[-1] - tangents[-2]
    
    # Arc length element (distance between points)
    ds = np.linalg.norm(np.diff(points, axis=0, prepend=points[0:1]), axis=1)
    ds = np.maximum(ds, 1e-6)
    
    # Curvature magnitude
    kappa = np.linalg.norm(dt, axis=1) / ds
    
    # Radius = 1/curvature, with safeguards
    # Minimum curvature = 1/1000m (very gentle), maximum = 1/5m (tight loop)
    # Increased minimum radius from 1m to 5m to reduce numerical artifacts
    kappa_clamped = np.clip(kappa, 1e-3, 0.2)
    R = 1.0 / kappa_clamped
    
    return R


def compute_acc_profile(points: np.ndarray,
                        dt: float = 0.02,
                        mass: float = 6000.0,
                        rho: float = 1.3,
                        Cd: float = 0.6,
                        A: float = 4.0,
                        mu: float = 0.02,
                        v0: float = 1.0,
                        use_energy_conservation: bool = False) -> Dict[str, np.ndarray]:
    """
    Compute per-sample inertial acceleration and accelerometer specific-force from 3D track points.

    Inputs:
    - points: Nx3 array of track coordinates in meters, ordered along the path
    - dt: sample step to project accelerations (s). Used for simple speed integration.
    - mass, rho, Cd, A, mu: physical parameters (used for drag/friction magnitudes)
    - v0: initial speed (m/s)

    Outputs (dict of arrays length N):
    - e_tan: unit tangent vectors
    - v: scalar speed (m/s)
    - a_tan: scalar tangential acceleration (m/s^2)
    - a_eq: centripetal acceleration vector (m/s^2)
    - a_tot: total inertial acceleration vector (a_tan*e_tan + a_eq)
    - f_spec: specific force vector (a_tot - g)
    - long/lat/vert: projections of a_tot in local axes
    - f_long/f_lat/f_vert: projections of specific force
    - f_long_g/f_lat_g/f_vert_g: projections normalized by g
    """
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
        raise ValueError("points must be an Nx3 array with N>=3")

    n = points.shape[0]
    
    # Apply Gaussian smoothing to reduce discretization noise
    # This is critical for curvature calculations on point clouds
    from scipy.ndimage import gaussian_filter1d
    points_smooth = points.copy()
    sigma = 2.0  # Moderate smoothing - balances noise reduction with peak preservation
    for i in range(3):
        points_smooth[:, i] = gaussian_filter1d(points[:, i], sigma=sigma, mode='nearest')
    
    e_tan = _tangents(points_smooth)

    # Vectorized gravity parallel magnitude per sample
    g_par_mag = e_tan @ G_VEC
    # normal magnitude approximation ||g - g_parallel||
    g_par_vec = (g_par_mag[:, None] * e_tan)
    g_N_vec = G_VEC - g_par_vec
    normal_mag = np.linalg.norm(g_N_vec, axis=1)

    # Speed calculation
    v = np.zeros(n, dtype=np.float64)
    a_tan = np.zeros(n, dtype=np.float64)
    
    if use_energy_conservation:
        # Use energy conservation for speeds (more reliable for coasters)
        # v = sqrt(2*g*(h_max - h))
        h = points[:, 2]  # Z-coordinate is vertical in our convention
        h_max = np.max(h)
        v_ideal = np.sqrt(2 * G_NORM * np.maximum(0, h_max - h))
        
        # Apply energy loss factor to account for friction/drag/track imperfections
        # Real coasters lose 15-25% of speed compared to ideal free fall
        # This gives more realistic G-forces
        energy_efficiency = 0.80  # 80% efficiency (20% loss)
        v = v_ideal * energy_efficiency
        v = np.maximum(v, v0)  # Don't go below initial speed
        
        # Tangential acceleration: use gravity parallel component directly
        # This is more accurate than finite differences on speed
        # a_tan = g·tangent (component of gravity along track direction)
        a_tan = g_par_mag  # Already computed above
    else:
        # Speed integration (semi-implicit) with friction/drag
        # Use Numba JIT if available for the simple recurrence loop.
        v[0] = float(v0)
        k_drag = float(0.5 * rho * Cd * A / mass)

        if NUMBA_AVAILABLE:
            @njit(cache=True, fastmath=True)
            def _integrate_speed(g_par_mag_arr, normal_mag_arr, v0_val, dt_val, k_drag_val, mu_val):
                N = g_par_mag_arr.shape[0]
                v_out = np.zeros(N, dtype=np.float64)
                a_out = np.zeros(N, dtype=np.float64)
                v_out[0] = v0_val
                a_out[0] = 0.0
                for i in range(1, N):
                    friction_acc_mag = mu_val * normal_mag_arr[i]
                    drag_acc_mag = k_drag_val * v_out[i-1] * v_out[i-1]
                    sign_motion = 1.0 if v_out[i-1] >= 0.0 else -1.0
                    a_out[i] = g_par_mag_arr[i] - sign_motion * (friction_acc_mag + drag_acc_mag)
                    v_out[i] = v_out[i-1] + a_out[i] * dt_val
                    if v_out[i] < 0.0 and abs(v_out[i]) < 1e-9:
                        v_out[i] = 0.0
                return v_out, a_out

            v, a_tan = _integrate_speed(g_par_mag.astype(np.float64), normal_mag.astype(np.float64), float(v0), float(dt), k_drag, float(mu))
        else:
            for i in range(1, n):
                friction_acc_mag = mu * normal_mag[i]
                drag_acc_mag = k_drag * v[i-1]**2
                sign_motion = 1.0 if v[i-1] >= 0 else -1.0
                a_tan[i] = g_par_mag[i] - sign_motion * (friction_acc_mag + drag_acc_mag)
                v[i] = v[i-1] + a_tan[i] * dt
                if v[i] < 0 and abs(v[i]) < 1e-9:
                    v[i] = 0.0

    # Curvature-based centripetal acceleration
    # Vectorized curvature radii (use smoothed points)
    R = _curvature_radius_vectorized(points_smooth)
    # normal direction from derivative of tangent (vectorized)
    d_el = np.zeros_like(e_tan)
    d_el[1:] = e_tan[1:] - e_tan[:-1]
    d_norm = np.linalg.norm(d_el, axis=1, keepdims=True)
    d_safe = np.where(d_norm < 1e-9, 1.0, d_norm)
    n_hat = d_el / d_safe
    # fallback where derivative is near zero: use horizontal normal
    ez = np.array([0.0, 0.0, 1.0])
    lat = np.cross(ez, e_tan)
    lat_norm = np.linalg.norm(lat, axis=1, keepdims=True)
    lat_safe = np.where(lat_norm < 1e-9, 1.0, lat_norm)
    lat_hat = lat / lat_safe
    # choose n_hat where valid else lat_hat
    use_lat = (d_norm.flatten() < 1e-9)
    n_hat[use_lat] = lat_hat[use_lat]
    # centripetal acceleration
    a_eq = np.zeros((n, 3), dtype=float)
    # Use minimum radius of 5m to prevent numerical explosion
    valid = np.isfinite(R) & (R >= 5.0)
    # Clamp centripetal acceleration magnitude to realistic max (~6g = 60 m/s²)
    # Most intense modern coasters max at 5-6g
    a_cent_mag = np.clip(v[valid]**2 / R[valid], 0.0, 60.0)
    a_eq[valid] = -(a_cent_mag[:, None] * n_hat[valid])

    # Total inertial acceleration and specific force
    a_tot = a_tan.reshape(-1, 1) * e_tan + a_eq
    f_spec = a_tot - G_VEC

    # Local axes: longitudinal = tangent; vertical = global Z; lateral = cross(ez, tangent)
    long = np.zeros(n, dtype=float)
    lat = np.zeros(n, dtype=float)
    vert = np.zeros(n, dtype=float)
    f_long = np.zeros(n, dtype=float)
    f_lat = np.zeros(n, dtype=float)
    f_vert = np.zeros(n, dtype=float)

    # vectorized projections
    el = e_tan
    lat_vec = np.cross(ez, el)
    lat_n = np.linalg.norm(lat_vec, axis=1, keepdims=True)
    lat_vec = lat_vec / np.where(lat_n < 1e-9, 1.0, lat_n)
    long[:] = np.einsum('ij,ij->i', a_tot, el)
    lat[:]  = np.einsum('ij,ij->i', a_tot, lat_vec)
    vert[:] = a_tot[:, 2]
    f_long[:] = np.einsum('ij,ij->i', f_spec, el)
    f_lat[:]  = np.einsum('ij,ij->i', f_spec, lat_vec)
    f_vert[:] = f_spec[:, 2]

    return {
        'e_tan': e_tan,
        'v': v,
        'a_tan': a_tan,
        'a_eq': a_eq,
        'a_tot': a_tot,
        'f_spec': f_spec,
        'long': long,
        'lat': lat,
        'vert': vert,
        'f_long': f_long,
        'f_lat': f_lat,
        'f_vert': f_vert,
        'f_long_g': f_long / G_NORM,
        'f_lat_g': f_lat / G_NORM,
        'f_vert_g': f_vert / G_NORM,
    }
