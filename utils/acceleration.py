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
    """Compute curvature radii per sample using circumcenter over sliding windows.
    Returns an array of length N with \infty at endpoints and degenerate windows."""
    n = points.shape[0]
    R = np.full(n, np.inf, dtype=float)
    if n < 3:
        return R
    a = points[:-2]
    b = points[1:-1]
    c = points[2:]
    ab = b - a
    ac = c - a
    cross = np.cross(ab, ac)
    cross_norm2 = np.einsum('ij,ij->i', cross, cross)
    mask = cross_norm2 >= 1e-12
    if not np.any(mask):
        return R
    ab_len2 = np.einsum('ij,ij->i', ab, ab)
    ac_len2 = np.einsum('ij,ij->i', ac, ac)
    num = np.cross(cross, ab) * ac_len2[:, None] + np.cross(ac, cross) * ab_len2[:, None]
    center = a + num / (2.0 * cross_norm2[:, None])
    Rvals = np.linalg.norm(a - center, axis=1)
    Rmid = np.where(Rvals > 1e-9, Rvals, np.inf)
    # map to indices 1..n-2
    R[1:-1] = Rmid
    return R


def compute_acc_profile(points: np.ndarray,
                        dt: float = 0.02,
                        mass: float = 6000.0,
                        rho: float = 1.3,
                        Cd: float = 0.6,
                        A: float = 4.0,
                        mu: float = 0.02,
                        v0: float = 1.0) -> Dict[str, np.ndarray]:
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
    e_tan = _tangents(points)

    # Vectorized gravity parallel magnitude per sample
    g_par_mag = e_tan @ G_VEC
    # normal magnitude approximation ||g - g_parallel||
    g_par_vec = (g_par_mag[:, None] * e_tan)
    g_N_vec = G_VEC - g_par_vec
    normal_mag = np.linalg.norm(g_N_vec, axis=1)

    # Speed integration (semi-implicit).
    # Use Numba JIT if available for the simple recurrence loop.
    v = np.zeros(n, dtype=np.float64)
    a_tan = np.zeros(n, dtype=np.float64)
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
    # Vectorized curvature radii
    R = _curvature_radius_vectorized(points)
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
    valid = np.isfinite(R) & (R > 1e-6)
    a_eq[valid] = -((v[valid]**2 / R[valid])[:, None] * n_hat[valid])

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
