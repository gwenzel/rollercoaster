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


def _circumcenter_radius_3pt(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute circumcenter radius from three points (geometric method, like Roller.py).
    
    Returns:
        R: radius of curvature (inf if points are colinear)
        center: center point of the circle
    """
    a = p1
    b = p2
    c = p3
    ab = b - a
    ac = c - a
    cross = np.cross(ab, ac)
    cross_norm2 = np.dot(cross, cross)
    if cross_norm2 < 1e-12:
        return np.inf, np.array([np.inf, np.inf, np.inf])  # Colinear points
    ab_len2 = np.dot(ab, ab)
    ac_len2 = np.dot(ac, ac)
    num = np.cross(cross, ab) * ac_len2 + np.cross(ac, cross) * ab_len2
    center = a + num / (2 * cross_norm2)
    Rvec = a - center
    R = np.linalg.norm(Rvec)
    return R, center


def _curvature_radius_circumcenter(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute curvature radii using circumcenter method (geometric, like Roller.py).
    More geometrically accurate for discrete track points, especially in tight turns.
    
    Returns:
        R: array of curvature radii (length N)
        centers: array of center points (Nx3), inf where invalid
    """
    n = points.shape[0]
    R = np.full(n, 1000.0, dtype=float)  # Default to large radius
    centers = np.full((n, 3), np.inf, dtype=float)
    
    if n < 3:
        return R, centers
    
    # Interior points: use 3-point circumcenter
    for i in range(1, n-1):
        idx1 = max(0, i-1)
        idx2 = i
        idx3 = min(n-1, i+1)
        R_val, center = _circumcenter_radius_3pt(
            points[idx1], points[idx2], points[idx3]
        )
        if np.isfinite(R_val) and R_val > 1e-6:
            R[i] = R_val
            centers[i] = center
    
    # Endpoints: use adjacent point's radius
    if n > 2:
        R[0] = R[1]
        centers[0] = centers[1]
        R[-1] = R[-2]
        centers[-1] = centers[-2]
    
    return R, centers


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
                        v0: float = 0.0,
                        use_energy_conservation: bool = False,
                        use_velocity_verlet: bool = True,
                        curvature_method: str = 'circumcenter',
                        launch_sections: list = None) -> Dict[str, np.ndarray]:
    """
    Compute per-sample inertial acceleration and accelerometer specific-force from 3D track points.

    Inputs:
    - points: Nx3 array of track coordinates in meters, ordered along the path
    - dt: sample step to project accelerations (s). Used for speed integration.
    - mass, rho, Cd, A, mu: physical parameters (used for drag/friction magnitudes)
    - v0: initial speed (m/s)
    - use_energy_conservation: if True, use energy-based speed calculation (ignores friction/drag)
    - use_velocity_verlet: if True, use Velocity-Verlet integration (more accurate, default)
                          if False, use semi-implicit Euler (legacy)
    - curvature_method: 'circumcenter' (geometric, more accurate) or 'finite_diff' (faster, default legacy)

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
    
    # Import gaussian_filter1d for later use in speed smoothing
    # (already imported above, but keeping for clarity)
    
    e_tan = _tangents(points_smooth)

    # Vectorized gravity parallel magnitude per sample
    g_par_mag = e_tan @ G_VEC
    # normal magnitude approximation ||g - g_parallel||
    g_par_vec = (g_par_mag[:, None] * e_tan)
    g_N_vec = G_VEC - g_par_vec
    normal_mag = np.linalg.norm(g_N_vec, axis=1)

    # CORRECT WORKFLOW: Forces → Acceleration → Velocity → Position → Relative Acceleration
    # 
    # Step 1: Calculate forces at each point
    #   - Gravity (parallel to tangent)
    #   - Friction (opposes motion)
    #   - Drag (opposes motion)
    #   - Launch force (during launch sections)
    #
    # Step 2: Calculate acceleration from forces: a = F/m
    #
    # Step 3: Integrate acceleration to get velocity: v = ∫a dt (Velocity-Verlet)
    #
    # Step 4: Calculate centripetal acceleration from curvature and velocity
    #
    # Step 5: Calculate relative acceleration (specific force) = a_tot - g
    
    # Calculate cumulative distance for launch section detection (use smoothed points for consistency)
    # This ensures launch sections are detected based on the same geometry used for velocity calculation
    ds_cumulative = np.zeros(n)
    for i in range(1, n):
        ds_cumulative[i] = ds_cumulative[i-1] + np.linalg.norm(points_smooth[i] - points_smooth[i-1])
    
    # Determine which points are in launch sections
    in_launch = np.zeros(n, dtype=bool)
    launch_acceleration = np.zeros(n, dtype=float)  # Additional acceleration from launch
    
    if launch_sections is not None and len(launch_sections) > 0:
        for launch_start_x, launch_end_x, target_speed in launch_sections:
            # Find indices in launch section
            in_launch_section = (ds_cumulative >= launch_start_x) & (ds_cumulative <= launch_end_x)
            in_launch = in_launch | in_launch_section
            
            if in_launch_section.any():
                launch_indices = np.where(in_launch_section)[0]
                if len(launch_indices) > 0:
                    launch_length = launch_end_x - launch_start_x
                    if launch_length > 0:
                        # Calculate constant acceleration needed: a = (v_target² - v0²) / (2*d)
                        # This will be applied as a force during integration
                        a_launch_mag = (target_speed**2 - v0**2) / (2 * launch_length)
                        launch_acceleration[launch_indices] = a_launch_mag
    
    # Initialize velocity and acceleration arrays
    v_estimate = np.zeros(n, dtype=np.float64)
    a_tan = np.zeros(n, dtype=np.float64)
    v_estimate[0] = float(v0)
    
    # Drag coefficient: k_drag = 0.5 * rho * Cd * A / mass
    k_drag = float(0.5 * rho * Cd * A / mass)
    
    if use_energy_conservation:
        # For energy conservation mode, we still need to calculate forces properly
        # but we'll use energy conservation as a check/override
        # This is a hybrid approach that ensures realistic physics
        # Use smoothed points for height to match velocity calculation
        h = points_smooth[:, 2]  # Z-coordinate is vertical (use smoothed for consistency)
        h_initial = h[0]
        energy_efficiency = 0.95  # 95% efficiency
        
        # Calculate energy-based speed estimate
        v_energy = np.sqrt(np.maximum(0, v0**2 + 2 * G_NORM * (h_initial - h) * energy_efficiency))
        
        # Still integrate forces to get realistic acceleration profile
        # But use energy conservation to guide/validate the result
        if use_velocity_verlet:
            if NUMBA_AVAILABLE:
                @njit(cache=True, fastmath=True)
                def _integrate_with_launch_verlet(g_par_mag_arr, normal_mag_arr, launch_acc_arr, v0_val, dt_val, k_drag_val, mu_val):
                    N = g_par_mag_arr.shape[0]
                    v_out = np.zeros(N, dtype=np.float64)
                    a_out = np.zeros(N, dtype=np.float64)
                    v_out[0] = v0_val
                    a_out[0] = 0.0
                    a_prev = 0.0
                    for i in range(1, N):
                        # Compute forces: gravity, friction, drag, launch
                        friction_acc_mag = mu_val * normal_mag_arr[i]
                        drag_acc_mag = k_drag_val * v_out[i-1] * v_out[i-1]
                        sign_motion = 1.0 if v_out[i-1] >= 0.0 else -1.0
                        # Add launch acceleration if in launch section
                        a_gravity = g_par_mag_arr[i]
                        a_launch = launch_acc_arr[i]  # Additional acceleration from launch
                        a_new = a_gravity + a_launch - sign_motion * (friction_acc_mag + drag_acc_mag)
                        
                        # Velocity-Verlet integration
                        v_half = v_out[i-1] + 0.5 * a_prev * dt_val
                        v_out[i] = v_half + 0.5 * a_new * dt_val
                        if v_out[i] < 0.0 and abs(v_out[i]) < 1e-9:
                            v_out[i] = 0.0
                        
                        a_out[i] = a_new
                        a_prev = a_new
                    return v_out, a_out
                
                v_estimate, a_tan = _integrate_with_launch_verlet(
                    g_par_mag.astype(np.float64), 
                    normal_mag.astype(np.float64),
                    launch_acceleration.astype(np.float64),
                    float(v0), float(dt), k_drag, float(mu)
                )
            else:
                a_prev = 0.0
                for i in range(1, n):
                    friction_acc_mag = mu * normal_mag[i]
                    drag_acc_mag = k_drag * v_estimate[i-1]**2
                    sign_motion = 1.0 if v_estimate[i-1] >= 0 else -1.0
                    # Add launch acceleration if in launch section
                    a_gravity = g_par_mag[i]
                    a_launch = launch_acceleration[i]
                    a_new = a_gravity + a_launch - sign_motion * (friction_acc_mag + drag_acc_mag)
                    
                    # Velocity-Verlet integration
                    v_half = v_estimate[i-1] + 0.5 * a_prev * dt
                    v_estimate[i] = v_half + 0.5 * a_new * dt
                    if v_estimate[i] < 0 and abs(v_estimate[i]) < 1e-9:
                        v_estimate[i] = 0.0
                    
                    a_tan[i] = a_new
                    a_prev = a_new
        
        # Use energy conservation as a guide (blend with integrated result)
        # This ensures we don't violate energy conservation while still having realistic acceleration
        v_estimate = np.maximum(v_estimate, v_energy * 0.9)  # Allow some loss, but respect energy
    else:
        # Full physics integration: Forces → Acceleration → Velocity
        if use_velocity_verlet:
            # Velocity-Verlet integration (2nd order, symplectic, energy-conserving)
            # v_half = v[i-1] + 0.5 * a[i-1] * dt
            # v[i] = v_half + 0.5 * a[i] * dt
            if NUMBA_AVAILABLE:
                @njit(cache=True, fastmath=True)
                def _integrate_speed_verlet(g_par_mag_arr, normal_mag_arr, launch_acc_arr, v0_val, dt_val, k_drag_val, mu_val):
                    N = g_par_mag_arr.shape[0]
                    v_out = np.zeros(N, dtype=np.float64)
                    a_out = np.zeros(N, dtype=np.float64)
                    v_out[0] = v0_val
                    a_out[0] = 0.0
                    a_prev = 0.0
                    for i in range(1, N):
                        # Compute forces: gravity, friction, drag, launch
                        friction_acc_mag = mu_val * normal_mag_arr[i]
                        drag_acc_mag = k_drag_val * v_out[i-1] * v_out[i-1]
                        sign_motion = 1.0 if v_out[i-1] >= 0.0 else -1.0
                        # Add launch acceleration if in launch section
                        a_gravity = g_par_mag_arr[i]
                        a_launch = launch_acc_arr[i]
                        a_new = a_gravity + a_launch - sign_motion * (friction_acc_mag + drag_acc_mag)
                        
                        # Velocity-Verlet: half-step velocity update
                        v_half = v_out[i-1] + 0.5 * a_prev * dt_val
                        v_out[i] = v_half + 0.5 * a_new * dt_val
                        if v_out[i] < 0.0 and abs(v_out[i]) < 1e-9:
                            v_out[i] = 0.0
                        
                        a_out[i] = a_new
                        a_prev = a_new
                    return v_out, a_out

                v_estimate, a_tan = _integrate_speed_verlet(
                    g_par_mag.astype(np.float64), 
                    normal_mag.astype(np.float64),
                    launch_acceleration.astype(np.float64),
                    float(v0), float(dt), k_drag, float(mu)
                )
            else:
                a_prev = 0.0
                for i in range(1, n):
                    friction_acc_mag = mu * normal_mag[i]
                    drag_acc_mag = k_drag * v_estimate[i-1]**2
                    sign_motion = 1.0 if v_estimate[i-1] >= 0 else -1.0
                    # Add launch acceleration if in launch section
                    a_gravity = g_par_mag[i]
                    a_launch = launch_acceleration[i]
                    a_new = a_gravity + a_launch - sign_motion * (friction_acc_mag + drag_acc_mag)
                    
                    # Velocity-Verlet: half-step velocity update
                    v_half = v_estimate[i-1] + 0.5 * a_prev * dt
                    v_estimate[i] = v_half + 0.5 * a_new * dt
                    if v_estimate[i] < 0 and abs(v_estimate[i]) < 1e-9:
                        v_estimate[i] = 0.0
                    
                    a_tan[i] = a_new
                    a_prev = a_new
        else:
            # Legacy semi-implicit Euler (1st order, less accurate)
            if NUMBA_AVAILABLE:
                @njit(cache=True, fastmath=True)
                def _integrate_speed_euler(g_par_mag_arr, normal_mag_arr, launch_acc_arr, v0_val, dt_val, k_drag_val, mu_val):
                    N = g_par_mag_arr.shape[0]
                    v_out = np.zeros(N, dtype=np.float64)
                    a_out = np.zeros(N, dtype=np.float64)
                    v_out[0] = v0_val
                    a_out[0] = 0.0
                    for i in range(1, N):
                        friction_acc_mag = mu_val * normal_mag_arr[i]
                        drag_acc_mag = k_drag_val * v_out[i-1] * v_out[i-1]
                        sign_motion = 1.0 if v_out[i-1] >= 0.0 else -1.0
                        # Add launch acceleration if in launch section
                        a_gravity = g_par_mag_arr[i]
                        a_launch = launch_acc_arr[i]
                        a_out[i] = a_gravity + a_launch - sign_motion * (friction_acc_mag + drag_acc_mag)
                        v_out[i] = v_out[i-1] + a_out[i] * dt_val
                        if v_out[i] < 0.0 and abs(v_out[i]) < 1e-9:
                            v_out[i] = 0.0
                    return v_out, a_out

                v_estimate, a_tan = _integrate_speed_euler(
                    g_par_mag.astype(np.float64), 
                    normal_mag.astype(np.float64),
                    launch_acceleration.astype(np.float64),
                    float(v0), float(dt), k_drag, float(mu)
                )
            else:
                for i in range(1, n):
                    friction_acc_mag = mu * normal_mag[i]
                    drag_acc_mag = k_drag * v_estimate[i-1]**2
                    sign_motion = 1.0 if v_estimate[i-1] >= 0 else -1.0
                    # Add launch acceleration if in launch section
                    a_gravity = g_par_mag[i]
                    a_launch = launch_acceleration[i]
                    a_tan[i] = a_gravity + a_launch - sign_motion * (friction_acc_mag + drag_acc_mag)
                    v_estimate[i] = v_estimate[i-1] + a_tan[i] * dt
                    if v_estimate[i] < 0 and abs(v_estimate[i]) < 1e-9:
                        v_estimate[i] = 0.0

    # Curvature-based centripetal acceleration
    # Define ez (vertical unit vector) for use in both methods
    ez = np.array([0.0, 0.0, 1.0])
    
    # Choose curvature calculation method
    if curvature_method == 'circumcenter':
        # Geometric method (like Roller.py): more accurate for discrete points
        R, centers = _curvature_radius_circumcenter(points_smooth)
        # Use geometric normal direction: vector from point to circumcenter
        n_hat = np.zeros_like(e_tan)
        valid = np.isfinite(R) & (R >= 5.0) & np.all(np.isfinite(centers), axis=1)
        for i in np.where(valid)[0]:
            R_vec = points_smooth[i] - centers[i]
            R_norm = np.linalg.norm(R_vec)
            if R_norm > 1e-9:
                n_hat[i] = R_vec / R_norm
            else:
                # Fallback: use derivative of tangent
                if i > 0 and i < n-1:
                    d_el = e_tan[i+1] - e_tan[i-1]
                    d_norm = np.linalg.norm(d_el)
                    if d_norm > 1e-9:
                        n_hat[i] = d_el / d_norm
                    else:
                        # Final fallback: horizontal normal
                        lat = np.cross(ez, e_tan[i])
                        lat_norm = np.linalg.norm(lat)
                        n_hat[i] = lat / (lat_norm if lat_norm > 1e-9 else 1.0)
                else:
                    lat = np.cross(ez, e_tan[i])
                    lat_norm = np.linalg.norm(lat)
                    n_hat[i] = lat / (lat_norm if lat_norm > 1e-9 else 1.0)
    else:
        # Legacy finite difference method
        R = _curvature_radius_vectorized(points_smooth)
        # normal direction from derivative of tangent (vectorized)
        d_el = np.zeros_like(e_tan)
        d_el[1:] = e_tan[1:] - e_tan[:-1]
        d_norm = np.linalg.norm(d_el, axis=1, keepdims=True)
        d_safe = np.where(d_norm < 1e-9, 1.0, d_norm)
        n_hat = d_el / d_safe
        # fallback where derivative is near zero: use horizontal normal
        lat = np.cross(ez, e_tan)
        lat_norm = np.linalg.norm(lat, axis=1, keepdims=True)
        lat_safe = np.where(lat_norm < 1e-9, 1.0, lat_norm)
        lat_hat = lat / lat_safe
        # choose n_hat where valid else lat_hat
        use_lat = (d_norm.flatten() < 1e-9)
        n_hat[use_lat] = lat_hat[use_lat]
    
    # Compute centripetal acceleration
    a_eq = np.zeros((n, 3), dtype=float)
    # Use minimum radius of 5m to prevent numerical explosion
    if curvature_method == 'circumcenter':
        valid = np.isfinite(R) & (R >= 5.0) & (np.linalg.norm(n_hat, axis=1) > 1e-9)
    else:
        valid = np.isfinite(R) & (R >= 5.0)
    # Clamp centripetal acceleration magnitude to realistic max (~6g = 60 m/s²)
    # Most intense modern coasters max at 5-6g
    # Use v_estimate for centripetal calculation (will be refined below)
    a_cent_mag = np.clip(v_estimate[valid]**2 / R[valid], 0.0, 60.0)
    a_eq[valid] = -(a_cent_mag[:, None] * n_hat[valid])

    # Total inertial acceleration and specific force
    # Calculate tangential acceleration from speed changes to ensure it captures all acceleration
    # This is more reliable than relying solely on the integrated a_tan
    dv_dt = np.gradient(v_estimate, dt)  # Rate of speed change (m/s²)
    # Use speed derivative as the primary source of tangential acceleration
    # This ensures we capture all speed changes, including those from energy conservation
    a_tan_from_speed = dv_dt
    
    # The integrated a_tan includes: g_parallel + a_launch - friction - drag
    # The speed derivative dv_dt should equal this (since v = ∫a_tan dt)
    # But due to numerical integration, they might differ slightly
    # Use the speed derivative as it's more direct and captures all effects
    a_tan_effective = a_tan_from_speed
    
    # Total inertial acceleration: tangential + centripetal
    # a_tan already includes gravity's effect along the tangent (via g_par_mag in the integration)
    # a_eq is centripetal acceleration (points toward center of curvature)
    # Note: a_tot does NOT include gravity vector explicitly - gravity is already accounted for
    # in a_tan (along track) and will be handled by the normal force (perpendicular to track)
    a_tot = a_tan_effective.reshape(-1, 1) * e_tan + a_eq
    
    # Specific force (what accelerometer/passenger feels) = total acceleration - gravity
    # This is the relative acceleration that the passenger experiences
    # The standard formula: f_spec = a_tot - G_VEC
    # This gives the acceleration relative to free-fall (what an accelerometer measures)
    f_spec = a_tot - G_VEC

    # Calculate 3D velocity from position differences (finite differences)
    # This avoids accumulation errors - we take differences, not cumulative sum
    # Use finite differences on positions: v = dp/dt
    v_3d = np.zeros((n, 3), dtype=float)
    
    # Calculate position differences (displacements) between consecutive points
    # This is the "difference" - we're NOT using cumsum, we're taking differences
    dp = np.zeros((n, 3), dtype=float)
    dp[1:] = points_smooth[1:] - points_smooth[:-1]
    # For first point, use forward difference
    dp[0] = dp[1] if n > 1 else np.array([0.0, 0.0, 0.0])
    
    # Calculate distance traveled between points
    ds = np.linalg.norm(dp, axis=1)
    
    # Calculate velocity from position differences
    # Use v_estimate to determine realistic time steps, then calculate v = dp/dt
    # This ensures velocity is physically realistic and doesn't require infinite energy
    dt_actual = np.zeros(n, dtype=float)
    dt_actual[0] = dt  # Initial time step
    
    # Calculate time steps using v_estimate (which includes launch acceleration if applicable)
    for i in range(1, n):
        if ds[i] > 1e-6:  # Only if there's meaningful distance
            # Use estimated speed to determine time step
            # v_estimate already includes launch acceleration, so this gives realistic dt
            v_est = v_estimate[i] if i < len(v_estimate) else (v_estimate[i-1] if i > 0 else v0)
            v_est = max(v_est, 0.1)  # Minimum to avoid division by zero
            dt_actual[i] = ds[i] / v_est
            # Cap maximum time step to prevent unrealistic jumps
            dt_actual[i] = min(dt_actual[i], 2.0)  # Max 2 seconds per segment
        else:
            dt_actual[i] = dt  # Default time step
    
    # Calculate velocity as finite difference: v = dp / dt
    # This gives velocity from position changes using realistic time steps
    dt_safe = np.where(dt_actual > 1e-9, dt_actual, dt)
    v_3d = dp / dt_safe[:, None]
    
    # For first point, set initial velocity in tangent direction
    v_3d[0] = v0 * e_tan[0]
    
    # Calculate speed as Euclidean magnitude of 3D velocity vector
    v = np.linalg.norm(v_3d, axis=1)
    v = np.maximum(v, 0.0)
    
    # Use v_estimate as the source of truth for speed
    # v_estimate is calculated from physics (energy conservation + launch acceleration)
    # This ensures realistic speeds that don't require infinite energy
    # The velocity from position differences (v_3d) gives us direction, but speed comes from physics
    v = v_estimate.copy()  # Use physics-based speed estimate (includes proper launch acceleration)
    
    # Apply light smoothing to speed to match track visualization smoothness
    # This removes sharp edges that come from energy conservation formula (sqrt amplifies small height changes)
    # Use a small sigma to preserve physics while matching visual smoothness
    from scipy.ndimage import gaussian_filter1d
    v_smooth = gaussian_filter1d(v, sigma=1.0, mode='nearest')
    # Preserve initial and final values to maintain physics constraints
    v_smooth[0] = v0
    # Blend: use smoothed version but ensure it doesn't violate energy conservation too much
    # Only smooth if the difference is small (preserve large changes from physics)
    v_diff = np.abs(v_smooth - v)
    v = np.where(v_diff < 0.5, v_smooth, v)  # Use smoothed if difference < 0.5 m/s, else keep original
    
    # Recalculate v_3d to have the correct magnitude (from smoothed v) while preserving direction
    # This ensures speed matches physics while direction comes from track geometry
    v_3d_norm = np.linalg.norm(v_3d, axis=1)
    v_3d_norm = np.where(v_3d_norm > 1e-9, v_3d_norm, 1.0)
    scale = v / v_3d_norm
    v_3d = v_3d * scale[:, None]
    
    # Ensure v_3d[0] matches initial velocity
    v_3d[0] = v0 * e_tan[0]
    v[0] = v0

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
        'v': v,  # Speed as Euclidean magnitude of 3D velocity
        'v_3d': v_3d,  # 3D velocity vector (x, y, z components)
        'a_tan': a_tan_effective,  # Use speed-derived tangential acceleration
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
