# Comparison: `compute_acc_profile` vs Roller.py Script

## Executive Summary

Both implementations compute roller coaster accelerations, but use **fundamentally different approaches**:

- **Roller.py**: Time-stepping simulation with Velocity-Verlet integration, projects positions onto track segments
- **compute_acc_profile**: Space-based (per track point) with semi-implicit Euler or energy conservation

## Key Differences

### 1. **Integration Method** ⚠️ **CRITICAL DIFFERENCE**

**Roller.py (Lines 270-275):**
```python
# Velocity-Verlet (2nd order, symplectic, energy-conserving)
v_half = v[i-1] + 0.5 * a_scalar_prev * dt
ds[i] = v_half * dt * e_dp[i]
s[i] = s[i-1] + ds[i]
a_scalar_new = a_tangential
v[i] = v_half + 0.5 * a_scalar_new * dt
```

**compute_acc_profile (Lines 191, 203):**
```python
# Semi-implicit Euler (1st order, less accurate)
a_out[i] = g_par_mag_arr[i] - sign_motion * (friction_acc_mag + drag_acc_mag)
v_out[i] = v_out[i-1] + a_out[i] * dt_val  # Uses current acceleration, but only 1st order
```

**Impact**: Velocity-Verlet is more accurate for long simulations, preserves energy better, and is more stable.

### 2. **Curvature Calculation** ⚠️ **SIGNIFICANT DIFFERENCE**

**Roller.py (Lines 184-200, 282-296):**
- Uses **circumcenter radius** from 3 consecutive track points
- Geometric method: finds circle passing through 3 points
- Computes radius vector from integrated position to circumcenter
- Uses this vector as normal direction for centripetal acceleration

**compute_acc_profile (Lines 47-92, 207-232):**
- Uses **finite differences on tangent vectors**
- Computes curvature as `κ = |dT/ds|` where T is unit tangent
- Uses derivative of tangent as normal direction
- More sensitive to discretization noise (mitigated by Gaussian smoothing)

**Impact**: Circumcenter method is more geometrically accurate for discrete track points, especially in tight turns.

### 3. **Normal Direction for Centripetal Acceleration**

**Roller.py (Line 292):**
```python
R_vec = s[i] - center  # Vector from integrated position to circumcenter
e_R_norm, e_R = safe_norm(R_vec)
a_eq[i] = - (v[i]**2 / R_val) * e_R
```

**compute_acc_profile (Lines 211-232):**
```python
d_el = e_tan[1:] - e_tan[:-1]  # Derivative of tangent
n_hat = d_el / d_norm  # Normal from tangent derivative
a_eq[valid] = -(a_cent_mag[:, None] * n_hat[valid])
```

**Impact**: Roller's method uses actual geometric center, compute_acc_profile uses tangent derivative (less accurate for non-circular arcs).

### 4. **Track Following Paradigm**

**Roller.py:**
- **Time-based**: Integrates position over time
- Projects integrated position `s[i]` onto nearest track segment
- Uses segment direction as local tangent
- Can handle variable spacing between track points

**compute_acc_profile:**
- **Space-based**: Works directly on track points
- Assumes uniform or known spacing
- Uses precomputed tangents from track points
- No position projection needed

**Impact**: Roller can handle variable track point density better, but compute_acc_profile is simpler and faster.

### 5. **Gravity Decomposition**

**Both are similar:**
- Both compute `g_par_mag = g · e_tan` (gravity component along tangent)
- Both compute normal component as `g_N = g - g_par_vec`
- Both use normal magnitude for friction calculation

**Minor difference**: Roller stores `g_a[i]` vector, compute_acc_profile only uses magnitude.

### 6. **Friction and Drag**

**Both are identical:**
- Friction: `mu * normal_magnitude`
- Drag: `0.5 * rho * Cd * A / mass * v²`
- Both apply sign based on velocity direction

### 7. **Speed Calculation**

**Roller.py:**
- Always uses Velocity-Verlet integration
- No energy conservation mode

**compute_acc_profile:**
- Has two modes:
  - **Integration mode**: Semi-implicit Euler (default)
  - **Energy conservation mode**: `v = sqrt(v0² + 2g(h0 - h) * efficiency)`
- Energy mode ignores friction/drag during speed calculation (only applies efficiency factor)

## What compute_acc_profile Does Better

1. ✅ **Vectorized operations**: Much faster for large track point arrays
2. ✅ **Gaussian smoothing**: Reduces discretization noise in curvature calculations
3. ✅ **Energy conservation option**: Useful for quick estimates without full physics
4. ✅ **Numba JIT support**: Optional acceleration for integration loop
5. ✅ **Cleaner API**: Returns dictionary with all computed quantities

## What Roller.py Does Better

1. ✅ **Velocity-Verlet integration**: More accurate and stable
2. ✅ **Circumcenter curvature**: More geometrically accurate
3. ✅ **Track projection**: Handles variable point spacing better
4. ✅ **Time-based output**: Natural for time-series analysis
5. ✅ **Geometric normal direction**: Uses actual radius vector from center

## Improvement Plan

### Priority 1: High Impact, Low Risk

#### 1.1 Replace Semi-Implicit Euler with Velocity-Verlet
**Location**: `utils/acceleration.py`, lines 178-205

**Current**:
```python
a_out[i] = g_par_mag_arr[i] - sign_motion * (friction_acc_mag + drag_acc_mag)
v_out[i] = v_out[i-1] + a_out[i] * dt_val
```

**Proposed**:
```python
# Velocity-Verlet: half-step velocity update
v_half = v_out[i-1] + 0.5 * a_out[i-1] * dt_val
# Update acceleration at half-step position (approximate)
a_out[i] = g_par_mag_arr[i] - sign_motion * (friction_acc_mag + drag_acc_mag)
v_out[i] = v_half + 0.5 * a_out[i] * dt_val
```

**Benefits**: Better accuracy, energy conservation, stability
**Risk**: Low - well-tested algorithm
**Effort**: ~30 minutes

#### 1.2 Add Circumcenter Curvature Option
**Location**: `utils/acceleration.py`, add new function `_curvature_radius_circumcenter`

**Proposed**:
```python
def _curvature_radius_circumcenter(points: np.ndarray) -> np.ndarray:
    """Compute curvature radii using circumcenter method (like Roller.py).
    More geometrically accurate for discrete track points."""
    n = points.shape[0]
    R = np.full(n, 1000.0, dtype=float)
    
    for i in range(1, n-1):
        idx1 = max(0, i-1)
        idx2 = i
        idx3 = min(n-1, i+1)
        R_val, center = _circumcenter_radius_3pt(
            points[idx1], points[idx2], points[idx3]
        )
        if np.isfinite(R_val) and R_val > 1e-6:
            R[i] = R_val
    
    return R
```

**Benefits**: More accurate curvature, especially in tight turns
**Risk**: Medium - need to handle edge cases
**Effort**: ~2 hours

#### 1.3 Use Geometric Normal Direction for Centripetal Acceleration
**Location**: `utils/acceleration.py`, lines 207-232

**Current**: Uses derivative of tangent as normal
**Proposed**: When using circumcenter method, use radius vector from point to center

**Benefits**: More physically accurate centripetal acceleration direction
**Risk**: Medium - need to ensure compatibility with vectorized approach
**Effort**: ~1 hour

### Priority 2: Medium Impact, Medium Risk

#### 2.1 Add Adaptive Time Stepping
**Current**: Fixed `dt` parameter
**Proposed**: Adaptive `dt` based on curvature and speed to maintain accuracy

**Benefits**: Better accuracy in high-curvature regions without over-sampling straight sections
**Risk**: Medium - complexity increase
**Effort**: ~4 hours

#### 2.2 Improve Energy Conservation Mode
**Location**: `utils/acceleration.py`, lines 150-171

**Current**: Simple efficiency factor, ignores local friction/drag
**Proposed**: 
- Integrate friction/drag losses along path
- Use arc length instead of height difference
- Account for banking effects

**Benefits**: More realistic energy-based speed calculation
**Risk**: Low - additive feature
**Effort**: ~3 hours

#### 2.3 Add Track Point Spacing Validation
**Location**: `utils/acceleration.py`, beginning of function

**Proposed**: Check for uniform spacing, warn if highly variable, suggest interpolation

**Benefits**: Prevents errors from poorly sampled tracks
**Risk**: Low - validation only
**Effort**: ~1 hour

### Priority 3: Lower Priority Enhancements

#### 3.1 Add Banking Angle Support
**Current**: Assumes track is always horizontal in lateral direction
**Proposed**: Add banking angle parameter, adjust normal force calculation

**Benefits**: More realistic G-forces in banked turns
**Risk**: Medium - requires track geometry data
**Effort**: ~6 hours

#### 3.2 Add Position Projection Mode (Optional)
**Proposed**: Optional mode that projects positions onto track (like Roller.py)

**Benefits**: Handles variable point spacing, more robust
**Risk**: High - major refactor
**Effort**: ~1-2 days

#### 3.3 Performance Optimizations
- Cache curvature calculations
- Optimize Gaussian smoothing (use FFT-based if available)
- Parallelize independent calculations

**Benefits**: Faster execution for large tracks
**Risk**: Low - optimization only
**Effort**: ~4 hours

## Recommended Implementation Order

1. **Week 1**: Priority 1 items (Velocity-Verlet, circumcenter option)
2. **Week 2**: Priority 2 items (adaptive dt, improved energy mode)
3. **Week 3+**: Priority 3 items (banking, optimizations)

## Testing Strategy

1. **Unit Tests**: Compare outputs with Roller.py on same track
2. **Validation**: Test on known tracks with measured G-forces
3. **Regression**: Ensure existing code using `compute_acc_profile` still works
4. **Performance**: Benchmark before/after improvements

## Compatibility Notes

- Current API should remain unchanged (backward compatible)
- New parameters should have defaults matching current behavior
- Energy conservation mode should remain available

