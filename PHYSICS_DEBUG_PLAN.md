# Physics Model Debugging Plan

## Overview
This plan validates both physics engines (Simple vs Advanced) and diagnoses any issues with the default geometry.

## Available Scripts

### 1. `debug_physics_models.py` - Comprehensive Model Validation
**Purpose:** Compare both physics models side-by-side with detailed metrics

**What it checks:**
- ✅ Speed profiles (should match energy conservation laws)
- ✅ G-force ranges (lateral, vertical, longitudinal)
- ✅ Safety thresholds (human tolerance limits)
- ✅ Continuity (detects sudden spikes > 2g)
- ✅ Model agreement (correlation between Simple vs Advanced)
- ✅ Lift hill detection (Advanced model only)

**Expected results with default geometry:**
- Max speed after 30m drop: ~79 km/h (22 m/s)
- Lift speed: 10.8 km/h (3 m/s)
- Lateral g-forces: < 2.0g
- Vertical g-forces: -2.0g to +5.0g
- Longitudinal g-forces: < 1.5g
- Model correlation: > 0.90

**Run:**
```bash
python scripts/debug_physics_models.py
```

**Output:**
- Console: Detailed metrics and safety checks
- Plot: `physics_debug_comparison.png` (6-panel comparison)

---

### 2. `debug_curvature.py` - Curvature Spike Analysis
**Purpose:** Identify where and why curvature spikes occur

**What it checks:**
- ✅ Curvature magnitude and distribution
- ✅ Radius of curvature (should be > 5m for realistic coasters)
- ✅ Spike locations (correlate with block boundaries?)
- ✅ Tangent vector continuity (C1 continuity at joints)
- ✅ Effect of smoothing on spikes

**Expected results:**
- Typical helix radius: ~15m
- Tight loop radius: ~5-10m
- Curvature spikes near block boundaries = Hermite interpolation issue
- Tangent jumps < 0.1 (smooth transitions)

**Run:**
```bash
python scripts/debug_curvature.py
```

**Output:**
- Console: Spike locations and block correlations
- Plot: `curvature_analysis.png` (4-panel diagnostic)

---

## Debugging Workflow

### Phase 1: Basic Validation
1. **Run `debug_physics_models.py`**
2. Check console output for:
   - [ ] Speed profile reasonable? (10.8 km/h at start, 79 km/h after 30m drop)
   - [ ] G-forces within safety limits?
   - [ ] Continuity checks pass? (no spikes > 2g)
   - [ ] Model correlation > 0.90?
3. Check `physics_debug_comparison.png`:
   - [ ] Speed curves smooth and monotonic on drops?
   - [ ] G-force curves continuous (no sharp spikes)?
   - [ ] Simple vs Advanced reasonably similar?

**If all checks pass → Physics models working correctly ✓**

**If checks fail → Proceed to Phase 2**

---

### Phase 2: Curvature Investigation
1. **Run `debug_curvature.py`**
2. Check console output for:
   - [ ] Are spikes marked "⚠️ NEAR JOINT"? (block boundary issue)
   - [ ] Is raw curvature >> smoothed? (needs more smoothing)
   - [ ] Tangent vector jumps > 0.1? (C1 discontinuity)
   - [ ] Radius < 5m? (unrealistic geometry)
3. Check `curvature_analysis.png`:
   - [ ] Panel 2: Do curvature spikes align with red dashed lines (boundaries)?
   - [ ] Panel 3: Any radius < 5m (tight curves)?
   - [ ] Panel 4: Tangent jumps near boundaries?

**Diagnosis:**
- **Spikes at boundaries** → Hermite interpolation not preserving C2 continuity
- **High tangent jumps** → Block connections have orientation mismatches
- **Radius < 5m** → Block geometry too tight (increase radius params)
- **Random spikes** → Numerical noise (increase smoothing sigma)

---

### Phase 3: Model-Specific Debugging

#### Simple Model Issues
**Symptoms:**
- Unrealistic high g-forces (> 10g)
- Extreme curvature spikes
- Poor correlation with Advanced

**Fixes to try:**
```python
# In accelerometer_transform.py compute_rider_accelerations()

# 1. Increase smoothing
tangent_x_smooth = gaussian_filter1d(tangent_x, sigma=2.5, mode='nearest')  # was 1.5
curvature = gaussian_filter1d(curvature, sigma=1.5, mode='nearest')  # was 1.0

# 2. Add radius clamping (like Advanced model)
radius = 1.0 / curvature
radius = np.clip(radius, 5.0, 1000.0)  # 5m to 1000m
curvature = 1.0 / radius

# 3. Clamp centripetal acceleration
a_centripetal = np.minimum(v**2 * curvature, 60.0)  # Cap at 6g
```

#### Advanced Model Issues
**Symptoms:**
- Lift hill not detected
- Speed too low throughout
- Poor energy conservation

**Fixes to try:**
```python
# In acceleration.py compute_acc_profile()

# 1. Adjust lift hill detection threshold
for i in range(1, min(n, int(n * 0.5))):  # Check first 50% (was 40%)

# 2. Adjust energy efficiency
energy_efficiency = 0.85  # 85% (was 80%)

# 3. Check coordinate mapping in app_builder.py
points = np.column_stack([x, z, y])  # Ensure correct (forward, lateral, vertical)
```

---

## Common Issues and Solutions

### Issue 1: "Speed is way too high!"
**Check:**
- Height profile correct? (track_df['y'] is vertical coordinate)
- Energy conservation formula: `v² = v0² + 2g∆h*eff`
- Lift hill detected? (should have constant v0 = 3 m/s at start)

**Fix:**
- Verify track geometry: `print(track_df[['x', 'y', 'z']].head(20))`
- Check coordinate mapping in Simple vs Advanced

### Issue 2: "G-forces have sharp spikes"
**Check:**
- Curvature spikes at block boundaries?
- Tangent vector discontinuities?
- Insufficient smoothing?

**Fix:**
- Increase smoothing sigma values
- Add radius clamping
- Improve Hermite interpolation at block joints

### Issue 3: "Models disagree significantly"
**Check:**
- Coordinate system mismatch?
- Different speed calculations?
- Different curvature methods?

**Fix:**
- Verify both use same coordinate convention
- Check energy conservation formula consistency
- Ensure same smoothing parameters

### Issue 4: "Airtime hill causes huge spikes"
**Check:**
- Recent smoothing fix applied? (single sine function)
- Curvature at hill peak?

**Fix:**
- Already fixed in `track_blocks.py` (lines 108-126)
- If still seeing issues, increase airtime hill radius parameter

---

## Physics Validation Checklist

### Speed Physics
- [ ] Lift hill: 3 m/s (10.8 km/h) constant
- [ ] After 30m drop: 22 m/s (79 km/h) ± 10%
- [ ] Energy formula: v² = v0² + 2g(h_lift - h) * 0.80
- [ ] No negative speeds
- [ ] Smooth acceleration (no sudden jumps)

### G-Force Limits
- [ ] Lateral: -2.0 to +2.0 g (comfort limit)
- [ ] Vertical positive: < 5.0 g (structural/comfort limit)
- [ ] Vertical negative: > -2.0 g (airtime limit)
- [ ] Longitudinal: -1.5 to +1.5 g (braking/acceleration limit)
- [ ] Peak-to-peak change: < 2g per timestep (0.02s)

### Curvature Realism
- [ ] Minimum radius: > 5m (tight loop)
- [ ] Typical curves: 10-20m radius
- [ ] Gentle sections: > 100m radius
- [ ] No infinite curvature (κ clamped)
- [ ] Smooth curvature profile (no spikes)

### Model Agreement
- [ ] Correlation > 0.90 for all axes
- [ ] Mean absolute error < 0.5g
- [ ] Systematic bias < 0.2g
- [ ] Peak values within 20% of each other

---

## Next Steps After Debugging

1. **If both models validated:**
   - Consider removing physics toggle (redundant)
   - Use Simple model (faster, simpler)
   - Document which model is canonical

2. **If only Advanced validated:**
   - Make Advanced the default
   - Fix or remove Simple model
   - Update UI to reflect single model

3. **If only Simple validated:**
   - Make Simple the default
   - Fix or remove Advanced model
   - Remove lift hill detection

4. **If neither validated:**
   - Check track geometry generation
   - Verify coordinate systems
   - Review energy conservation math
   - Consider external validation data

---

## Files Modified by This Plan

### Created:
- `scripts/debug_physics_models.py` (comprehensive validation)
- `scripts/debug_curvature.py` (curvature spike analysis)
- `PHYSICS_DEBUG_PLAN.md` (this file)

### May need fixing based on results:
- `utils/accelerometer_transform.py` (Simple model)
- `utils/acceleration.py` (Advanced model)
- `utils/track_blocks.py` (geometry generation)
- `app_builder.py` (coordinate mapping, model selection)

### Diagnostic outputs:
- `physics_debug_comparison.png` (model comparison plots)
- `curvature_analysis.png` (curvature diagnostic plots)

---

## Contact/Notes
- Both scripts use default geometry for consistency
- All safety thresholds from human factors research
- Smoothing parameters tuned to balance noise vs peak preservation
- Coordinate conventions: x=forward, y=vertical, z=lateral (track)
