# Coordinate System Issues Found

## Summary

The debug analysis revealed **fundamental coordinate system mismatches** between the Simple and Advanced models, explaining the large differences in G-force calculations.

## Key Findings

### 1. Coordinate System Mismatch

**Simple Model (`compute_rider_accelerations`):**
- Uses **original track coordinates**: `x` = forward, `y` = vertical (height), `z` = lateral
- Gravity vector: `[0, -9.81, 0]` (Y-down convention)
- Vertical G-force: `-a_spec[:, 1]` (Y component, inverted)

**Advanced Model (`compute_acc_profile`):**
- Uses **remapped coordinates**: `x` = forward, `y` = lateral, `z` = vertical
- Remapping: `points = np.column_stack([x, z, y])` → `(forward, lateral, vertical)`
- Gravity vector: `[0, 0, -9.81]` (Z-down convention)
- Vertical G-force: `f_spec[:, 2]` (Z component)

### 2. Speed Calculations Match

Both models use identical energy conservation formulas:
- `v² = v0² + 2g(h_initial - h) * 0.95`
- Heights match correctly after coordinate remapping
- Only difference: Simple model adds 0.1 m/s minimum velocity (to avoid division by zero)

### 3. G-Force Differences Explained

The large differences in G-forces are **NOT due to physics errors**, but due to:

1. **Different coordinate systems**: Simple uses Y-up, Advanced uses Z-up
2. **Different gravity directions**: `[0, -g, 0]` vs `[0, 0, -g]`
3. **Different vertical definitions**: Y-component vs Z-component

**Example from debug output:**
- Simple Vertical: `[-10.00, 0.29] g`
- Advanced Vertical: `[-0.05, 6.76] g`
- These are measuring **different things** in different coordinate systems!

## Impact

- **Speed profiles**: Match (except for minimum velocity)
- **Lateral G-forces**: Should match (both use same lateral direction)
- **Vertical G-forces**: Cannot be directly compared (different coordinate systems)
- **Longitudinal G-forces**: May differ due to different acceleration calculation methods

## Recommendations

### Option 1: Transform Simple Model to Advanced Coordinate System (Recommended)

Transform the simple model's output to match the advanced model's coordinate system:

```python
# After computing simple model accelerations
# Transform from (x, y, z) to (x, z, y) coordinate system
a_spec_simple_transformed = np.column_stack([
    a_spec_simple[:, 0],  # X (forward) stays same
    a_spec_simple[:, 2],  # Z (lateral) becomes Y
    a_spec_simple[:, 1]   # Y (vertical) becomes Z
])

# Gravity also needs transformation
gravity_simple = np.array([0, -9.81, 0])  # Original
gravity_transformed = np.array([0, 0, -9.81])  # After transformation
```

### Option 2: Transform Advanced Model to Simple Coordinate System

Transform the advanced model's output to match the simple model's coordinate system:

```python
# After computing advanced model accelerations
# Transform from (x, y, z) back to (x, y, z) = (forward, vertical, lateral)
a_spec_advanced_transformed = np.column_stack([
    f_spec[:, 0],  # X (forward) stays same
    f_spec[:, 2],  # Z (vertical) becomes Y
    f_spec[:, 1]   # Y (lateral) becomes Z
])
```

### Option 3: Use Common Coordinate System (Best Long-term)

Modify one of the models to use a consistent coordinate system throughout. Recommended:
- Use **Z-up convention** (like advanced model) everywhere
- Standardize on: `x` = forward, `y` = lateral, `z` = vertical
- Update simple model to work in this coordinate system

## Files to Update

1. **`utils/accelerometer_transform.py`**: Update simple model to use Z-up convention
2. **`scripts/debug_physics_models.py`**: Add coordinate transformation for proper comparison
3. **Documentation**: Update to clarify coordinate system conventions

## Next Steps

1. ✅ Identified coordinate system mismatch
2. ⏳ Implement coordinate transformation in debug script
3. ⏳ Update simple model to use consistent coordinate system
4. ⏳ Re-run validation with corrected coordinate systems
5. ⏳ Verify G-forces match after transformation

