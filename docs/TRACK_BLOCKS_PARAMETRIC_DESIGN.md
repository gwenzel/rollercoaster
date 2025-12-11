# Track Blocks Parametric Design Documentation

This document describes the parametric design of each building block available in the roller coaster builder. Each block is defined by a set of configurable parameters that control its geometric properties.

## Coordinate System

All blocks use a consistent coordinate system:
- **X-axis**: Forward distance along the track (horizontal, in meters)
- **Y-axis**: Vertical height (elevation, in meters)
- **Z-axis**: Lateral displacement (banking, in meters)

Blocks are designed to start at the origin (0, 0, 0) and end at a relative position. When blocks are concatenated, their outputs are translated and rotated to form continuous track profiles.

---

## 1. ⛰️ Lift Hill

**Purpose**: Chain lift mechanism that raises the train to initial height.

### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `length` | float | 100 m | 30-150 m | Total horizontal length of lift section |
| `height` | float | 80 m | 20-120 m | Target height to reach |

### Geometric Properties

The lift hill consists of three segments:

1. **Entry (15% of length)**: Gentle acceleration curve using quadratic easing
   - Starts horizontal (y = 0)
   - Reaches 10% of target height
   - Smooth transition to main section

2. **Main (60% of length)**: Linear ascent
   - Constant slope from 10% to 85% of target height
   - Dense point sampling (1 point per 2 meters)

3. **Exit (25% of length)**: Deceleration curve
   - Cubic easing to final height
   - Smooth transition to next block

### Design Constraints

- Maximum realistic height: `min(height, length * 0.8)`
- No banking (z = 0 throughout)
- Ensures smooth curvature transitions at boundaries

### Usage Notes

- Typically placed at the start of a coaster
- Longer lengths allow for gentler slopes (more comfortable)
- Shorter lengths create steeper climbs (more compact designs)

---

## 2. ⬇️ Vertical Drop

**Purpose**: Steep initial drop that converts potential energy to kinetic energy.

### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `height` | float | 40 m | 10-100 m | Vertical drop distance |
| `steepness` | float | 0.9 | 0.3-1.0 | Drop angle factor (1.0 = vertical) |

### Geometric Properties

- **Single smooth curve**: Uses cosine-based easing for natural curvature
- **Horizontal distance**: Computed as `drop_height * 1.73 / max(steepness, 0.3)`
  - Ensures minimum horizontal distance for safety
  - Steeper drops (higher steepness) = shorter horizontal distance
- **Curvature**: Raised cosine function `(1 - cos(π·progress)) / 2`
  - Starts and ends horizontal (smooth transitions)
  - Maximum steepness at midpoint

### Design Constraints

- Minimum horizontal distance: `height * 1.73` (for 60° minimum angle)
- Actual drop height: Limited by `current_height` if provided
- No banking (z = 0)

### Usage Notes

- Higher `steepness` values create more intense drops
- Lower `steepness` values create gentler, more comfortable descents
- Typically follows a lift hill or launch section

---

## 3. 🔄 Loop (Clothoid)

**Purpose**: Vertical loop element using clothoid transitions for smooth G-forces.

### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `diameter` | float | 30 m | 15-50 m | Diameter of the circular loop section |

### Geometric Properties

The loop consists of three segments:

1. **Entry Transition (Clothoid)**: 
   - Length: `diameter * 1.6`
   - Clothoid parameter: `a² = r * transition_length`
   - Smoothly transitions from horizontal to vertical
   - 50 points for smooth curvature

2. **Circular Loop**:
   - Radius: `diameter / 2`
   - Full 360° rotation (minus transition angles)
   - 80 points for smooth circular arc
   - Centered based on entry transition endpoint

3. **Exit Transition (Clothoid)**:
   - Mirrored entry transition
   - Smoothly transitions from vertical back to horizontal
   - Ensures y-coordinate returns to starting level

### Design Constraints

- Clothoid transitions ensure constant rate of change of curvature
- Prevents sudden G-force spikes
- No lateral banking (z = 0)

### Usage Notes

- Larger diameters create gentler loops (lower G-forces)
- Smaller diameters create more intense loops (higher G-forces)
- Clothoid transitions are essential for rider comfort
- Typically requires significant speed to complete (check physics simulation)

---

## 4. 🎈 Airtime Hill

**Purpose**: Hill that creates negative G-forces (airtime/weightlessness).

### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `length` | float | 40 m | 20-100 m | Horizontal length of hill |
| `height` | float | 15 m | 5-30 m | Peak height of hill |

### Geometric Properties

- **Single continuous function**: Combines sine and polynomial for smooth curvature
- **Profile**: `height * sin²(π·progress) * (1 - 0.4·|2·progress - 1|³)`
  - Smooth rise and fall using sine squared
  - Slight plateau at peak using polynomial shaping
  - Ensures continuous derivatives (no curvature spikes)
- **60 points** for smooth representation

### Design Constraints

- Maximum safe height: `length * 0.5` (prevents excessive steepness)
- No banking (z = 0)
- Smooth curvature throughout (no discontinuities)

### Usage Notes

- Creates "floater" airtime (mild negative G) or "ejector" airtime (strong negative G)
- Longer lengths with same height = gentler airtime
- Shorter lengths with same height = more intense airtime
- Typically placed after high-speed sections

---

## 5. 🌀 Spiral (Helix)

**Purpose**: Horizontal spiral/helix with vertical undulation and banking.

### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `diameter` | float | 25 m | 15-40 m | Effective diameter of spiral |
| `turns` | float | 1.5 | 0.5-3.0 | Number of complete rotations |

### Geometric Properties

- **Total length**: `diameter * turns * π`
- **Vertical undulation**: 
  - Amplitude: 4.0 meters
  - Cycles: `turns * 2` (up/down per turn)
  - Smooth envelope using `sin(π·progress)` for ease in/out
- **Lateral banking**:
  - Amplitude: `diameter * 0.3`
  - Creates corkscrew motion
  - Banking follows cosine of spiral angle
- **Point density**: Scales with number of turns (50 points per turn)

### Design Constraints

- Vertical motion is relative (starts and ends at y = 0)
- Banking creates realistic corkscrew forces
- Smooth transitions at entry/exit

### Usage Notes

- More turns = longer, more complex spirals
- Larger diameter = gentler spiral (lower lateral G-forces)
- Creates both vertical and lateral forces
- Typically used for variety and intensity

---

## 6. ↪️ Banked Turn

**Purpose**: Horizontal turn with banking to reduce lateral G-forces.

### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `radius` | float | 30 m | 15-50 m | Turn radius (curvature) |
| `angle` | float | 90° | 30-180° | Total turn angle in degrees |
| `bank_deg` | float | 25° | 10-45° | Banking angle (optional) |

### Geometric Properties

- **Horizontal arc**: Circular arc with specified radius and angle
- **Banking easing**:
  - Entry (0-20%): Cubic ease-in to target banking
  - Middle (20-80%): Constant banking
  - Exit (80-100%): Cubic ease-out from banking
- **Vertical modulation**: 
  - Slight elevation change using `sin(2·θ)`
  - Eased at entry/exit to ensure y[0] = y[-1] = 0
- **Lateral banking**:
  - Banking height: `radius * sin(bank_deg)`
  - Negative z for inward banking (realistic physics)
  - Smoothly transitions in/out

### Design Constraints

- Banking angle typically 20-30° for comfort
- Larger radius = gentler turn (lower lateral G-forces)
- Smaller radius = tighter turn (higher lateral G-forces)
- Banking reduces perceived lateral forces

### Usage Notes

- Essential for high-speed turns
- Insufficient banking causes uncomfortable lateral G-forces
- Excessive banking can cause negative vertical G-forces
- Typically used after high-speed sections

---

## 7. 🐰 Bunny Hop

**Purpose**: Quick airtime bump for brief moments of weightlessness.

### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `length` | float | 20 m | 10-40 m | Horizontal length of bump |
| `height` | float | 8 m | 3-15 m | Peak height of bump |

### Geometric Properties

- **Smooth bump profile**: 
  - Base shape: `sin(π·progress)`
  - Easing: Cubic smoothstep `progress²(3 - 2·progress)`
  - Combined: `height * sin(π·progress) * smoothstep(progress)`
- **40 points** for smooth representation
- Gentle entry/exit transitions

### Design Constraints

- Maximum safe height: `length * 0.3` (prevents excessive steepness)
- No banking (z = 0)
- Smooth curvature (no spikes)

### Usage Notes

- Creates quick "pop" of airtime
- Shorter lengths = more intense airtime
- Typically used for variety between larger elements
- Good for maintaining speed while adding excitement

---

## 8. 🚀 Launch Section

**Purpose**: Magnetic launch (LSM/LIM) that accelerates the train.

### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `length` | float | 40 m | 20-80 m | Length of launch track |
| `speed_boost` | float | 20 m/s | 10-40 m/s | Target speed increase (72-144 km/h) |

### Geometric Properties

- **Completely flat**: y = 0 throughout (straight acceleration zone)
- **No banking**: z = 0
- **Dense sampling**: 1 point per 1.5 meters (minimum 30 points)
  - Ensures accurate launch detection in physics engine

### Design Constraints

- Must be flat for linear acceleration
- Speed boost is applied by physics engine (not geometric)
- Typically placed at start (replaces lift hill) or mid-ride

### Usage Notes

- `speed_boost` is the target velocity increase, not absolute speed
- Longer lengths allow for gentler acceleration (more comfortable)
- Shorter lengths require more intense acceleration
- Can be used multiple times in a single ride
- Physics engine detects launch blocks and applies acceleration

---

## 9. ➡️ Flat Section

**Purpose**: Straight section for transitions or maintaining speed.

### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `length` | float | 30 m | 10-100 m | Horizontal length |
| `slope` | float | -0.02 | -0.1 to 0.1 | Slight slope (negative = downhill) |

### Geometric Properties

- **Linear profile**: `y = slope * x`
- **20 points** for representation
- No banking (z = 0)

### Design Constraints

- Slope is typically small (near zero)
- Negative slope = slight downhill (maintains speed)
- Positive slope = slight uphill (slight deceleration)

### Usage Notes

- Used for transitions between elements
- Slight negative slope helps maintain speed
- Essential for smooth track connections
- Can be used to create "floating" sections over water/terrain

---

## 10. 🛑 Brake Run

**Purpose**: Final braking section to safely stop the train.

### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `length` | float | 30 m | 20-80 m | Length of brake section |

### Geometric Properties

- **Completely flat**: y = 0 throughout
- **No banking**: z = 0
- **Dense sampling**: 1 point per 1.5 meters (minimum 20 points)
- Straight and level for smooth, comfortable braking

### Design Constraints

- Must be flat for effective braking
- Typically placed at end of ride
- Longer lengths allow for gentler deceleration

### Usage Notes

- Essential for safe ride completion
- Should be long enough to stop from maximum speed
- Physics engine applies braking forces
- Can be followed by station/loading area

---

## Block Assembly and Smoothing

When blocks are assembled sequentially:

1. **Translation**: Each block's output is translated to start where the previous block ended
2. **Rotation**: Blocks are rotated to match the exit angle of the previous block
3. **Smoothing**: Automatic smoothing is applied at block boundaries to ensure:
   - Continuous position (no gaps)
   - Continuous tangent (smooth direction changes)
   - Continuous curvature (smooth G-force transitions)

### Smoothing Algorithm

- Uses spline interpolation (`scipy.interpolate.splprep`)
- Enforces C² continuity (continuous second derivatives)
- Prevents curvature spikes at block boundaries
- Maintains geometric accuracy while ensuring physical realism

---

## Parameter Recommendations

### For Beginner Designs
- **Lift Hill**: length=100-120m, height=60-80m
- **Drop**: height=40-60m, steepness=0.7-0.9
- **Loop**: diameter=30-35m
- **Airtime Hill**: length=50-70m, height=15-20m
- **Banked Turn**: radius=30-40m, angle=60-90°, bank_deg=20-25°

### For Intense Designs
- **Lift Hill**: length=80-100m, height=80-100m
- **Drop**: height=60-100m, steepness=0.9-1.0
- **Loop**: diameter=20-30m
- **Airtime Hill**: length=30-50m, height=20-30m
- **Banked Turn**: radius=20-30m, angle=90-120°, bank_deg=25-35°

### For Family-Friendly Designs
- **Lift Hill**: length=100-150m, height=40-60m
- **Drop**: height=30-50m, steepness=0.5-0.7
- **Loop**: diameter=35-45m (or omit)
- **Airtime Hill**: length=60-80m, height=10-15m
- **Banked Turn**: radius=40-50m, angle=45-60°, bank_deg=15-20°

---

## Physics Considerations

### Speed Requirements

- **Loops**: Require minimum speed to complete (typically > 15 m/s at top)
- **Airtime Hills**: Work best at high speeds (20-30 m/s)
- **Banked Turns**: Higher speeds require more banking

### G-Force Limits

- **Vertical positive**: > 5g is dangerous, > 3g is intense
- **Vertical negative**: < -1.5g is ejector airtime, < -1.0g is flojector
- **Lateral**: > 2g is uncomfortable, > 5g is dangerous

### Energy Conservation

- Train starts with potential energy from lift hill or kinetic energy from launch
- Energy is lost to friction (95% efficiency) and air resistance
- Each element converts energy between potential and kinetic forms

---

## Best Practices

1. **Start with lift hill or launch**: Provides initial energy
2. **Follow drop with high-speed elements**: Loops, turns, airtime hills
3. **Use flat sections for transitions**: Smooth connections between elements
4. **End with brake run**: Safe deceleration to stop
5. **Check G-forces**: Use the physics simulation to verify safety
6. **Balance intensity**: Mix intense and gentle elements for variety
7. **Consider rider comfort**: Avoid excessive G-forces or rapid transitions

---

## Technical Implementation

All block functions follow this signature:
```python
def block_profile(param1=default1, param2=default2, **kwargs):
    """
    Generate track profile for block.
    
    Returns:
        x, y, z: numpy arrays of coordinates
        - x: forward distance (meters)
        - y: vertical height (meters, relative)
        - z: lateral displacement (meters, for banking)
    """
    # Implementation...
    return x, y, z
```

Blocks are combined in `app_builder.py` using the `TrackBlock` class, which:
- Wraps profile functions with metadata (name, description, icon)
- Provides parameter validation
- Handles block sequencing and smoothing

---

## References

- Clothoid (Euler spiral) mathematics for loop transitions
- Frenet-Serret frame for coordinate transformations
- Energy conservation for physics simulation
- Spline interpolation for track smoothing

