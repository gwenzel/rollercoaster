# Interactive AI-Powered Roller Coaster Design and Rating System

## Abstract

We present an interactive web-based application for designing roller coasters and predicting their rider experience ratings using machine learning. The system combines modular track design, real-time physics simulation, and a LightGBM-based rating prediction model trained on accelerometer data from 1,299 real roller coasters. Users can construct custom coaster designs using parametric building blocks, visualize G-forces in real-time, and receive instant AI-powered ratings that correlate with actual rider experiences.

## 1. Introduction

Roller coaster design traditionally requires extensive engineering expertise and iterative physical prototyping. This work introduces an accessible computational framework that enables both enthusiasts and engineers to design, simulate, and evaluate roller coaster concepts through an intuitive web interface. The system bridges the gap between geometric track design and rider experience prediction by leveraging machine learning models trained on real-world accelerometer data and user ratings.

## 2. System Architecture

The application is built as a multi-page Streamlit web application with three primary components: (1) an interactive track builder with modular design elements, (2) a physics simulation engine that computes rider-frame accelerations, and (3) a machine learning pipeline that predicts fun ratings from accelerometer features.

### 2.1 Track Design Interface

The track builder employs a modular architecture where users compose rides from parametric building blocks. Ten distinct block types are available: lift hills, vertical drops, clothoid loops, airtime hills, spirals (helices), bunny hops, banked turns, launch sections, flat sections, and brake runs. Each block accepts configurable parameters (e.g., height, length, radius, angle) that control its geometric properties. Blocks are assembled sequentially to form complete track profiles, with automatic smoothing applied at transitions to ensure physical continuity.

The interface provides two design workflows: (1) manual block-by-block construction with parameter adjustment, and (2) random template generation that creates complete designs with 6-10 blocks using randomized parameters within safe ranges. Users can modify, reorder, or remove blocks dynamically, with the system automatically recomputing physics and ratings upon changes.

### 2.2 Physics Simulation Engine

The physics engine transforms geometric track coordinates into rider-frame accelerometer data using differential geometry principles. The system employs the Frenet-Serret frame to establish a local coordinate system at each point along the track, with axes defined as: (1) **tangent** (longitudinal, forward direction), (2) **normal** (vertical, perpendicular to track), and (3) **binormal** (lateral, side-to-side).

Velocity is computed using conservation of energy with efficiency losses (95% efficiency factor) and accounts for launch sections that provide initial kinetic energy. The acceleration profile includes tangential acceleration (speed changes), centripetal acceleration (curvature-induced), and gravitational effects. The system projects these world-frame accelerations onto the rider's reference frame, subtracting gravity to yield specific forces—the accelerations that riders actually perceive.

The simulation accounts for realistic physics parameters including vehicle mass (500-1200 kg), air resistance (drag coefficient Cd = 0.08-0.1, frontal area A = 2.0-2.5 m²), and rolling friction (μ = 0.001). Energy conservation is used as the primary method, with fallback to force-based calculations when launch sections are present.

### 2.3 Machine Learning Pipeline

The rating prediction system uses a LightGBM gradient boosting model trained on 1,299 roller coasters with paired accelerometer data and user ratings. The model processes a 26-feature vector extracted from rider-frame acceleration time series:

**Dynamics Features (20):** Maximum positive/negative vertical G-forces, maximum lateral and longitudinal G-forces, variance measures, jerk (rate of change of acceleration), G-force range, skewness, intensity pacing metrics, force transition counts, peak density, rhythm scores, and vibration measures in all three axes.

**Airtime Features (3):** Total airtime duration (log-transformed), floater airtime proportion (-0.5g to 0.0g), and flojector airtime proportion (-1.0g to -0.5g). Ejector airtime (< -1.0g) is captured implicitly through maximum negative vertical G.

**Metadata Features (3):** Maximum height, peak speed (estimated from energy conservation), and total track length.

The model achieves a test R² of 0.73 and mean absolute error (MAE) of 0.24 stars on a 1-5 star rating scale. Predictions are clipped to the [1.0, 5.0] range to match the training data distribution. The model was trained on accelerometer recordings from RideForcesDB paired with ratings from Captain Coaster, ensuring predictions reflect actual rider experiences rather than geometric properties alone.

## 3. Data Sources and Training

The system integrates two primary data sources:

**RideForcesDB (RFDB):** A collection of 2,088 accelerometer recordings from 794 unique roller coasters across 152 amusement parks. Each recording contains 3-axis acceleration data (lateral, vertical, longitudinal) sampled at 50 Hz, collected using wearable sensors during actual rides. The data represents 1,299 coasters after mapping to rating sources.

**Captain Coaster:** A database of 1,768 roller coasters with detailed rating distributions, including average ratings, total review counts, and percentage breakdowns by star rating (0.5★ to 5.0★). The mapping between RFDB and Captain Coaster was established using fuzzy string matching with park hierarchy awareness, achieving 73.5% coverage (1,299 coasters with both accelerometer data and ratings).

The training pipeline processes accelerometer data through a feature engineering stage that computes the 26 features described above, with careful handling of edge cases (NaN values, extreme outliers) and normalization where appropriate. The LightGBM model uses no feature scaling, as gradient boosting handles feature ranges inherently.

## 4. User Interface and Visualization

The application provides real-time feedback through multiple visualization components:

**Track Profile:** A 2D elevation plot showing the track geometry with distance along the track (x-axis) and height (y-axis). The plot is interactive, allowing zoom and pan for detailed inspection.

**G-Force Analysis:** Three subplots displaying vertical, lateral, and longitudinal G-forces as functions of distance along the track. Safety zones are color-coded: orange for intense but acceptable ranges, red for potentially dangerous levels. Zero-reference lines help users identify positive vs. negative G-forces.

**Airtime Timeline:** A distance-based visualization showing airtime segments color-coded by intensity (floater, flojector, ejector), helping users understand where riders experience weightlessness.

**Particle Simulation (Debug Mode):** An animated visualization showing a particle traversing the track with acceleration vectors overlaid, useful for understanding force directions during inversions and complex maneuvers.

**Advanced Features Panel:** A collapsible section displaying computed ride dynamics including vibration metrics, jerk statistics, intensity pacing, and rhythm scores—the same features used by the ML model.

The interface also includes a leaderboard system where users can submit their designs with predicted ratings and safety scores. The leaderboard displays submissions ranked by combined score (fun rating + safety score) and includes a Pareto front visualization showing the trade-off between fun and safety.

## 5. Safety Analysis

Beyond fun rating prediction, the system performs continuous safety analysis. Safety scores are computed using penalty functions that penalize extreme G-forces more severely as they exceed thresholds:

- **Vertical G-forces:** Penalties begin at 2.0g, with increasingly severe penalties for values above 3.0g, 4.0g, 5.0g, and 7.0g. The penalty function is continuous and monotonic, ensuring smooth optimization landscapes.

- **Lateral G-forces:** Penalties for values exceeding ±2.0g, with severe penalties above ±5.0g.

- **Longitudinal G-forces:** Penalties for extreme acceleration/deceleration above ±3.0g.

The safety score starts at 5.0 and decreases with each violation, providing a clear metric for ride safety that complements the fun rating.

## 6. Technical Implementation

The system is implemented in Python using Streamlit for the web interface, NumPy and SciPy for numerical computations, Plotly for interactive visualizations, and LightGBM for machine learning inference. The physics engine uses vectorized operations for efficiency, processing entire track profiles in single passes. Coordinate transformations employ NumPy's einsum operations for efficient matrix-vector products.

The application supports both local development and cloud deployment, with debug features (particle simulation, curvature profiles, speed profiles) automatically hidden in deployment environments to reduce computational overhead and improve user experience.

## 7. Results and Applications

The system successfully bridges geometric design and rider experience prediction, enabling users to:

1. **Design Custom Coasters:** Create unique layouts using intuitive building blocks without requiring engineering expertise.

2. **Predict Rider Experience:** Receive instant AI-powered ratings that correlate with real-world rider feedback, validated on 1,299 actual coasters.

3. **Optimize Designs:** Iteratively refine designs using real-time feedback on both fun ratings and safety scores.

4. **Compare Designs:** Submit designs to a leaderboard and compare against other user creations and real-world coasters from the RFDB dataset.

5. **Educational Use:** Understand the relationship between track geometry, physics, and rider experience through interactive visualization.

The leaderboard currently contains 300+ entries including both user-submitted designs and real-world coasters from the RFDB dataset, providing a rich comparison space for design evaluation.

## 8. Limitations and Future Work

Current limitations include: (1) 2D track design (lateral banking is supported but limited), (2) simplified physics (no detailed vehicle dynamics, no wind effects), (3) static rating prediction (no personalization for different rider preferences), and (4) limited block types (no corkscrews, dive loops, or other complex inversions).

Future enhancements could include: 3D track design with full banking control, multi-objective optimization algorithms to automatically generate high-rated designs, personalized rating predictions based on user preferences, expanded block library with more inversion types, and integration with CAD software for export to engineering tools.

## 9. Conclusion

This work demonstrates that machine learning models trained on real-world accelerometer data can effectively predict rider experience ratings for roller coaster designs. The interactive system makes this capability accessible to non-experts through an intuitive web interface, enabling rapid prototyping and evaluation of coaster concepts. The combination of modular design, real-time physics simulation, and ML-based rating prediction provides a comprehensive tool for coaster design and analysis.

---

**Keywords:** Roller coaster design, machine learning, accelerometer data, physics simulation, interactive design tools, rider experience prediction

