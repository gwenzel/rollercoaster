# Inferring Coaster Geometry From POV Video

This document outlines a practical pipeline to reconstruct an approximate 2D/3D track path from a rider POV video. As a concrete example, we reference Steel Vengeance POV: https://www.youtube.com/watch?v=RTXTbzctl0c

Goals
- Recover an approximate track centerline (x,y,z) over distance/time
- Estimate local tangent, curvature, and grade
- Produce derived accelerations compatible with our app for visualization and model input

High-Level Pipeline
1. Data Ingestion
   - Download video (yt-dlp) and extract frames with timestamps
   - Optionally stabilize footage (video stabilization) to reduce head jitter
2. Visual Signals
   - Horizon line detection (estimate pitch/roll variations)
   - Optical flow along forward axis (estimate velocity proxy)
   - Vanishing point/homography (estimate heading changes)
   - Track edge or spine detection (template-based, classical + deep features)
3. IMU Estimation (Visual-Inertial-Approximation)
   - Approximate angular rates from frame-to-frame rotations (roll/pitch/yaw)
   - Estimate forward acceleration from optic expansion + frame rate
   - Fuse signals via a simple Kalman filter to produce a smooth pose sequence
4. Camera Pose → Path Integration
   - Assume camera mounted near rider head, forward-looking
   - Integrate heading + speed to generate a 2D path (x,y)
   - Use pitch to infer vertical profile (z) relative to ground/horizon
   - Constrain with physical priors (no impossible slopes/curvature)
5. Post-Processing & Smoothing
   - Enforce continuity (C1) and light curvature smoothing
   - Match typical RMC/GCI profiles to reduce drift (prior-informed regularization)
6. Outputs
   - `track_path.csv` with time, x,y,z
   - Derived `accel_df.csv` computed via `utils/acceleration.compute_acc_profile`
   - Plots: side view, curvature, g-force and timeline

Assumptions & Priors
- Camera FOV ~90°–120°; fixed relative to rider
- Constant or slowly varying playback speed; known frame rate
- No excessive shaking (use stabilization if needed)
- Ground plane roughly horizontal; horizon detection viable for pitch

Steel Vengeance Specifics
- Long ride with mixed elements (steep drops, airtime hills, inversions)
- Strong forward motion; good optic flow signal
- Wood-steel hybrid track with visible spine/rails aiding edge detection

Implementation Sketch
- Use OpenCV for frame extraction, optical flow (Farneback or RAFT), horizon estimation
- Use `cv2.RANSAC` with line detection on sky/ground segmentation for pitch
- Estimate yaw from vanishing point drift across frames
- Recover speed proxy from optic flow magnitude around central regions
- Integrate speeds + yaw over time to form (x,y), integrate pitch for z with gravity constraints
- Smooth with cubic splines; enforce physical constraints similar to `generate_track_from_blocks`

Quality & Validation
- Cross-check derived curvature against expected coaster stats
- Compare airtime segments where vertical g < 0 to ride moments in the video
- Sanity-check total track length against published values

Artifacts
- `data/pov/steel_vengeance/frames/*.jpg`
- `data/pov/steel_vengeance/track_path.csv`
- `data/pov/steel_vengeance/accel_df.csv`
- `data/pov/steel_vengeance/figures/*`

Next Steps
- Add a script `scripts/pov_geometry_infer.py` to run end-to-end
- Optional: lightweight UI in Streamlit to preview reconstruction
- Optional: Incorporate deep models (MiDaS for depth, RAFT for flow) for robustness

Dependencies
- OpenCV (`opencv-python`), NumPy, SciPy, pandas, yt-dlp
- Optional: `torch` for RAFT/MiDaS; `vidstab` or ffmpeg for stabilization

Try-It Fast
1. Install dependencies
   - `pip install opencv-python numpy scipy pandas yt-dlp`
2. Download video & extract frames
   - `yt-dlp -o data/pov/steel_vengeance/video.mp4 https://www.youtube.com/watch?v=RTXTbzctl0c`
   - `python scripts/pov_geometry_infer.py --video data/pov/steel_vengeance/video.mp4 --out data/pov/steel_vengeance`
3. Inspect outputs in the app by loading the generated `track_path.csv` and `accel_df.csv`.
