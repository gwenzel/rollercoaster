import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import math
from pathlib import Path

# ----------------------------
# Configuration (adjust if you want)
# ----------------------------
# Time step and total simulation time:
# - dt: integration timestep in seconds. Smaller dt -> more accurate integration but more steps and longer runtime.
#   Choose dt based on the fastest dynamics you need to resolve and the desired accuracy. For this coarse
#   track-following model dt in the range ~0.001..0.01 is common; dt=0.005 is a reasonable compromise.
# - t_tot: total simulation duration in seconds. This determines how long the simulation runs and how many
#   output rows are produced (steps = t_tot / dt).
#
# Impact of changing t_tot:
# - Increasing t_tot increases the number of steps, memory usage, CSV/XLSX size, and plotting time.
# - Decreasing t_tot shortens the simulated time interval; if too small you may not traverse the full track.
# - t_tot does NOT automatically stop the simulation when the vehicle reaches the end of the track. The
#   current main loop always runs for `steps` iterations. If you want an automatic stop when the vehicle
#   reaches the final track point, we can add a check in the loop to break when the integrated position
#   reaches/passes the last track point (recommended for convenience).
#
# How to pick a good t_tot value:
# - Estimate track_length (m) and expected average speed (m/s):
#     t_est = track_length / expected_average_speed
#   Then choose t_tot = t_est * margin (e.g. 1.1..1.5) to allow for slowdowns/pauses.
# - Example: track_length = 200 m, avg_speed = 8 m/s -> t_est = 25 s (so t_tot = 25 is reasonable).
# - If you don't know expected speed, pick a conservatively large t_tot and/or implement an automatic
#   end-of-track break so the simulation stops when the vehicle finishes the track.
#
# Performance note:
# - steps = t_tot / dt. With dt=0.005 and t_tot=25 -> steps = 5000 rows, which is modest. Reducing dt
#   to 0.001 or increasing t_tot to 100 will increase rows and memory proportionally.

dt = 0.005           # time step [s]
t_tot = 25           # total duration [s] — adjust based on track length/expected speed (see notes above)
steps = int(t_tot / dt)

vectorspacing = max(1, int(steps / 20))
vectorscale = 0.2

# physics
m = 6000.0
g = np.array([0.0, 0.0, -9.81])
g_norm = np.linalg.norm(g)
rho = 1.3
Cd = 0.6
A = 4.0
mu = 0.02

# files
track_file = Path('physics/trackpoints.txt')  # file with columns x,y,z (in mm according to the original code)
#track_file = Path('trackpoints3.txt')  # file with columns x,y,z (in mm according to the original code)
out_csv = Path('simulacion_rc.csv')
out_xlsx = Path('simulacion_rc.xlsx')

# ----------------------------
# Loads and checks
# ----------------------------
if not track_file.exists():
    raise FileNotFoundError(f"{track_file} not found. Place the track points file (x,y,z) in the same directory.")

ps = np.loadtxt(track_file, delimiter=',', encoding='utf-8-sig') / 1000.0  # convert to meters
if ps.ndim != 2 or ps.shape[1] != 3:
    raise ValueError("trackpoints.txt must have 3 columns x,y,z per row.")

n_points = ps.shape[0]
if n_points < 3:
    raise ValueError("At least 3 track points are required.")

tangents = np.zeros_like(ps)
for i in range(n_points):
    if i == 0:
        tang = ps[1] - ps[0]
    elif i == n_points - 1:
        tang = ps[-1] - ps[-2]
    else:
        tang = (ps[i+1] - ps[i-1]) / 2.0
    norm = np.linalg.norm(tang)
    tangents[i] = (tang / norm) if norm > 1e-9 else np.array([1.0, 0.0, 0.0])

# ----------------------------
# Track precomputations (smoothed tangents)
# ----------------------------
# tangents at each point using central difference (more stable)
tangents = np.zeros_like(ps)
for i in range(n_points):
    if i == 0:
        tang = ps[1] - ps[0]
    elif i == n_points - 1:
        tang = ps[-1] - ps[-2]
    else:
        tang = (ps[i+1] - ps[i-1]) / 2.0
    norm = np.linalg.norm(tang)
    tangents[i] = (tang / norm) if norm > 1e-9 else np.array([1.0, 0.0, 0.0])

# ----------------------------
# Prepare simulation arrays
# ----------------------------
tussen = np.zeros(steps, dtype=int)         # index of the "nearest" track point
dp = np.zeros((steps, 3))
e_dp = np.zeros((steps, 3))
theta = np.zeros(steps)
phi = np.zeros(steps)
r = np.zeros(steps)
g_a = np.zeros((steps, 3))
g_N = np.zeros((steps, 3))
a_f = np.zeros((steps, 3))
a_d = np.zeros((steps, 3))
a = np.zeros(steps)        # scalar acceleration along tangent (m/s^2)
v = np.zeros(steps)        # scalar speed along tangent (m/s)
ds = np.zeros((steps, 3))
s = np.zeros((steps, 3))   # integrated 3D position
R = np.zeros((steps, 3))
M = np.zeros((steps, 3))
a_eq = np.zeros((steps, 3))   # centripetal acceleration vector
G = np.zeros((steps, 3))       # sum of forces per unit mass (g + others)
G_N = np.zeros((steps, 3))     # modified "normal" component

# initial conditions
tussen[0] = 0
dp[0] = np.array([0.0, 0.0, 0.0])
e_dp[0] = tangents[0]
theta[0] = np.pi/2
phi[0] = 0.0
r[0] = 0.0
g_a[0] = np.zeros(3)
a_eq[0] = np.zeros(3)
a_eq[0] = np.zeros(3)
# compute initial decomposition of gravity and initial tangential acceleration
g_par_mag0 = np.dot(g, e_dp[0])
g_a[0] = g_par_mag0 * e_dp[0]
g_N[0] = g - g_a[0]
a_f[0] = np.zeros(3)
a_d[0] = np.zeros(3)
a[0] = 0.0
v[0] = 1                       # velocidad inicial (ajustable)
ds[0] = np.zeros(3)
s[0] = ps[0].copy()              # comenzar exactamente en el primer punto de la pista
R[0] = np.array([np.inf, np.inf, np.inf])
M[0] = np.array([np.inf, np.inf, np.inf])
# evaluate initial resistances and centripetal (a_eq[0] already zeros)
normal_magnitude0 = np.linalg.norm(g_N[0])
friction_acc_mag0 = mu * normal_magnitude0
drag_acc_mag0 = 0.5 * rho * Cd * A / m * v[0]**2
a_gravity_tang0 = g_par_mag0
sign_motion0 = 1.0 if v[0] >= 0 else -1.0
a_tangential0 = a_gravity_tang0 - sign_motion0 * (friction_acc_mag0 + drag_acc_mag0)

# set initial scalar acceleration and construct G[0], G_N[0]
a[0] = a_tangential0
a_scalar_prev = a[0]
friction_vec0 = -friction_acc_mag0 * e_dp[0] * sign_motion0
drag_vec0 = -drag_acc_mag0 * e_dp[0] * sign_motion0
G[0] = g + friction_vec0 + drag_vec0 + a_eq[0]
G_N[0] = g_N[0] + a_eq[0]

# search parameters for the reference index 'p'
search_radius = 10   # search in p - radius ... p + radius

# helper: safe normalization
def safe_norm(vect, eps=1e-9):
    n = np.linalg.norm(vect)
    if n < eps:
        return n, vect * 0.0
    else:
        return n, vect / n

# helper: point-segment distance
def point_segment_distance(point, a, b):
    ab = b - a
    ab_len2 = np.dot(ab, ab)
    if ab_len2 == 0:
        return np.linalg.norm(point - a), a
    t = np.dot(point - a, ab) / ab_len2
    t_clamped = max(0.0, min(1.0, t))
    proj = a + t_clamped * ab
    return np.linalg.norm(point - proj), proj

# helper: circumcenter radius from three points
def circumcenter_radius(p1, p2, p3):
    a = p1
    b = p2
    c = p3
    ab = b - a
    ac = c - a
    cross = np.cross(ab, ac)
    cross_norm2 = np.dot(cross, cross)
    if cross_norm2 < 1e-12:
        return np.inf, np.array([np.inf, np.inf, np.inf])  # colineales o casi
    ab_len2 = np.dot(ab, ab)
    ac_len2 = np.dot(ac, ac)
    num = np.cross(cross, ab) * ac_len2 + np.cross(ac, cross) * ab_len2
    center = a + num / (2 * cross_norm2)
    Rvec = a - center
    R = np.linalg.norm(Rvec)
    return R, center

# ----------------------------
# Main loop (Velocity-Verlet integrating scalar speed along the tangent)
# ----------------------------
p = 1  # reference index in ps (p will be used for local search)
a_scalar_prev = a_scalar_prev  # preserve initial acceleration computed above

for i in range(1, steps):
    # ---------- 1) find nearest segment (search window around p)
    start_j = max(1, p - search_radius)
    end_j = min(n_points - 1, p + search_radius)
    min_dist = np.inf
    best_proj = None
    best_seg = p
    # compute point->segment distance for each segment [j-1, j]
    for j in range(start_j, end_j + 1):
        d, proj = point_segment_distance(s[i-1], ps[j-1], ps[j])
        if d < min_dist:
            min_dist = d
            best_proj = proj
            best_seg = j
    p = best_seg
    tussen[i] = p

    # use the segment direction (ps[p] - ps[p-1]) as local tangent
    seg_vec = ps[p] - ps[p-1]
    seg_len, seg_dir = safe_norm(seg_vec)
    if seg_len < 1e-9:
    # if the segment is almost zero-length, fallback to precomputed tangent
        e_dp[i] = tangents[p]
    else:
        e_dp[i] = seg_dir

    # distance to the projected point and magnitude r
    dp_vec = best_proj - s[i-1] if best_proj is not None else ps[p] - s[i-1]
    r[i], e_dp_calc = safe_norm(dp_vec)
    if r[i] < 1e-9:
        # if we are exactly on the track, keep r = 0 and keep tangent
        r[i] = 0.0
        e_dp[i] = e_dp[i]
    else:
        pass

    # angles (informational)
    if r[i] > 1e-9:
        theta[i] = np.arccos(max(-1.0, min(1.0, dp_vec[2] / r[i])))
        phi[i] = np.arctan2(dp_vec[1], dp_vec[0])
    else:
        theta[i] = theta[i-1]
        phi[i] = phi[i-1]

    # ---------- 2) gravity decomposition
    # component parallel to the tangent (vector)
    g_par_mag = np.dot(g, e_dp[i])
    g_a[i] = g_par_mag * e_dp[i]            # vector gravedad paralelo a la tangente
    g_N[i] = g - g_a[i]                     # resto (aprox "normal" al track)

    # ---------- 3) resistive forces (magnitudes) and tangential acceleration (scalar)
    # friction: mu * normal_magnitude (approx using ||G_N||)
    normal_magnitude = np.linalg.norm(g_N[i])  # aproximación de fuerza normal por unidad de masa (m*s^-2)
    friction_acc_mag = mu * normal_magnitude
    # aerodynamic drag opposite to motion: 0.5*rho*Cd*A/m * v^2
    drag_acc_mag = 0.5 * rho * Cd * A / m * v[i-1]**2
    # tangential acceleration from gravity (already signed)
    a_gravity_tang = g_par_mag
    # resistances oppose the motion
    sign_motion = 1.0 if v[i-1] >= 0 else -1.0
    a_tangential = a_gravity_tang - sign_motion * (friction_acc_mag + drag_acc_mag)

    # ---------- Velocity-Verlet (scalar along the tangent)
    v_half = v[i-1] + 0.5 * a_scalar_prev * dt
    ds[i] = v_half * dt * e_dp[i]
    s[i] = s[i-1] + ds[i]
    a_scalar_new = a_tangential
    v[i] = v_half + 0.5 * a_scalar_new * dt
    if v[i] < 0 and abs(v[i]) < 1e-6:
        v[i] = 0.0

    a[i] = a_scalar_new
    a_scalar_prev = a_scalar_new

    # ---------- 4) local curvature and centripetal acceleration
    idx1 = max(0, p-2)
    idx2 = max(1, p-1)
    idx3 = min(n_points-1, p)
    R_val, center = circumcenter_radius(ps[idx1], ps[idx2], ps[idx3])
    if np.isfinite(R_val) and R_val > 1e-6:
        R_vec = s[i] - center
        R[i] = R_vec
        M[i] = center
        e_R_norm, e_R = safe_norm(R_vec)
        a_eq[i] = - (v[i]**2 / R_val) * e_R if e_R_norm > 1e-9 else np.zeros(3)
    else:
        R[i] = np.array([np.inf, np.inf, np.inf])
        M[i] = np.array([np.inf, np.inf, np.inf])
        a_eq[i] = np.zeros(3)

    # ---------- 5) total forces (per unit mass) and normal G_N
    friction_vec = -friction_acc_mag * e_dp[i] * sign_motion
    drag_vec = -drag_acc_mag * e_dp[i] * sign_motion
    G[i] = g + friction_vec + drag_vec + a_eq[i]
    G_N[i] = g_N[i] + a_eq[i]

# ----------------------------
# 3D plots and vectors (you can disable if you don't want to show them)
# ----------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(projection='3d')
ax.scatter(s[:,0], s[:,1], s[:,2], s=2, c="b", label='integrated trajectory (s)')
ax.plot(ps[:,0], ps[:,1], ps[:,2], c="r", label='track points (ps)')
# quivers (sample to avoid overcrowding)
idxs = np.arange(0, steps, max(1, vectorspacing))
ax.quiver(s[idxs,0], s[idxs,1], s[idxs,2],
          v[idxs]*e_dp[idxs,0], v[idxs]*e_dp[idxs,1], v[idxs]*e_dp[idxs,2],
          length=1.0, normalize=False, arrow_length_ratio=vectorscale, linewidth=0.6, label='velocity (v * tangent)')
ax.quiver(s[idxs,0], s[idxs,1], s[idxs,2],
          a_eq[idxs,0], a_eq[idxs,1], a_eq[idxs,2],
          length=1.0, normalize=False, arrow_length_ratio=vectorscale, linewidth=0.6, color='red', label='a_centripetal')
ax.quiver(s[idxs,0], s[idxs,1], s[idxs,2],
          G[idxs,0], G[idxs,1], G[idxs,2],
          length=1.0, normalize=False, arrow_length_ratio=vectorscale, linewidth=0.6, color='orange', label='G (total per unit mass)')
ax.set_aspect('equal')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

# ----------------------------
# Export CSV and XLSX
# ----------------------------
with open(out_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile, delimiter=",")
    writer.writerow(["Time", "Between", "x", "y", "z", "v", "a", "g_N_norm", "g_a_vector", "R_norm", "a_eq_norm", "G_N_signed",
                     "a_tot_x", "a_tot_y", "a_tot_z",
                     "specific_force_x", "specific_force_y", "specific_force_z",
                     "a_longitudinal", "a_lateral", "a_vertical",
                     "f_long", "f_lat", "f_vert",
                     "f_long_g", "f_lat_g", "f_vert_g"])
    for row in range(0, steps):
        gn_vec = G_N[row]
        if gn_vec[2] > 0:
            G_N_signed = -1.0 * np.linalg.norm(gn_vec) / np.linalg.norm(g)
        else:
            G_N_signed = np.linalg.norm(gn_vec) / np.linalg.norm(g)
    # compute total inertial acceleration vector: tangential component + centripetal
        a_tot_vec = a[row] * e_dp[row] + a_eq[row]
        # acelerómetro mide la "specific force" = a_tot_vec - g (si g = [0,0,-9.81])
        specific_force = a_tot_vec - g

        # definir ejes locales: longitudinal = e_dp, vertical = global z, lateral = vertical x longitudinal
        ez = np.array([0.0, 0.0, 1.0])
        el = e_dp[row]
        lat_norm, lat = safe_norm(np.cross(ez, el))
        # fallback si lat es nulo (por ejemplo el paralelo a ez)
        if lat_norm < 1e-9:
            # choose an arbitrary lateral axis perpendicular to el
            lat = np.array([0.0, 1.0, 0.0]) if abs(el[0]) < 0.9 else np.array([1.0, 0.0, 0.0])
            _, lat = safe_norm(lat - np.dot(lat, el) * el)
        # componentes: longitudinal (a lo largo de la tangente), lateral (lado a lado), vertical (global Z)
        a_long = float(np.dot(a_tot_vec, el))
        a_lat = float(np.dot(a_tot_vec, lat))
        a_vert = float(np.dot(a_tot_vec, ez))

        # specific-force (what an accelerometer measures) and its projections
        f_spec = specific_force
        f_long = float(np.dot(f_spec, el))
        f_lat = float(np.dot(f_spec, lat))
        f_vert = float(np.dot(f_spec, ez))

        writer.writerow([
            round(row * dt, 6),
            f"{int(tussen[row]) - 1} - {int(tussen[row])}",
            float(s[row][0]), float(s[row][1]), float(s[row][2]),
            float(v[row]),
            float(a[row]),
            float(np.linalg.norm(G_N[row])),
            f"{g_a[row].tolist()}",
            float(np.linalg.norm(R[row])) if np.isfinite(np.linalg.norm(R[row])) else "",
            float(np.linalg.norm(a_eq[row])),
            float(G_N_signed),
            # a_tot components
            float(a_tot_vec[0]), float(a_tot_vec[1]), float(a_tot_vec[2]),
            # specific force components (what an accelerometer would read, roughly)
            float(specific_force[0]), float(specific_force[1]), float(specific_force[2]),
            # projected components (inertial a_tot)
            a_long, a_lat, a_vert,
            # projected components of specific-force and their normalized-by-g versions
            f_long, f_lat, f_vert,
            f_long / g_norm, f_lat / g_norm, f_vert / g_norm
        ])


# convert to xlsx
df = pd.read_csv(out_csv)
# OPTION: rename to keep 'G_N' as in your plotting lines if you prefer
# Uncomment these 3 lines if you want to use df['G_N'] in the 2D plot
# df = df.rename(columns={
#     'g_N_norm': 'G_N',
#     'a_eq_norm': 'a_eq',
#     'R_norm': 'R'
# })

# Try to export to Excel. pandas uses openpyxl for .xlsx files by default.
# If openpyxl is not installed we catch the ImportError and print a clear
# instruction for the user to install it in their environment.
try:
    df.to_excel(out_xlsx, index=False)
except ImportError as e:
    # Friendly message telling the user how to install openpyxl
    print("Could not export to Excel because the optional dependency 'openpyxl' is missing.")
    print("Install it with:\n  python -m pip install --upgrade pip; python -m pip install openpyxl")
    print(f"Error details: {e}")
except Exception as e:
    # Catch other errors when writing the Excel file and report them
    print("Error while trying to write the Excel file:", e)
    print("You can inspect the generated CSV instead:", out_csv.resolve())

# ----------------------------
# 2D plot (G_N, v, a, z)
# The block below is disabled to avoid creating/showing the figure. It is preserved for
# reference and can be re-enabled by changing the `if False` to `if True` or removing
# the guard entirely.
# ----------------------------
if False:
    df.set_index('Time', inplace=True)

    # If you did NOT rename, use 'g_N_norm'. If you DID rename, change 'g_N_norm' to 'G_N' here.
    gn_col = 'G_N' if 'G_N' in df.columns else 'g_N_norm'

    # Ensure numeric types
    for col in [gn_col, 'v', 'a', 'z']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(right=0.75)

    twin1 = ax.twinx()
    twin2 = ax.twinx()
    twin3 = ax.twinx()

    twin2.spines.right.set_position(("axes", 1.1))
    twin3.spines.right.set_position(("axes", 1.2))

    p0, = ax.plot(df.index, df[gn_col]/g_norm, "k-", label="Normal G-force (relative)")
    p1, = twin1.plot(df.index, df['v'], "r-", label="Speed [m/s]")
    p2, = twin2.plot(df.index, df['a'], "b-", label="Acceleration [m/s^2]")
    p3, = twin3.plot(df.index, df['z'], "g-", label="Height [m]")

    ax.set_xlim(df.index.min(), df.index.max())
    ax.set_ylim(df[gn_col].min()/g_norm - 2, df[gn_col].max()/g_norm + 2)
    twin1.set_ylim(df['v'].min() - 3, df['v'].max() + 3)
    twin2.set_ylim(df['a'].min() - 3, df['a'].max() + 3)
    twin3.set_ylim(df['z'].min() - 3, df['z'].max() + 3)

    ax.set_xlabel("t [s]")
    ax.set_ylabel("G_N [x 9.81 m/s^2]")
    twin1.set_ylabel("v [m/s]")
    twin2.set_ylabel("a [m/s^2]")
    twin3.set_ylabel("h [m]")

    ax.yaxis.label.set_color(p0.get_color())
    twin1.yaxis.label.set_color(p1.get_color())
    twin2.yaxis.label.set_color(p2.get_color())
    twin3.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=7, width=1.5)
    ax.tick_params(axis='y', colors=p0.get_color(), **tkw)
    twin1.tick_params(axis='y', colors=p1.get_color(), **tkw)
    twin2.tick_params(axis='y', colors=p2.get_color(), **tkw)
    twin3.tick_params(axis='y', colors=p3.get_color(), **tkw)
    ax.tick_params(axis='x', **tkw)

    handles = [p0, p1, p2, p3]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    # plt.show()  # intentionally disabled

print(f"Simulation completed. CSV saved at: {out_csv.resolve()}")
print(f"Excel saved at: {out_xlsx.resolve()}")



# ----------------------------
# ----------------------------
# Plot: acceleration components normalized by g (longitudinal, lateral, vertical, total)
# ----------------------------
# compute total acceleration vectors per step (a_tangential * e_dp + a_eq)
a_tot = a.reshape((-1, 1)) * e_dp + a_eq
g_norm = np.linalg.norm(g)

# prepare projection arrays
a_long = np.zeros(steps)
a_lat = np.zeros(steps)
a_vert = np.zeros(steps)
a_tot_norm = np.zeros(steps)

# additionally compute specific force (what an accelerometer measures): f_spec = a_tot - g
f_long = np.zeros(steps)
f_lat = np.zeros(steps)
f_vert = np.zeros(steps)
f_tot_norm = np.zeros(steps)

for i in range(steps):
    el = e_dp[i]
    ez = np.array([0.0, 0.0, 1.0])
    # lateral axis: cross(ez, el) (horizontal lateral). if nearly zero, use robust fallback
    lat_norm, lat = safe_norm(np.cross(ez, el))
    if lat_norm < 1e-9:
        tmp = np.array([0.0, 1.0, 0.0]) if abs(el[0]) < 0.9 else np.array([1.0, 0.0, 0.0])
        _, lat = safe_norm(tmp - np.dot(tmp, el) * el)

    a_tot_vec = a_tot[i]
    a_long[i] = float(np.dot(a_tot_vec, el))
    a_lat[i] = float(np.dot(a_tot_vec, lat))
    a_vert[i] = float(np.dot(a_tot_vec, ez))
    a_tot_norm[i] = float(np.linalg.norm(a_tot_vec))

    # specific force (accelerometer) and its projections
    f_spec = a_tot_vec - g
    f_long[i] = float(np.dot(f_spec, el))
    f_lat[i] = float(np.dot(f_spec, lat))
    f_vert[i] = float(np.dot(f_spec, ez))
    f_tot_norm[i] = float(np.linalg.norm(f_spec))


# Graph App Style (Vertical Blue, Lateral Green, Longitudinal Red) in g
t = np.arange(steps) * dt
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(t, f_vert / g_norm, color='tab:blue', label='Vertical')
ax.plot(t, f_lat  / g_norm, color='tab:green', label='Lateral')
ax.plot(t, f_long / g_norm, color='tab:red', label='Longitudinal')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Acceleration [g]')
ax.set_title('G-Forces (accelerometer specific force)')
ax.set_ylim(-12, 12)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()



# ----------------------------
# Export simple accelerations CSV for phone comparison
# Format: Time, Lateral, Vertical, Longitudinal  (units: g)
# ----------------------------
acc_csv = Path('Accelerations_CSV.csv')
with open(acc_csv, 'w', newline='') as accfile:
    acc_writer = csv.writer(accfile, delimiter=',')
    acc_writer.writerow(['Time', 'Lateral', 'Vertical', 'Longitudinal'])
    for i in range(steps):
        t_val = i * dt
        # write in g-units (divide by standard gravity magnitude)
        acc_writer.writerow([f"{t_val:.2f}",
                             f_lat[i] / g_norm,
                             f_vert[i] / g_norm,
                             f_long[i] / g_norm])

print(f"Accelerations CSV saved at: {acc_csv.resolve()}")
