import numpy as np


def lift_hill_profile(length=50, height=40, **kwargs):
    max_realistic_height = length * 0.8
    actual_height = min(height, max_realistic_height)

    entry_length = length * 0.15
    x_entry = np.linspace(0, entry_length, 12)
    entry_progress = x_entry / entry_length
    y_entry = actual_height * entry_progress ** 2 * 0.1

    main_length = length * 0.6
    x_main = np.linspace(entry_length, entry_length + main_length, int(main_length/2))
    main_progress = (x_main - entry_length) / main_length
    y_main = actual_height * (0.1 + main_progress * 0.75)

    exit_length = length * 0.25
    x_exit = np.linspace(entry_length + main_length, length, 15)
    exit_progress = (x_exit - entry_length - main_length) / exit_length
    y_exit = actual_height * (0.85 + 0.15 * (1 - (1 - exit_progress) ** 3))

    x = np.concatenate([x_entry, x_main, x_exit])
    y = np.concatenate([y_entry, y_main, y_exit])
    z = np.zeros_like(x)  # No banking on lift hill
    return x, y, z


def vertical_drop_profile(height=40, steepness=0.9, **kwargs):
    current_height = kwargs.get('current_height', height)
    drop_height = min(height, current_height)

    min_horizontal_distance = drop_height * 1.73
    horizontal_distance = min_horizontal_distance / max(steepness, 0.3)

    # Single smooth curve from top to bottom - no segments
    # Use a single cosine-based ease for natural curvature
    n_points = 60
    x = np.linspace(0, horizontal_distance, n_points)
    
    # Progress from 0 to 1
    progress = x / horizontal_distance
    
    # Smooth ease using raised cosine (starts horizontal, ends horizontal)
    # This gives natural constant-curvature transition
    ease = (1 - np.cos(progress * np.pi)) / 2
    
    # Apply the drop
    y = -drop_height * ease
    
    z = np.zeros_like(x)  # No banking on vertical drop
    return x, y, z


def loop_profile(diameter=30, **kwargs):
    r = diameter / 2
    transition_length = r * 1.6
    a_squared = r * transition_length

    n_trans = 50
    ds = transition_length / (n_trans - 1)
    s_entry = np.linspace(0, transition_length, n_trans)
    theta_entry = s_entry**2 / (2 * a_squared)

    x_entry = np.zeros(n_trans)
    y_entry = np.zeros(n_trans)
    for i in range(1, n_trans):
        avg_theta = (theta_entry[i] + theta_entry[i-1]) / 2
        x_entry[i] = x_entry[i-1] + np.cos(avg_theta) * ds
        y_entry[i] = y_entry[i-1] + np.sin(avg_theta) * ds

    final_entry_angle = theta_entry[-1]
    final_entry_x = x_entry[-1]
    final_entry_y = y_entry[-1]

    start_angle = final_entry_angle - np.pi/2
    end_angle = (2 * np.pi - final_entry_angle) - np.pi/2

    n_loop = 80
    theta_circle = np.linspace(start_angle, end_angle, n_loop)

    center_x = final_entry_x - r * np.sin(final_entry_angle)
    center_y = final_entry_y + r * np.cos(final_entry_angle)

    x_loop = center_x + r * np.cos(theta_circle)
    y_loop = center_y + r * np.sin(theta_circle)

    x_exit = np.zeros(n_trans)
    y_exit = np.zeros(n_trans)
    x_exit[0] = x_loop[-1]
    y_exit[0] = y_loop[-1]
    for i in range(1, n_trans):
        s_curr = transition_length - i * ds
        s_prev = transition_length - (i-1) * ds
        theta_curr = s_curr**2 / (2 * a_squared)
        theta_prev = s_prev**2 / (2 * a_squared)
        avg_theta = (theta_curr + theta_prev) / 2
        x_exit[i] = x_exit[i-1] + np.cos(-avg_theta) * ds
        y_exit[i] = y_exit[i-1] + np.sin(-avg_theta) * ds

    x = np.concatenate([x_entry, x_loop, x_exit])
    y = np.concatenate([y_entry, y_loop, y_exit])
    y = y - np.linspace(0, y[-1], len(y))
    z = np.zeros_like(x)  # No lateral banking in vertical loop
    return x, y, z


def airtime_hill_profile(length=40, height=15, **kwargs):
    max_safe_height = length * 0.5
    actual_height = min(height, max_safe_height)

    entry_length = length * 0.35
    x_entry = np.linspace(0, entry_length, 20)
    entry_progress = x_entry / entry_length
    y_entry = actual_height * entry_progress ** 3 * 0.3

    main_length = length * 0.3
    x_main = np.linspace(entry_length, entry_length + main_length, 20)
    main_progress = (x_main - entry_length) / main_length
    y_main = actual_height * (0.3 + 0.7 * (1 - (2 * main_progress - 1) ** 2))

    exit_length = length * 0.35
    x_exit = np.linspace(entry_length + main_length, length, 20)
    exit_progress = (x_exit - (entry_length + main_length)) / exit_length
    y_exit = actual_height * (1 - exit_progress) ** 3 * 0.3

    x = np.concatenate([x_entry, x_main, x_exit])
    y = np.concatenate([y_entry, y_main, y_exit])
    z = np.zeros_like(x)  # No banking on airtime hill
    return x, y, z


def spiral_profile(diameter=25, turns=1.5, **kwargs):
    """Horizontal spiral/helix profile that starts and ends at y=0 (relative)."""
    # Gentle spiral with smooth undulation
    total_length = diameter * turns * np.pi  # More horizontal space for gentler spiral
    n_points = int(50 * turns)  # Scale points with turns
    
    x = np.linspace(0, total_length, n_points)
    progress = x / total_length
    
    # Gentle vertical undulation (helical motion)
    # Amplitude increases at start, constant in middle, decreases at end
    amplitude = 4.0  # meters of vertical motion
    
    # Smooth envelope to ease in/out
    envelope = np.sin(progress * np.pi)  # Peaks at middle, zero at ends
    
    # Multiple cycles of up/down based on turns
    y = amplitude * np.sin(progress * turns * 2 * np.pi) * envelope
    
    # Add lateral banking for the spiral (realistic corkscrew motion)
    # Banking angle increases/decreases smoothly
    bank_amplitude = diameter * 0.3  # Lateral displacement
    z = bank_amplitude * np.cos(progress * turns * 2 * np.pi) * envelope
    
    return x, y, z


def banked_turn_profile(radius=30, angle=90, **kwargs):
    """Banked horizontal turn that starts and ends at y=0 (relative)."""
    theta = np.linspace(0, np.radians(angle), 60)
    bank_target = kwargs.get('bank_deg', 25)

    # Banking easing
    n = len(theta)
    t = np.linspace(0, 1, n)
    bank = np.piecewise(
        t,
        [t < 0.2, (t >= 0.2) & (t <= 0.8), t > 0.8],
        [lambda u: bank_target * (u / 0.2) ** 3,
         lambda u: bank_target,
         lambda u: bank_target * (1 - ((u - 0.8) / 0.2) ** 3)]
    )

    x = radius * np.sin(theta)

    # Elevation modulation around zero with easing to ensure y[0]=y[-1]=0
    # Use sin(2*theta) scaled by entry/exit ease
    ease = np.piecewise(
        t,
        [t < 0.2, (t >= 0.2) & (t <= 0.8), t > 0.8],
        [lambda u: (u/0.2)**2,
         1.0,
         lambda u: (1 - (u-0.8)/0.2)**2]
    )
    y = 0.5 * np.sin(2 * theta) * ease
    
    # Add banking for the turn (tilts inward based on bank angle)
    # Banking smoothly increases/decreases with the ease curve
    bank_height = radius * np.sin(np.radians(bank_target))
    z = -bank_height * ease  # Negative for inward banking
    
    return x, y, z


def bunny_hop_profile(length=20, height=8, **kwargs):
    max_safe_height = length * 0.3
    actual_height = min(height, max_safe_height)

    # Smooth bump with eased entrance/exit to avoid curvature spikes
    n_points = 40
    x = np.linspace(0, length, n_points)
    progress = x / length
    
    # Apply smoothstep to the sine wave amplitude to soften entry/exit
    # This creates a "smooth sine" that eases in and out
    ease_in_out = progress ** 2 * (3 - 2 * progress)  # cubic smoothstep
    
    # Combine: sine for the bump shape, smoothstep for gentle transitions
    y = actual_height * np.sin(progress * np.pi) * ease_in_out
    
    z = np.zeros_like(x)  # No banking on bunny hop
    return x, y, z


def flat_section_profile(length=30, slope=-0.02, **kwargs):
    x = np.linspace(0, length, 20)
    y = slope * x
    z = np.zeros_like(x)  # No banking on flat section
    return x, y, z


def launch_profile(length=40, speed_boost=20.0, **kwargs):
    """Magnetic launch section (LSM/LIM) that adds speed to the train.
    
    Args:
        length: Length of launch section in meters
        speed_boost: Target speed increase in m/s (e.g., 20 m/s = 72 km/h)
        
    Note: The speed boost is applied in the physics engine by detecting this block type.
    The track itself is flat with slight incline for realism.
    """
    n_points = max(30, int(length / 1.5))  # Denser sampling for launch detection
    x = np.linspace(0, length, n_points)
    
    # Slight downward slope during launch (realistic for many launched coasters)
    y = -0.5 * (x / length)  # 0.5m drop over the launch
    
    z = np.zeros_like(x)  # No banking
    return x, y, z
