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
    return x, y


def vertical_drop_profile(height=40, steepness=0.9, **kwargs):
    current_height = kwargs.get('current_height', height)
    drop_height = min(height, current_height)

    min_horizontal_distance = drop_height * 1.73
    horizontal_distance = min_horizontal_distance / max(steepness, 0.3)

    # Add eased pre-drop and extended runout for smoother connection
    pre_length = horizontal_distance * 0.15
    main_length = horizontal_distance * 0.7
    exit_length = horizontal_distance * 0.15

    x_pre = np.linspace(0, pre_length, 12)
    pre_progress = x_pre / pre_length
    y_pre = - (drop_height * 0.15) * (pre_progress ** 3)

    x_main = np.linspace(pre_length, pre_length + main_length, 30)
    main_progress = (x_main - pre_length) / main_length
    y_main = y_pre[-1] - (drop_height * 0.7) * main_progress

    x_exit = np.linspace(pre_length + main_length, horizontal_distance, 12)
    exit_progress = (x_exit - (pre_length + main_length)) / exit_length
    y_exit = y_main[-1] - (drop_height * 0.15) * exit_progress ** 2

    x = np.concatenate([x_pre, x_main, x_exit])
    y = np.concatenate([y_pre, y_main, y_exit])
    return x, y


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
    return x, y


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
    return x, y


def spiral_profile(diameter=25, turns=1.5, **kwargs):
    """Horizontal spiral/helix profile that starts and ends at y=0 (relative)."""
    r0 = diameter / 2
    total_length = diameter * turns * 0.8

    # Entry: ease-in banking/undulation, starting exactly at y=0
    entry_length = total_length * 0.2
    n_entry = 12
    x_entry = np.linspace(0, entry_length, n_entry)
    theta_entry = np.linspace(0, 0.3 * np.pi, n_entry)
    entry_progress = (x_entry / entry_length)
    # Undulation scaled by progress^2 to start flat at y=0
    y_entry = (3.0 * np.sin(theta_entry)) * (entry_progress ** 2)

    # Main: gentle undulation around y=0 with slight radius tightening
    main_length = total_length * 0.6
    n_main = 40
    x_main = np.linspace(entry_length, entry_length + main_length, n_main)
    radius_progress = np.linspace(0, 1, n_main)
    r_main = r0 * (1 + 0.1 * radius_progress)
    theta_main = np.linspace(0.3 * np.pi, (turns * 2 - 0.3) * np.pi, n_main)
    y_main = 3.0 * np.sin(theta_main)

    # Exit: ease-out to y=0
    exit_length = total_length * 0.2
    n_exit = 12
    x_exit = np.linspace(entry_length + main_length, total_length, n_exit)
    theta_exit = np.linspace((turns * 2 - 0.3) * np.pi, turns * 2 * np.pi, n_exit)
    exit_progress = (x_exit - (entry_length + main_length)) / exit_length
    # Undulation scaled by (1 - progress)^2 to finish flat at y=0
    y_exit = (3.0 * np.sin(theta_exit)) * ((1 - exit_progress) ** 2)

    x = np.concatenate([x_entry, x_main, x_exit])
    y = np.concatenate([y_entry, y_main, y_exit])
    return x, y


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
    return x, y


def bunny_hop_profile(length=20, height=8, **kwargs):
    max_safe_height = length * 0.3
    actual_height = min(height, max_safe_height)

    entry_length = length * 0.35
    x_entry = np.linspace(0, entry_length, 12)
    entry_progress = x_entry / entry_length
    y_entry = actual_height * (entry_progress ** 3) * 0.5

    main_length = length * 0.3
    x_main = np.linspace(entry_length, entry_length + main_length, 12)
    main_progress = (x_main - entry_length) / main_length
    y_main = actual_height * (0.5 + 0.5 * (1 - (2 * main_progress - 1) ** 2))

    exit_length = length * 0.35
    x_exit = np.linspace(entry_length + main_length, length, 12)
    exit_progress = (x_exit - (entry_length + main_length)) / exit_length
    y_exit = actual_height * (1 - exit_progress) ** 3 * 0.5

    x = np.concatenate([x_entry, x_main, x_exit])
    y = np.concatenate([y_entry, y_main, y_exit])
    return x, y


def flat_section_profile(length=30, slope=-0.02, **kwargs):
    x = np.linspace(0, length, 20)
    y = slope * x
    return x, y
