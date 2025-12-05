import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_track(track_df, color_by=None):
    fig = go.Figure()

    if color_by and color_by in track_df.columns:
        values = track_df[color_by]
        norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
        cmap = cm.get_cmap('turbo')  # vibrant blue–red colormap
        colors = [mcolors.to_hex(cmap(norm(v))) for v in values]
    else:
        colors = ["#1f77b4"] * len(track_df)  # default blue if no color data

    # Draw each segment
    for i in range(len(track_df) - 1):
        fig.add_trace(go.Scatter(
            x=track_df['x'].iloc[i:i+2],
            y=track_df['y'].iloc[i:i+2],
            mode='lines',
            line=dict(color=colors[i], width=4),
            hoverinfo='none'
        ))

    fig.update_layout(
        title="Rollercoaster Track (Colored by Acceleration)",
        xaxis_title="Distance (m)",
        yaxis_title="Height (m)",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        plot_bgcolor="white"
    )

    return fig


def plot_gforce_timeseries(accel_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        subplot_titles=("Vertical (g)", "Lateral (g)", "Longitudinal (g)"))
    fig.add_trace(go.Scatter(x=accel_df['Time'], y=accel_df['Vertical'], name='Vertical', line=dict(color='#1f77b4')), row=1, col=1)
    fig.add_trace(go.Scatter(x=accel_df['Time'], y=accel_df['Lateral'], name='Lateral', line=dict(color='#ff7f0e')), row=2, col=1)
    fig.add_trace(go.Scatter(x=accel_df['Time'], y=accel_df['Longitudinal'], name='Longitudinal', line=dict(color='#2ca02c')), row=3, col=1)
    fig.update_layout(height=600, showlegend=False, margin=dict(l=40, r=20, t=40, b=40))
    return fig


def plot_eggplots(accel_df: pd.DataFrame):
    """Return three egg-plot figures with fixed aspect and comfort envelopes.
    Uses columns 'Lateral', 'Longitudinal', 'Vertical' and optionally 'Time' for coloring.
    """
    # Choose color field (prefer 'Time' if present)
    color_field = 'Time' if 'Time' in accel_df.columns else 'Vertical'

    # Lateral vs Vertical
    fig_lv = go.Figure()
    fig_lv.add_trace(go.Scatter(
        x=accel_df.get('Lateral', pd.Series(dtype=float)),
        y=accel_df.get('Vertical', pd.Series(dtype=float)),
        mode='markers', marker=dict(size=4, color=accel_df.get(color_field, pd.Series(dtype=float)), colorscale='Viridis'),
        name='Samples', hovertemplate='Lat: %{x:.2f} g<br>Vert: %{y:.2f} g<extra></extra>'
    ))
    # Comfort envelopes (ellipses)
    def ellipse(rx, ry, n=180):
        t = np.linspace(0, 2*np.pi, n)
        return rx*np.cos(t), ry*np.sin(t)
    ex, ey = ellipse(2.0, 3.0)
    fig_lv.add_trace(go.Scatter(x=ex, y=ey, mode='lines', line=dict(color='green'), name='Comfort'))
    ex2, ey2 = ellipse(3.5, 4.5)
    fig_lv.add_trace(go.Scatter(x=ex2, y=ey2, mode='lines', line=dict(color='orange', dash='dash'), name='Intense'))
    ex3, ey3 = ellipse(5.0, 6.0)
    fig_lv.add_trace(go.Scatter(x=ex3, y=ey3, mode='lines', line=dict(color='red', dash='dot'), name='Danger'))
    fixed_r = 6.0
    fig_lv.update_layout(
        title='Lat vs Vert (g)', xaxis_title='Lateral (g)', yaxis_title='Vertical (g)', height=360,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(range=[-fixed_r, fixed_r], scaleanchor='y', scaleratio=1),
        yaxis=dict(range=[-fixed_r, fixed_r])
    )

    # Longitudinal vs Vertical
    fig_lv2 = go.Figure()
    fig_lv2.add_trace(go.Scatter(
        x=accel_df.get('Longitudinal', pd.Series(dtype=float)),
        y=accel_df.get('Vertical', pd.Series(dtype=float)),
        mode='markers', marker=dict(size=4, color=accel_df.get(color_field, pd.Series(dtype=float)), colorscale='Viridis'),
        name='Samples', hovertemplate='Long: %{x:.2f} g<br>Vert: %{y:.2f} g<extra></extra>'
    ))
    exa, eya = ellipse(3.0, 3.0)
    fig_lv2.add_trace(go.Scatter(x=exa, y=eya, mode='lines', line=dict(color='green'), name='Comfort'))
    exa2, eya2 = ellipse(4.0, 4.5)
    fig_lv2.add_trace(go.Scatter(x=exa2, y=eya2, mode='lines', line=dict(color='orange', dash='dash'), name='Intense'))
    exa3, eya3 = ellipse(6.0, 6.0)
    fig_lv2.add_trace(go.Scatter(x=exa3, y=eya3, mode='lines', line=dict(color='red', dash='dot'), name='Danger'))
    fixed_r2 = 6.0
    fig_lv2.update_layout(
        title='Long vs Vert (g)', xaxis_title='Longitudinal (g)', yaxis_title='Vertical (g)', height=360,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(range=[-fixed_r2, fixed_r2], scaleanchor='y', scaleratio=1),
        yaxis=dict(range=[-fixed_r2, fixed_r2])
    )

    # Horizontal signed magnitude vs Vertical (left/right indicated by sign)
    fig_mag = go.Figure()
    lat = accel_df.get('Lateral', pd.Series(dtype=float))
    lon = accel_df.get('Longitudinal', pd.Series(dtype=float))
    mag = np.sqrt(lat**2 + lon**2)
    signed_mag = np.sign(lat) * mag
    fig_mag.add_trace(go.Scatter(
        x=signed_mag, y=accel_df.get('Vertical', pd.Series(dtype=float)),
        mode='markers', marker=dict(size=4, color=accel_df.get(color_field, pd.Series(dtype=float)), colorscale='Viridis'),
        name='Samples', hovertemplate='Signed Mag: %{x:.2f} g<br>Vert: %{y:.2f} g<extra></extra>'
    ))
    em, eym = ellipse(4.0, 3.0)
    fig_mag.add_trace(go.Scatter(x=em, y=eym, mode='lines', line=dict(color='green'), name='Comfort'))
    em2, eym2 = ellipse(5.5, 4.0)
    fig_mag.add_trace(go.Scatter(x=em2, y=eym2, mode='lines', line=dict(color='orange', dash='dash'), name='Intense'))
    em3, eym3 = ellipse(7.5, 5.0)
    fig_mag.add_trace(go.Scatter(x=em3, y=eym3, mode='lines', line=dict(color='red', dash='dot'), name='Danger'))
    fixed_r3 = 7.5
    fig_mag.update_layout(
        title='Horiz Magnitude vs Vert (g)', xaxis_title='Horizontal magnitude (g)', yaxis_title='Vertical (g)', height=360,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(range=[-fixed_r3, fixed_r3], scaleanchor='y', scaleratio=1),
        yaxis=dict(range=[-fixed_r3/2, fixed_r3])
    )

    return fig_lv, fig_lv2, fig_mag
