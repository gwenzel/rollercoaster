import plotly.graph_objects as go
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
