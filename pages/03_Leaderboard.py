import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from utils.submission_manager import load_submissions_from_s3, load_submission_geometry_from_s3


def calculate_pareto_front(submissions):
    """
    Calculate the Pareto front from submissions.
    A submission is on the Pareto front if no other submission has both:
    - Higher fun rating (score)
    - Higher safety score
    
    Returns:
        List of indices of submissions on the Pareto front, sorted by score
    """
    if not submissions:
        return []
    
    pareto_indices = []
    
    for i, sub_i in enumerate(submissions):
        is_dominated = False
        for j, sub_j in enumerate(submissions):
            if i == j:
                continue
            # Check if sub_j dominates sub_i
            # sub_j dominates sub_i if it has both higher score AND higher safety_score
            # Using >= to handle ties properly
            if sub_j['score'] >= sub_i['score'] and sub_j['safety_score'] >= sub_i['safety_score']:
                # If one is strictly greater, it dominates
                if sub_j['score'] > sub_i['score'] or sub_j['safety_score'] > sub_i['safety_score']:
                    is_dominated = True
                    break
        
        if not is_dominated:
            pareto_indices.append(i)
    
    # Sort by score (descending) for consistent ordering
    pareto_indices.sort(key=lambda idx: submissions[idx]['score'], reverse=True)
    
    return pareto_indices

st.set_page_config(page_title="Leaderboard", page_icon="🏆", layout="wide")

st.title("🏆 Rollercoaster Leaderboard")
st.caption("Ranked by combined score (Fun Rating + Safety Score)")

# Load submissions from S3
with st.spinner("Loading leaderboard..."):
    submissions = load_submissions_from_s3()

if not submissions:
    st.info("No submissions yet. Be the first to submit your rollercoaster design!")
    if st.button("🎢 Go to Builder"):
        st.switch_page("pages/01_Builder.py")
else:
    # Display statistics
    total_submissions = len(submissions)
    avg_score = sum(s['score'] for s in submissions) / total_submissions if submissions else 0
    avg_safety = sum(s['safety_score'] for s in submissions) / total_submissions if submissions else 0
    top_score = max(s['score'] for s in submissions) if submissions else 0
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("Total Submissions", total_submissions)
    with col_stat2:
        st.metric("Top Score", f"{top_score:.2f}⭐")
    with col_stat3:
        st.metric("Average Score", f"{avg_score:.2f}⭐")
    with col_stat4:
        st.metric("Average Safety", f"{avg_safety:.2f}★")
    
    st.divider()
    
    # Pareto Front Visualization
    st.subheader("📈 Pareto Front Analysis")
    st.caption("Submissions on the Pareto front are optimal - no other submission has both higher fun rating AND higher safety score")
    
    # Calculate Pareto front
    pareto_indices = calculate_pareto_front(submissions)
    pareto_set = set(pareto_indices)
    
    # Calculate utopia point (ideal point: max fun rating, max safety score)
    all_scores = [s['score'] for s in submissions]
    all_safety = [s['safety_score'] for s in submissions]
    utopia_score = max(all_scores) if all_scores else 0
    utopia_safety = max(all_safety) if all_safety else 0
    
    # Create scatter plot
    fig_pareto = go.Figure()
    
    # Plot all submissions
    all_names = [s['submitter_name'] for s in submissions]
    
    # Separate user submissions from RFDB submissions
    non_pareto_user_scores = []
    non_pareto_user_safety = []
    non_pareto_user_names = []
    non_pareto_rfdb_scores = []
    non_pareto_rfdb_safety = []
    non_pareto_rfdb_names = []
    
    for i in range(len(submissions)):
        if i not in pareto_set:
            sub = submissions[i]
            is_rfdb = sub.get('source') == 'RFDB' or 'RFDB:' in sub.get('submitter_name', '')
            if is_rfdb:
                non_pareto_rfdb_scores.append(all_scores[i])
                non_pareto_rfdb_safety.append(all_safety[i])
                non_pareto_rfdb_names.append(all_names[i])
            else:
                non_pareto_user_scores.append(all_scores[i])
                non_pareto_user_safety.append(all_safety[i])
                non_pareto_user_names.append(all_names[i])
    
    # Plot RFDB submissions (less visible)
    if non_pareto_rfdb_scores:
        fig_pareto.add_trace(go.Scatter(
            x=non_pareto_rfdb_scores,
            y=non_pareto_rfdb_safety,
            mode='markers',
            marker=dict(
                color='lightgray',
                size=6,
                opacity=0.4,
                line=dict(width=0.5, color='gray')
            ),
            name='RFDB Submissions',
            text=non_pareto_rfdb_names,
            hovertemplate='<b>%{text}</b><br>Fun Rating: %{x:.2f}⭐<br>Safety: %{y:.2f}★<extra></extra>'
        ))
    
    # Plot user submissions (more visible)
    if non_pareto_user_scores:
        fig_pareto.add_trace(go.Scatter(
            x=non_pareto_user_scores,
            y=non_pareto_user_safety,
            mode='markers',
            marker=dict(
                color='#4CAF50',  # Bright green
                size=10,
                opacity=0.8,
                line=dict(width=2, color='darkgreen')
            ),
            name='Your Submissions',
            text=non_pareto_user_names,
            hovertemplate='<b>%{text}</b><br>Fun Rating: %{x:.2f}⭐<br>Safety: %{y:.2f}★<extra></extra>'
        ))
    
    # Separate Pareto front points into user and RFDB
    pareto_user_points = []
    pareto_rfdb_points = []
    
    for i in pareto_indices:
        sub = submissions[i]
        is_rfdb = sub.get('source') == 'RFDB' or 'RFDB:' in sub.get('submitter_name', '')
        point = (all_scores[i], all_safety[i], all_names[i], i)
        if is_rfdb:
            pareto_rfdb_points.append(point)
        else:
            pareto_user_points.append(point)
    
    # Sort both lists by fun rating (x-axis) descending
    pareto_user_points.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    pareto_rfdb_points.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    
    # Combine for line (all Pareto points)
    all_pareto_points = pareto_user_points + pareto_rfdb_points
    all_pareto_points.sort(key=lambda x: x[0], reverse=True)
    
    if all_pareto_points:
        pareto_scores = [p[0] for p in all_pareto_points]
        pareto_safety = [p[1] for p in all_pareto_points]
        
        # Add connecting line FIRST (so it appears behind markers) - only if we have multiple points
        if len(pareto_scores) > 1:
            # For a proper Pareto front, we want to show the trade-off curve
            # Sort by x-axis (fun rating) descending for line connection
            sorted_for_line = sorted(zip(pareto_scores, pareto_safety), key=lambda x: x[0], reverse=True)
            line_scores = [p[0] for p in sorted_for_line]
            line_safety = [p[1] for p in sorted_for_line]
            
            # Add the line trace with more visible styling
            # Draw line BEFORE markers so it appears behind them
            fig_pareto.add_trace(go.Scatter(
                x=line_scores,
                y=line_safety,
                mode='lines',
                line=dict(
                    color='#FF0000',  # Bright red
                    width=4,  # Thicker line for visibility
                    dash='dash'
                ),
                name='Pareto Front',
                hoverinfo='skip',
                showlegend=True,
                connectgaps=False,
                opacity=1.0,  # Fully opaque
                legendgroup='pareto'  # Group with markers
            ))
        
        # Add RFDB Pareto markers (less visible)
        if pareto_rfdb_points:
            rfdb_pareto_scores = [p[0] for p in pareto_rfdb_points]
            rfdb_pareto_safety = [p[1] for p in pareto_rfdb_points]
            rfdb_pareto_names = [p[2] for p in pareto_rfdb_points]
            
            fig_pareto.add_trace(go.Scatter(
                x=rfdb_pareto_scores,
                y=rfdb_pareto_safety,
                mode='markers',
                marker=dict(
                    color='red',
                    size=10,
                    symbol='star',
                    opacity=0.6,
                    line=dict(width=1.5, color='darkred')
                ),
                name='Pareto Front (RFDB)',
                text=rfdb_pareto_names,
                hovertemplate='<b>%{text}</b> (Pareto Optimal - RFDB)<br>Fun Rating: %{x:.2f}⭐<br>Safety: %{y:.2f}★<extra></extra>',
                showlegend=True,
                legendgroup='pareto'
            ))
        
        # Add user Pareto markers (more visible)
        if pareto_user_points:
            user_pareto_scores = [p[0] for p in pareto_user_points]
            user_pareto_safety = [p[1] for p in pareto_user_points]
            user_pareto_names = [p[2] for p in pareto_user_points]
            
            fig_pareto.add_trace(go.Scatter(
                x=user_pareto_scores,
                y=user_pareto_safety,
                mode='markers',
                marker=dict(
                    color='#FF6B00',  # Bright orange
                    size=14,
                    symbol='star',
                    opacity=1.0,
                    line=dict(width=3, color='darkorange')
                ),
                name='Pareto Front (Your Submissions)',
                text=user_pareto_names,
                hovertemplate='<b>%{text}</b> (Pareto Optimal - Your Design!)<br>Fun Rating: %{x:.2f}⭐<br>Safety: %{y:.2f}★<extra></extra>',
                showlegend=True,
                legendgroup='pareto'
            ))
    
    # Add utopia point (ideal point)
    fig_pareto.add_trace(go.Scatter(
        x=[utopia_score],
        y=[utopia_safety],
        mode='markers',
        marker=dict(
            color='blue',
            size=16,
            symbol='star',
            line=dict(width=2, color='darkblue')
        ),
        name='Utopia Point (Ideal)',
        text=['Utopia Point'],
        hovertemplate='<b>Utopia Point (Ideal)</b><br>Fun Rating: %{x:.2f}⭐<br>Safety: %{y:.2f}★<br><i>Theoretical best: max of both objectives</i><extra></extra>'
    ))
    
    # Calculate axis ranges - ensure we can see the full 0-5 range, especially near boundaries
    if all_scores and all_safety:
        x_min = min(all_scores)
        x_max = max(all_scores)
        y_min = min(all_safety)
        y_max = max(all_safety)
        
        # Ensure we show at least 0-5 range, with extra padding to see boundaries clearly
        # Always show full 0-5 range for both axes to see the 5-star boundaries
        x_axis_min = 0
        x_axis_max = 5.2  # Show slightly beyond 5.0 to see points at exactly 5.0
        y_axis_min = 0
        y_axis_max = 5.2  # Show slightly beyond 5.0 to see points at exactly 5.0
        
        # If data extends beyond 5.0, show that too
        if x_max > 5.0:
            x_axis_max = max(5.2, x_max + 0.2)
        if y_max > 5.0:
            y_axis_max = max(5.2, y_max + 0.2)
    else:
        x_axis_min = 0
        x_axis_max = 5.2
        y_axis_min = 0
        y_axis_max = 5.2
    
    # Update layout
    fig_pareto.update_layout(
        title="Fun Rating vs Safety Score - Pareto Front",
        xaxis_title="Fun Rating (⭐)",
        yaxis_title="Safety Score (★)",
        height=500,
        hovermode='closest',
        xaxis=dict(range=[x_axis_min, x_axis_max]),
        yaxis=dict(range=[y_axis_min, y_axis_max]),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig_pareto, use_container_width=True)
    
    # Display Pareto front info
    col_pareto1, col_pareto2 = st.columns(2)
    with col_pareto1:
        st.metric("Submissions on Pareto Front", len(pareto_indices))
    with col_pareto2:
        if pareto_indices:
            best_combined = max(
                (submissions[i]['score'] + submissions[i]['safety_score'] for i in pareto_indices),
                default=0
            )
            st.metric("Best Combined Score (Pareto)", f"{best_combined:.2f}")
    
    # List Pareto front submissions
    if pareto_indices:
        with st.expander("📋 View Pareto Front Submissions", expanded=False):
            pareto_data = []
            for rank, idx in enumerate(pareto_indices, start=1):
                sub = submissions[idx]
                pareto_data.append({
                    'Rank': rank,
                    'Submitter': sub['submitter_name'],
                    'Fun Rating': f"{sub['score']:.2f}⭐",
                    'Safety Score': f"{sub['safety_score']:.1f}★",
                    'Combined': f"{(sub['score'] + sub['safety_score']):.2f}"
                })
            pareto_df = pd.DataFrame(pareto_data)
            st.dataframe(pareto_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Create leaderboard table
    st.subheader("📊 Leaderboard")
    
    # Prepare data for display
    leaderboard_data = []
    for idx, submission in enumerate(submissions, start=1):
        # Parse timestamp
        try:
            timestamp = datetime.fromisoformat(submission['timestamp'].replace('Z', '+00:00'))
            formatted_time = timestamp.strftime("%Y-%m-%d %H:%M")
        except:
            formatted_time = submission['timestamp']
        
        leaderboard_data.append({
            'Rank': idx,
            'Submitter': submission['submitter_name'],
            'Fun Rating': f"{submission['score']:.2f}⭐",
            'Safety Score': f"{submission['safety_score']:.1f}★",
            'Combined': f"{(submission['score'] + submission['safety_score']):.2f}",
            'Submitted': formatted_time,
            'submission_id': submission['submission_id']
        })
    
    df = pd.DataFrame(leaderboard_data)
    
    # Display table with styling
    st.dataframe(
        df[['Rank', 'Submitter', 'Fun Rating', 'Safety Score', 'Combined', 'Submitted']],
        use_container_width=True,
        hide_index=True
    )
    
    st.divider()
    
    # Allow viewing individual submissions
    st.subheader("🔍 View Submission Details")
    
    # Create selection interface
    submission_options = [f"{idx+1}. {s['submitter_name']} - {s['score']:.2f}⭐" 
                          for idx, s in enumerate(submissions)]
    
    selected_idx = st.selectbox(
        "Select a submission to view:",
        options=range(len(submission_options)),
        format_func=lambda x: submission_options[x],
        index=0
    )
    
    if selected_idx is not None and selected_idx < len(submissions):
        selected_submission = submissions[selected_idx]
        
        col_view1, col_view2 = st.columns([1, 1])
        
        with col_view1:
            st.markdown("**Submission Info**")
            st.write(f"**Submitter:** {selected_submission['submitter_name']}")
            st.write(f"**Fun Rating:** {selected_submission['score']:.2f}⭐")
            st.write(f"**Safety Score:** {selected_submission['safety_score']:.1f}★")
            st.write(f"**Combined Score:** {(selected_submission['score'] + selected_submission['safety_score']):.2f}")
            try:
                timestamp = datetime.fromisoformat(selected_submission['timestamp'].replace('Z', '+00:00'))
                st.write(f"**Submitted:** {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            except:
                st.write(f"**Submitted:** {selected_submission['timestamp']}")
        
        with col_view2:
            # Check if this is an RFDB submission (no geometry available)
            is_rfdb = selected_submission.get('source') == 'RFDB' or 'RFDB:' in selected_submission.get('submitter_name', '')
            
            if is_rfdb:
                st.markdown("**Track Information**")
                st.info("📊 This is a real rollercoaster from RFDB (RideForcesDB).\n\n"
                       "Track geometry is not available as this data comes from accelerometer recordings, "
                       "not track design files.")
                if 'park' in selected_submission:
                    st.write(f"**Park:** {selected_submission.get('park', 'N/A')}")
                if 'coaster' in selected_submission:
                    st.write(f"**Coaster:** {selected_submission.get('coaster', 'N/A')}")
            else:
                # Load and display geometry for user-submitted tracks
                with st.spinner("Loading track geometry..."):
                    geometry = load_submission_geometry_from_s3(selected_submission['submission_id'])
                
                if geometry and len(geometry.get('x', [])) > 1:
                    st.markdown("**Track Profile**")
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=geometry['x'],
                        y=geometry['y'],
                        mode='lines',
                        line=dict(color='rgb(255, 75, 75)', width=3),
                        name='Track',
                        hovertemplate='Distance: %{x:.1f}m<br>Height: %{y:.1f}m<extra></extra>'
                    ))
                    
                    # Ground line
                    if geometry['x']:
                        fig.add_shape(
                            type="line",
                            x0=0, x1=max(geometry['x']),
                            y0=0, y1=0,
                            line=dict(color="green", width=2, dash="dash")
                        )
                    
                    fig.update_layout(
                        title="Track Side View",
                        xaxis_title="Distance (m)",
                        yaxis_title="Height (m)",
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Could not load track geometry.")
    
    st.divider()
    
    # Navigation
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("🎢 Go to Builder", use_container_width=True):
            st.switch_page("pages/01_Builder.py")
    with col_nav2:
        if st.button("🔄 Refresh Leaderboard", use_container_width=True):
            st.rerun()

