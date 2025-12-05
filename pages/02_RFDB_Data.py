import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="RFDB Data Analysis", page_icon="📊", layout="wide")

st.title("📊 RFDB Data Analysis")
st.caption("Load real RFDB CSVs and visualize g-forces and egg plots")

rfdb_root = os.path.join(os.path.dirname(__file__), '..', 'rfdb_csvs')

# Captain Coaster list (sorted by rating) with default Steel Vengeance
cc_list = []
cc_display = []
cc_df = None
mapping_path = os.path.join(os.path.dirname(__file__), '..', 'ratings_data', 'rating_to_rfdb_mapping_enhanced.csv')
if os.path.exists(mapping_path):
    try:
        cc_df = pd.read_csv(mapping_path)
        # Prefer explicit columns: ratings_coaster and ratings_park
        coaster_col = 'ratings_coaster' if 'ratings_coaster' in cc_df.columns else ('coaster_name' if 'coaster_name' in cc_df.columns else None)
        park_col = 'ratings_park' if 'ratings_park' in cc_df.columns else ('park_name' if 'park_name' in cc_df.columns else None)
        rating_col = 'average_rating' if 'average_rating' in cc_df.columns else None
        if coaster_col and rating_col and not cc_df.empty:
            cc_df = cc_df.sort_values(rating_col, ascending=False)
            cc_list = cc_df[coaster_col].astype(str).tolist()
            # Build display names "Coaster (Park)"
            if park_col and park_col in cc_df.columns:
                cc_display = [f"{c} ({p})" for c, p in zip(cc_df[coaster_col].astype(str), cc_df[park_col].astype(str))]
            else:
                cc_display = cc_list
    except Exception as e:
        cc_list = []
        cc_display = []
        st.warning(f"Could not load Captain Coaster mapping: {e}")
else:
    st.info("Captain Coaster mapping CSV not found at ratings_data/rating_to_rfdb_mapping_enhanced.csv")

default_cc = 'Steel Vengeance'
selected_display = None
if cc_display:
    default_idx = None
    for i, d in enumerate(cc_display):
        if d.lower().startswith(default_cc.lower()):
            default_idx = i
            break
    if default_idx is None:
        default_idx = 0
    selected_display = st.selectbox(
        "Select Coaster (Captain Coaster)",
        options=cc_display,
        index=default_idx
    )
else:
    cc_search = st.text_input("Search coaster by name", value="", placeholder="Type a Captain Coaster name…")
selected_cc = None
selected_park_cc = None
if selected_display and cc_df is not None:
    # Parse back from display "Coaster (Park)"
    if '(' in selected_display and selected_display.endswith(')'):
        try:
            selected_cc = selected_display[:selected_display.rfind('(')].strip()
            selected_park_cc = selected_display[selected_display.rfind('(')+1:-1].strip()
        except Exception:
            selected_cc = selected_display
            selected_park_cc = None
    else:
        selected_cc = selected_display

# Try to resolve RFDB path for selected CC coaster by heuristic matching
resolved_park = None
resolved_coaster = None
# Determine desired coaster name from dropdown or search
desired_cc_name = selected_cc
if not desired_cc_name:
    # Try search box
    try:
        desired_cc_name = cc_search if 'cc_search' in locals() else None
    except Exception:
        desired_cc_name = None

if desired_cc_name and os.path.isdir(rfdb_root):
    try:
        # heuristic: match folder names containing keywords from coaster name
        cc_key = desired_cc_name.lower().replace(' ', '')
        for park_dir in os.listdir(rfdb_root):
            park_path = os.path.join(rfdb_root, park_dir)
            if not os.path.isdir(park_path):
                continue
            # if we know the park, bias towards it
            coaster_dirs = os.listdir(park_path)
            if selected_park_cc and selected_park_cc.lower().replace(' ', '') not in park_dir.lower().replace(' ', ''):
                # skip non-matching parks if a park name is provided
                continue
            for coaster_dir in coaster_dirs:
                c_key = coaster_dir.lower().replace(' ', '')
                if cc_key in c_key or c_key in cc_key:
                    resolved_park = park_dir
                    resolved_coaster = coaster_dir
                    break
            if resolved_park:
                break
    except Exception:
        pass
try:
    parks = [d for d in os.listdir(rfdb_root) if os.path.isdir(os.path.join(rfdb_root, d))]
except Exception:
    parks = []

park = st.selectbox("Select Park", options=parks, index=(parks.index(resolved_park) if resolved_park in parks else (0 if parks else None)))
coasters = []
if park:
    park_path = os.path.join(rfdb_root, park)
    coasters = [d for d in os.listdir(park_path) if os.path.isdir(os.path.join(park_path, d))]

coaster = st.selectbox("Select Coaster", options=coasters, index=(coasters.index(resolved_coaster) if resolved_coaster in coasters else (0 if coasters else None)))
csv_files = []
if park and coaster:
    coaster_path = os.path.join(rfdb_root, park, coaster)
    csv_files = [f for f in os.listdir(coaster_path) if f.endswith('.csv')]

csv_name = st.selectbox("Select CSV Run", options=csv_files, index=0 if csv_files else None)

if park and coaster and csv_name:
    full_csv = os.path.join(rfdb_root, park, coaster, csv_name)
    try:
        df = pd.read_csv(full_csv)
        # Column resolution: accommodate RFDB variations
        cols_lower = {c.lower(): c for c in df.columns}
        def resolve_any(candidates):
            for cand in candidates:
                if cand.lower() in cols_lower:
                    return cols_lower[cand.lower()]
            return None
        # Common time candidates
        time_col = resolve_any(['Time','time','t','timestamp','elapsed','seconds','s'])
        # Map RFDB variants: xforce/yforce/zforce, gx/gy/gz, accel_x/y/z, etc.
        # Convention: z -> Vertical, x -> Lateral, y -> Longitudinal
        vert_col = resolve_any(['Vertical','vertical','vert','zforce','g_vert','gvertical','gz','accel_z','az'])
        lat_col = resolve_any(['Lateral','lateral','lat','xforce','g_lat','glateral','gx','accel_x','ax'])
        long_col = resolve_any(['Longitudinal','longitudinal','long','yforce','g_long','glongitudinal','gy','accel_y','ay'])
        if not all([vert_col, lat_col, long_col]):
            st.error("Could not resolve required columns (Vertical, Lateral, Longitudinal) -> please check RFDB CSV headers.")
        else:
            accel_df = pd.DataFrame({
                'Time': df[time_col] if time_col else np.arange(len(df)),
                'Vertical': df[vert_col],
                'Lateral': df[lat_col],
                'Longitudinal': df[long_col],
            })

            # Captain Coaster rating and rank (global)
            cc_rating = None
            cc_rank = None
            cc_total = None
            mapping_path = os.path.join(os.path.dirname(__file__), '..', 'ratings_data', 'rating_to_rfdb_mapping_enhanced.csv')
            try:
                mapping_df = pd.read_csv(mapping_path)
                if 'coaster_name' in mapping_df.columns and 'average_rating' in mapping_df.columns:
                    cc_total = len(mapping_df)
                    # naive match by coaster folder name
                    name_match = mapping_df[mapping_df['coaster_name'].str.lower() == coaster.lower()]
                    if not name_match.empty:
                        cc_rating = float(name_match['average_rating'].iloc[0])
                    # rank by average_rating (descending)
                    mapping_df = mapping_df.sort_values('average_rating', ascending=False).reset_index(drop=True)
                    if cc_rating is not None:
                        ranks = (mapping_df['average_rating'].rank(method='min', ascending=False)).astype(int)
                        # find first index with same rating (approx)
                        idx = (np.abs(mapping_df['average_rating'] - cc_rating)).argmin()
                        cc_rank = int(ranks.iloc[idx])
            except Exception:
                pass

            # Summary metrics
            max_g_vert = float(accel_df['Vertical'].max())
            min_g_vert = float(accel_df['Vertical'].min())
            max_g_lat = float(accel_df['Lateral'].abs().max())
            duration_s = float(accel_df['Time'].iloc[-1] - accel_df['Time'].iloc[0]) if 'Time' in accel_df.columns else float(len(accel_df))
            # Airtime: proportion of time with vertical g below 0.5g (tunable)
            airtime_mask = accel_df['Vertical'] < 0.5
            airtime_ratio = float(airtime_mask.sum()) / float(len(accel_df)) if len(accel_df) else 0.0
            airtime_seconds = airtime_ratio * duration_s

            # Comparative metrics across runs of the selected coaster
            comp_airtimes = []
            comp_max_vert = []
            comp_max_lat = []
            coaster_path = os.path.join(rfdb_root, park, coaster)
            for f in csv_files:
                try:
                    df_run = pd.read_csv(os.path.join(coaster_path, f))
                    cols_lower_run = {c.lower(): c for c in df_run.columns}
                    def res_any_run(cands):
                        for cand in cands:
                            if cand.lower() in cols_lower_run:
                                return cols_lower_run[cand.lower()]
                        return None
                    t_col = res_any_run(['Time','time','t','timestamp','elapsed','seconds','s'])
                    v_col = res_any_run(['Vertical','vertical','vert','zforce','g_vert','gvertical','gz','accel_z','az'])
                    la_col = res_any_run(['Lateral','lateral','lat','xforce','g_lat','glateral','gx','accel_x','ax'])
                    lo_col = res_any_run(['Longitudinal','longitudinal','long','yforce','g_long','glongitudinal','gy','accel_y','ay'])
                    if not all([v_col, la_col, lo_col]):
                        continue
                    df_run2 = pd.DataFrame({
                        'Time': df_run[t_col] if t_col else np.arange(len(df_run)),
                        'Vertical': df_run[v_col],
                        'Lateral': df_run[la_col],
                        'Longitudinal': df_run[lo_col],
                    })
                    dur = float(df_run2['Time'].iloc[-1] - df_run2['Time'].iloc[0]) if 'Time' in df_run2.columns else float(len(df_run2))
                    a_mask = df_run2['Vertical'] < 0.5
                    a_ratio = float(a_mask.sum()) / float(len(df_run2)) if len(df_run2) else 0.0
                    comp_airtimes.append(a_ratio * dur)
                    comp_max_vert.append(float(df_run2['Vertical'].max()))
                    comp_max_lat.append(float(df_run2['Lateral'].abs().max()))
                except Exception:
                    continue

            def percentile_of(value, arr):
                if not arr:
                    return None
                sorted_arr = np.sort(np.array(arr))
                return float((np.searchsorted(sorted_arr, value, side='right') / len(sorted_arr)) * 100.0)

            airtime_pct = percentile_of(airtime_seconds, comp_airtimes)
            max_vert_pct = percentile_of(max_g_vert, comp_max_vert)
            max_lat_pct = percentile_of(max_g_lat, comp_max_lat)

            # Top header with Captain Coaster stars if available
            header_left, header_right = st.columns([3, 2])
            with header_left:
                title_text = f"{coaster} — {park}" if coaster and park else "Selected Run"
                st.subheader(title_text)
            with header_right:
                if cc_rating is not None:
                    rank_text = f"#{cc_rank} of {cc_total}" if cc_rank is not None and cc_total is not None else ""
                    st.metric("Captain Coaster Rating", f"{cc_rating:.2f}⭐", rank_text)

            st.markdown("**Ride Metrics**")
            mcol1, mcol2, mcol3, mcol4, mcol5, mcol6 = st.columns(6)
            vert_color = "🚨" if max_g_vert > 5 else ("⚠️" if max_g_vert > 3 else "✅")
            neg_color = "🚨" if min_g_vert < -1.5 else ("⚠️" if min_g_vert < -0.5 else "✅")
            lat_color = "🚨" if max_g_lat > 2.5 else ("⚠️" if max_g_lat > 1.5 else "✅")
            mcol1.metric("Max Vertical G", f"{max_g_vert:.2f}g {vert_color}")
            mcol2.metric("Min Vertical G", f"{min_g_vert:.2f}g {neg_color}")
            mcol3.metric("Max Lateral G", f"{max_g_lat:.2f}g {lat_color}")
            mcol4.metric("Ride Duration", f"{duration_s:.1f}s")
            mcol5.metric("Airtime", f"{airtime_seconds:.1f}s ({airtime_ratio*100:.1f}%)")
            if cc_rating is not None and cc_total is not None:
                rank_text = f"#{cc_rank} of {cc_total}" if cc_rank is not None else f"of {cc_total}"
                mcol6.metric("Captain Coaster Rating", f"{cc_rating:.2f}⭐ ({rank_text})")

            # Comparative percentiles row
            if airtime_pct is not None or max_vert_pct is not None or max_lat_pct is not None:
                st.caption("Comparative run percentiles (within selected coaster runs)")
                c1, c2, c3 = st.columns(3)
                c1.metric("Airtime Percentile", f"{(airtime_pct or 0):.1f}th")
                c2.metric("Max Vert G Percentile", f"{(max_vert_pct or 0):.1f}th")
                c3.metric("Max Lat G Percentile", f"{(max_lat_pct or 0):.1f}th")

            st.markdown("**G-Force Time Series**")
            from utils.visualize import plot_gforce_timeseries, plot_eggplots
            fig_ts = plot_gforce_timeseries(accel_df)
            st.plotly_chart(fig_ts, use_container_width=True)

            st.markdown("**Egg Plots (Comfort Envelopes)**")
            fig_lv, fig_lv2, fig_mag = plot_eggplots(accel_df)
            col_egg1, col_egg2, col_egg3 = st.columns(3)
            with col_egg1:
                st.plotly_chart(fig_lv, use_container_width=True)
            with col_egg2:
                st.plotly_chart(fig_lv2, use_container_width=True)
            with col_egg3:
                st.plotly_chart(fig_mag, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load RFDB CSV: {e}")
else:
    st.info("Select a park, coaster, and CSV to view plots.")
