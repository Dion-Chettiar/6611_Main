import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Soccer Analytics Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN DARK THEME CSS ---
st.markdown("""
<style>
    .main { padding-top: 1rem; background-color: #0E1117; }
    .metric-card {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        padding: 1.5rem; border-radius: 12px; color: #f9fafb;
        text-align: center; margin: 0.5rem 0; border: 1px solid #4b5563;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .stMetric {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid #475569; padding: 1rem; border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    .chart-container {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 12px; padding: 1.5rem; border: 1px solid #475569;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    h1 { color: #f9fafb !important; font-weight: 700 !important; }
    h2, h3 { color: #e5e7eb !important; }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    try:
        sub_optimizer_url = "https://raw.githubusercontent.com/Dion-Chettiar/6611_Main/refs/heads/main/sub_optimizer%202.csv"
        performance_url = "https://raw.githubusercontent.com/Dion-Chettiar/6611_Main/refs/heads/main/Performance_Dropoff_Per_Player.csv"
        trios_url = "https://raw.githubusercontent.com/Dion-Chettiar/6611_Main/refs/heads/main/trios.parquet"  # Update with your actual file path or URL
        sub_data = pd.read_csv(sub_optimizer_url)
        perf_data = pd.read_csv(performance_url)
        trios = pd.read_parquet(trios_url)
        sub_data.columns = sub_data.columns.str.strip()
        perf_data.columns = perf_data.columns.str.strip()
        merged_data = pd.merge(
            sub_data, 
            perf_data[['Player', 'Actual Impact']].rename(columns={'Actual Impact': 'Actual_Impact_Perf'}), 
            on='Player', 
            how='left'
        )
        merged_data['Overperformance'] = merged_data['Impact'] - merged_data['Predicted Impact']
        merged_data = merged_data.rename(columns={
            'Impact': 'Actual Impact',
            'Fatigue_Score': 'Fatigue Score',
            'Sub_Recommendation': 'Sub Recommendation',
            'Sub Early Probability': 'Sub Early Probability'
        })
        return merged_data, trios
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

DARK_COLORS = {
    'primary': '#3b82f6',
    'secondary': '#8b5cf6',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'info': '#06b6d4',
    'dark': '#1f2937',
    'light': '#f9fafb'
}

data, trios = load_data()

# --- DASHBOARD HEADER ---
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">⚽ Soccer Analytics Dashboard</h1>
    <p style="color: #9ca3af; font-size: 1.2rem;">Advanced performance analysis with modern visualization</p>
</div>
""", unsafe_allow_html=True)

# --- METRICS ---
if not data.empty:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Players", len(data), delta=None)
    with col2:
        avg_fatigue = data['Fatigue Score'].mean()
        st.metric("Avg Fatigue Score", f"{avg_fatigue:.2f}")
    with col3:
        high_fatigue = len(data[data['Fatigue Score'] > 2])
        st.metric("High Fatigue Players", high_fatigue)
    with col4:
        sub_early_count = len(data[data['Sub Recommendation'] == 'Sub Early'])
        st.metric("Sub Early Recommendations", sub_early_count)

    # --- SIDEBAR FILTERS ---
    st.sidebar.markdown("### 📊 Dashboard Controls")
    unique_recommendations = ['All'] + sorted(data['Sub Recommendation'].unique().tolist())
    unique_positions = ['All'] + sorted([pos.strip() for pos in ','.join(data['Position'].unique()).split(',') if pos.strip()])
    top_n = st.sidebar.slider("Show Top N Players", 5, 50, 15)
    selected_recommendation = st.sidebar.selectbox("Filter by Recommendation", unique_recommendations)
    selected_position = st.sidebar.selectbox("Filter by Position", unique_positions)
    fatigue_range = st.sidebar.slider(
        "Fatigue Score Range",
        float(data['Fatigue Score'].min()),
        float(data['Fatigue Score'].max()),
        (float(data['Fatigue Score'].min()), float(data['Fatigue Score'].max()))
    )
    filtered_data = data.copy()
    if selected_recommendation != 'All':
        filtered_data = filtered_data[filtered_data['Sub Recommendation'] == selected_recommendation]
    if selected_position != 'All':
        filtered_data = filtered_data[filtered_data['Position'].str.contains(selected_position, na=False)]
    filtered_data = filtered_data[
        (filtered_data['Fatigue Score'] >= fatigue_range[0]) & 
        (filtered_data['Fatigue Score'] <= fatigue_range[1])
    ]
    top_performers = filtered_data.nlargest(top_n, 'Overperformance')

    # --- PLAYER OVERPERFORMANCE BAR CHART ---
    col_left, col_right = st.columns([3, 1])
    with col_left:
        st.markdown("### 🎯 Player Overperformance Analysis")
        if not top_performers.empty:
            fig = px.bar(
                top_performers,
                x='Overperformance',
                y='Player',
                orientation='h',
                color='Fatigue Score',
                color_continuous_scale=[
                    [0.0, DARK_COLORS['success']],
                    [0.5, DARK_COLORS['warning']],
                    [1.0, DARK_COLORS['danger']]
                ],
                title=f"Top {top_n} Overperforming Players",
                labels={'Overperformance': 'Overperformance Value', 'Player': 'Player Name'},
                hover_data=['Position', 'Minutes', 'Actual Impact', 'Sub Recommendation']
            )
            fig.update_layout(
                height=max(400, len(top_performers) * 25),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=DARK_COLORS['light'], size=12),
                title_font=dict(color=DARK_COLORS['light'], size=16),
                xaxis=dict(gridcolor='rgba(75, 85, 99, 0.3)', color=DARK_COLORS['light']),
                yaxis=dict(gridcolor='rgba(75, 85, 99, 0.3)', color=DARK_COLORS['light']),
                coloraxis_colorbar=dict(title_font_color=DARK_COLORS['light'], tickfont_color=DARK_COLORS['light'])
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No players found matching the current filters.")
    with col_right:
        st.markdown("### 🚨 Featured Player")
        if not filtered_data.empty:
            highest_fatigue_player = filtered_data.loc[filtered_data['Fatigue Score'].idxmax()]
            fatigue_score = highest_fatigue_player['Fatigue Score']
            if fatigue_score > 2:
                fatigue_color = DARK_COLORS['danger']
                fatigue_emoji = "🔴"
                fatigue_status = "High Alert"
            elif fatigue_score > 1:
                fatigue_color = DARK_COLORS['warning']
                fatigue_emoji = "🟡"
                fatigue_status = "Monitor"
            else:
                fatigue_color = DARK_COLORS['success']
                fatigue_emoji = "🟢"
                fatigue_status = "Good Condition"
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {fatigue_color}20 0%, {fatigue_color}40 100%);
                padding: 1.5rem;
                border-radius: 12px;
                color: {DARK_COLORS['light']};
                text-align: center;
                border: 2px solid {fatigue_color};
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            ">
                <h3 style="margin-bottom: 1rem; color: {DARK_COLORS['light']};">{highest_fatigue_player['Player']}</h3>
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">{fatigue_emoji} {fatigue_score:.2f}</div>
                <p style="color: {fatigue_color}; font-size: 1.2rem; margin-bottom: 1rem; font-weight: bold;">{fatigue_status}</p>
                <hr style="border-color: {fatigue_color}; margin: 1rem 0;">
                <div style="text-align: left;">
                    <p><strong>Position:</strong> {highest_fatigue_player['Position']}</p>
                    <p><strong>Minutes:</strong> {highest_fatigue_player['Minutes']:,.0f}</p>
                    <p><strong>Overperformance:</strong> {highest_fatigue_player['Overperformance']:.4f}</p>
                    <p><strong>Recommendation:</strong> {highest_fatigue_player['Sub Recommendation']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("No players match current filters.")

    # --- TRIO CHART (SINGLE DATASET, CENTERED) ---
    st.markdown("---")
    st.markdown("## ⚽ Optimal Player Trio Combination")
    if not trios.empty:
        selected_row = st.selectbox(
            "Select a trio to visualize",
            trios.index,
            format_func=lambda idx: (
                f"{trios.loc[idx, 'Player 1']} + {trios.loc[idx, 'Player 2']} + {trios.loc[idx, 'Player 3']}"
            )
        )
        row = trios.loc[selected_row]
        p1, p2, p3 = row['Player 1'], row['Player 2'], row['Player 3']
        pos1, pos2, pos3 = row['Position 1'], row['Position 2'], row['Position 3']
        img1, img2, img3 = row['Img_Url1'], row['Img_Url2'], row['Img_Url3']
        nodes = {
            p1: (0.5, 1.0),
            p2: (0.15, 0.2),
            p3: (0.85, 0.2)
        }
        images = {p1: img1, p2: img2, p3: img3}
        labels = {
            p1: f"<b>{p1}</b><br>({pos1})",
            p2: f"<b>{p2}</b><br>({pos2})",
            p3: f"<b>{p3}</b><br>({pos3})"
        }
        st.markdown('<div style="display: flex; justify-content: center;">', unsafe_allow_html=True)
        fig = go.Figure()
        # Draw triangle as lines
        x_coords = [nodes[p1][0], nodes[p2][0], nodes[p3][0], nodes[p1][0]]
        y_coords = [nodes[p1][1], nodes[p2][1], nodes[p3][1], nodes[p1][1]]
        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords,
            mode="lines",
            line=dict(color="#1e293b", width=4),
            hoverinfo='skip',
            showlegend=False
        ))
        # Add images and labels
        for name, (x, y) in nodes.items():
            if name == p3:
                img_x = x + 0.08  # Move image right
                sizex, sizey, opacity = 0.20, 0.22, 0.85
            else:
                img_x = x - 0.08
                sizex, sizey, opacity = 0.16, 0.16, 1.0
            fig.add_layout_image(
                dict(
                    source=images[name],
                    xref="x", yref="y",
                    x=img_x, y=y+0.06,
                    sizex=sizex, sizey=sizey,
                    xanchor="center", yanchor="middle",
                    layer="above",
                    opacity=opacity
                )
            )
            fig.add_trace(go.Scatter(
                x=[x], y=[y-0.13], text=[labels[name]], mode="text",
                textfont=dict(color="#1e293b", size=16),
                hoverinfo='skip',
                showlegend=False
            ))
        fig.update_xaxes(visible=False, range=[0, 1.2])
        fig.update_yaxes(visible=False, range=[0, 1.15])
        fig.update_layout(
            width=650, height=500,
            plot_bgcolor="#f8fafc",
            paper_bgcolor="#f8fafc",
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False
        )
        st.plotly_chart(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("No trio combination data available.")

    # --- PLAYER PERFORMANCE ANALYSIS TABLE ---
    if not filtered_data.empty:
        st.markdown("### 📋 Player Performance Analysis")
        sort_col1, sort_col2 = st.columns(2)
        with sort_col1:
            sort_column = st.selectbox(
                "Sort by:", 
                ['Overperformance', 'Fatigue Score', 'Minutes', 'Actual Impact', 'Sub Early Probability']
            )
        with sort_col2:
            sort_order = st.radio("Order:", ['Descending', 'Ascending'], horizontal=True)
        ascending = sort_order == 'Ascending'
        display_data = filtered_data.sort_values(sort_column, ascending=ascending)
        display_columns = [
            'Player', 'Position', 'Minutes', 'Actual Impact', 
            'Predicted Impact', 'Overperformance', 'Fatigue Score', 
            'Sub Recommendation', 'Sub Early Probability'
        ]
        formatted_data = display_data[display_columns].copy()
        numeric_columns = ['Actual Impact', 'Predicted Impact', 'Overperformance']
        for col in numeric_columns:
            formatted_data[col] = formatted_data[col].apply(lambda x: f"{x:.4f}")
        formatted_data['Fatigue Score'] = formatted_data['Fatigue Score'].apply(lambda x: f"{x:.2f}")
        formatted_data['Minutes'] = formatted_data['Minutes'].apply(lambda x: f"{x:,.0f}")
        formatted_data['Sub Early Probability'] = formatted_data['Sub Early Probability'].apply(lambda x: f"{x:.3f}")
        st.dataframe(
            formatted_data,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Player": st.column_config.TextColumn("Player", width="medium"),
                "Position": st.column_config.TextColumn("Position", width="small"),
                "Minutes": st.column_config.TextColumn("Minutes", width="small"),
                "Fatigue Score": st.column_config.TextColumn("Fatigue Score", width="small"),
                "Sub Recommendation": st.column_config.TextColumn("Recommendation", width="medium"),
                "Sub Early Probability": st.column_config.TextColumn("Sub Probability", width="small")
            }
        )
        st.markdown("### 📥 Export Data")
        col1, col2 = st.columns(2)
        with col1:
            csv_data = formatted_data.to_csv(index=False)
            st.download_button(
                label="📁 Download Filtered Data as CSV",
                data=csv_data,
                file_name=f"soccer_analytics_filtered_{len(formatted_data)}_players.csv",
                mime="text/csv"
            )
        with col2:
            full_csv = data.to_csv(index=False)
            st.download_button(
                label="📁 Download Full Dataset as CSV", 
                data=full_csv,
                file_name="soccer_analytics_full_dataset.csv",
                mime="text/csv"
            )
        # --- ANALYTICS SUMMARY ---
        st.markdown("### 📊 Analytics Summary")
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            recommendation_counts = filtered_data['Sub Recommendation'].value_counts()
            fig_pie = px.pie(
                values=recommendation_counts.values,
                names=recommendation_counts.index,
                title="Recommendation Distribution",
                color_discrete_map={
                    'Sub Early': DARK_COLORS['danger'],
                    'Monitor': DARK_COLORS['warning'],
                    'Keep in Game': DARK_COLORS['success']
                }
            )
            fig_pie.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=DARK_COLORS['light']),
                title_font=dict(color=DARK_COLORS['light'], size=14)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        with summary_col2:
            fig_hist = px.histogram(
                filtered_data,
                x='Fatigue Score',
                nbins=20,
                title="Fatigue Score Distribution",
                color_discrete_sequence=[DARK_COLORS['primary']]
            )
            fig_hist.add_vline(x=1, line_dash="dash", line_color=DARK_COLORS['warning'], annotation_text="Moderate Fatigue")
            fig_hist.add_vline(x=2, line_dash="dash", line_color=DARK_COLORS['danger'], annotation_text="High Fatigue")
            fig_hist.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=DARK_COLORS['light']),
                title_font=dict(color=DARK_COLORS['light'], size=14),
                xaxis=dict(gridcolor='rgba(75, 85, 99, 0.3)', color=DARK_COLORS['light']),
                yaxis=dict(gridcolor='rgba(75, 85, 99, 0.3)', color=DARK_COLORS['light'])
            )
            st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.error("❌ No data available. Please check your data files.")
