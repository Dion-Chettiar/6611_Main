import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page configuration with dark theme
st.set_page_config(
    page_title="Soccer Analytics Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Dark Corporate CSS Styling
st.markdown("""
<style>
    /* Main background and container styling */
    .main {
        padding-top: 1rem;
        background-color: #0E1117;
    }
    
    /* Custom metric cards with modern gradients */
    .metric-card {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: #f9fafb;
        text-align: center;
        margin: 0.5rem 0;
        border: 1px solid #4b5563;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    /* Enhanced metric styling */
    .stMetric {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid #475569;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Chart container styling */
    .chart-container {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #475569;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    /* Alert styling with modern colors */
    .alert-high {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        border: 1px solid #ef4444;
        color: #fef2f2;
        padding: 0.75rem;
        border-radius: 8px;
    }
    
    .alert-medium {
        background: linear-gradient(135deg, #d97706 0%, #b45309 100%);
        border: 1px solid #f59e0b;
        color: #fffbeb;
        padding: 0.75rem;
        border-radius: 8px;
    }
    
    .alert-low {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        border: 1px solid #10b981;
        color: #ecfdf5;
        padding: 0.75rem;
        border-radius: 8px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1f2937;
    }
    
    /* Title styling */
    h1 {
        color: #f9fafb !important;
        font-weight: 700 !important;
    }
    
    h2, h3 {
        color: #e5e7eb !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load CSV data from GitHub URLs with enhanced error handling"""
    try:
        # Your GitHub raw URLs
        sub_optimizer_url = "https://raw.githubusercontent.com/Dion-Chettiar/Streamlit_Dashboard/refs/heads/main/sub_optimizer%202.csv"
        performance_url = "https://raw.githubusercontent.com/Dion-Chettiar/Streamlit_Dashboard/refs/heads/main/Performance_Dropoff_Per_Player.csv"
        players_df_url = "https://raw.githubusercontent.com/Dion-Chettiar/6611_Main/refs/heads/main/Praveen_Dataset.csv"
        combos_df_url = "https://raw.githubusercontent.com/Dion-Chettiar/6611_Main/refs/heads/main/data.parquet"
        
        # Load CSV files
        sub_data = pd.read_csv(sub_optimizer_url)
        perf_data = pd.read_csv(performance_url)
        players_df = pd.read_csv(players_df_url)
        combos_df = pd.read_parquet(combos_df_url)
        
        # Clean and process data
        sub_data.columns = sub_data.columns.str.strip()
        perf_data.columns = perf_data.columns.str.strip()
        
        # Merge datasets
        merged_data = pd.merge(
            sub_data, 
            perf_data[['Player', 'Actual Impact']].rename(columns={'Actual Impact': 'Actual_Impact_Perf'}), 
            on='Player', 
            how='left'
        )
        
        # Calculate overperformance
        merged_data['Overperformance'] = merged_data['Impact'] - merged_data['Predicted Impact']
        
        # Rename columns for better display
        merged_data = merged_data.rename(columns={
            'Impact': 'Actual Impact',
            'Fatigue_Score': 'Fatigue Score',
            'Sub_Recommendation': 'Sub Recommendation',
            'Sub Early Probability': 'Sub Early Probability'
        })
        
        return merged_data, players_df, combos_df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Custom color palette for charts (modern dark theme)
DARK_COLORS = {
    'primary': '#3b82f6',      # Modern blue
    'secondary': '#8b5cf6',    # Purple
    'success': '#10b981',      # Emerald green
    'warning': '#f59e0b',      # Amber
    'danger': '#ef4444',       # Red
    'info': '#06b6d4',         # Cyan
    'dark': '#1f2937',         # Dark gray
    'light': '#f9fafb'         # Light gray
}

# Load data
data = load_data()

if not data.empty:
    # Dashboard header with modern styling
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">⚽ Soccer Analytics Dashboard</h1>
        <p style="color: #9ca3af; font-size: 1.2rem;">Advanced performance analysis with modern visualization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick stats with enhanced metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Players", len(data), delta=None)
    with col2:
        avg_fatigue = data['Fatigue Score'].mean()
        st.metric("Avg Fatigue Score", f"{avg_fatigue:.2f}", delta=f"{avg_fatigue-1.5:.2f}")
    with col3:
        high_fatigue = len(data[data['Fatigue Score'] > 2])
        st.metric("High Fatigue Players", high_fatigue, delta=None)
    with col4:
        sub_early_count = len(data[data['Sub Recommendation'] == 'Sub Early'])
        st.metric("Sub Early Recommendations", sub_early_count, delta=None)
    
    # Sidebar controls with modern styling
    st.sidebar.markdown("### 📊 Dashboard Controls")
    
    # Enhanced filtering options
    unique_recommendations = ['All'] + sorted(data['Sub Recommendation'].unique().tolist())
    unique_positions = ['All'] + sorted([pos.strip() for pos in ','.join(data['Position'].unique()).split(',') if pos.strip()])
    
    # Filters with modern styling
    top_n = st.sidebar.slider("Show Top N Players", 5, 50, 15)
    selected_recommendation = st.sidebar.selectbox("Filter by Recommendation", unique_recommendations)
    selected_position = st.sidebar.selectbox("Filter by Position", unique_positions)
    
    # Fatigue score filter
    fatigue_range = st.sidebar.slider(
        "Fatigue Score Range",
        float(data['Fatigue Score'].min()),
        float(data['Fatigue Score'].max()),
        (float(data['Fatigue Score'].min()), float(data['Fatigue Score'].max()))
    )
    
    # Apply filters
    filtered_data = data.copy()
    
    if selected_recommendation != 'All':
        filtered_data = filtered_data[filtered_data['Sub Recommendation'] == selected_recommendation]
    
    if selected_position != 'All':
        filtered_data = filtered_data[filtered_data['Position'].str.contains(selected_position, na=False)]
    
    filtered_data = filtered_data[
        (filtered_data['Fatigue Score'] >= fatigue_range[0]) & 
        (filtered_data['Fatigue Score'] <= fatigue_range[1])
    ]
    
    # Get top performers
    top_performers = filtered_data.nlargest(top_n, 'Overperformance')
    
    # Main dashboard layout
    col_left, col_right = st.columns([3, 1])
    
    with col_left:
        st.markdown("### 🎯 Player Overperformance Analysis")
        
        if not top_performers.empty:
            # Enhanced bar chart with modern colors
            fig = px.bar(
                top_performers,
                x='Overperformance',
                y='Player',
                orientation='h',
                color='Fatigue Score',
                color_continuous_scale=[
                    [0.0, DARK_COLORS['success']],     # Low fatigue - green
                    [0.5, DARK_COLORS['warning']],     # Medium fatigue - amber  
                    [1.0, DARK_COLORS['danger']]       # High fatigue - red
                ],
                title=f"Top {top_n} Overperforming Players",
                labels={'Overperformance': 'Overperformance Value', 'Player': 'Player Name'},
                hover_data=['Position', 'Minutes', 'Actual Impact', 'Sub Recommendation']
            )
            
            # Modern chart styling
            fig.update_layout(
                height=max(400, len(top_performers) * 25),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=DARK_COLORS['light'], size=12),
                title_font=dict(color=DARK_COLORS['light'], size=16),
                xaxis=dict(
                    gridcolor='rgba(75, 85, 99, 0.3)',
                    color=DARK_COLORS['light']
                ),
                yaxis=dict(
                    gridcolor='rgba(75, 85, 99, 0.3)',
                    color=DARK_COLORS['light']
                ),
                coloraxis_colorbar=dict(
                    title_font_color=DARK_COLORS['light'],
                    tickfont_color=DARK_COLORS['light']
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No players found matching the current filters.")
    
    with col_right:
        st.markdown("### 🚨 Featured Player")
        
        if not filtered_data.empty:
            highest_fatigue_player = filtered_data.loc[filtered_data['Fatigue Score'].idxmax()]
            
            # Determine fatigue level and color
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
    
    # Enhanced data table
    if not filtered_data.empty:
        st.markdown("### 📋 Player Performance Analysis")
        
        # Sorting controls
        sort_col1, sort_col2 = st.columns(2)
        with sort_col1:
            sort_column = st.selectbox(
                "Sort by:", 
                ['Overperformance', 'Fatigue Score', 'Minutes', 'Actual Impact', 'Sub Early Probability']
            )
        with sort_col2:
            sort_order = st.radio("Order:", ['Descending', 'Ascending'], horizontal=True)
        
        # Sort and display data
        ascending = sort_order == 'Ascending'
        display_data = filtered_data.sort_values(sort_column, ascending=ascending)
        
        # Format data for better display
        display_columns = [
            'Player', 'Position', 'Minutes', 'Actual Impact', 
            'Predicted Impact', 'Overperformance', 'Fatigue Score', 
            'Sub Recommendation', 'Sub Early Probability'
        ]
        
        formatted_data = display_data[display_columns].copy()
        
        # Numeric formatting
        numeric_columns = ['Actual Impact', 'Predicted Impact', 'Overperformance']
        for col in numeric_columns:
            formatted_data[col] = formatted_data[col].apply(lambda x: f"{x:.4f}")
        
        formatted_data['Fatigue Score'] = formatted_data['Fatigue Score'].apply(lambda x: f"{x:.2f}")
        formatted_data['Minutes'] = formatted_data['Minutes'].apply(lambda x: f"{x:,.0f}")
        formatted_data['Sub Early Probability'] = formatted_data['Sub Early Probability'].apply(lambda x: f"{x:.3f}")
        
        # Display the enhanced table
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
        
        # Export functionality
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

        # Enhanced summary charts
        st.markdown("### 📊 Analytics Summary")
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            # Recommendation distribution pie chart
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
            
            # Style the pie chart
            fig_pie.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=DARK_COLORS['light']),
                title_font=dict(color=DARK_COLORS['light'], size=14)
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with summary_col2:
            # Fatigue score distribution histogram
            fig_hist = px.histogram(
                filtered_data,
                x='Fatigue Score',
                nbins=20,
                title="Fatigue Score Distribution",
                color_discrete_sequence=[DARK_COLORS['primary']]
            )
            
            # Add reference lines
            fig_hist.add_vline(x=1, line_dash="dash", line_color=DARK_COLORS['warning'], 
                              annotation_text="Moderate Fatigue")
            fig_hist.add_vline(x=2, line_dash="dash", line_color=DARK_COLORS['danger'], 
                              annotation_text="High Fatigue")
            
            # Style the histogram
            fig_hist.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=DARK_COLORS['light']),
                title_font=dict(color=DARK_COLORS['light'], size=14),
                xaxis=dict(
                    gridcolor='rgba(75, 85, 99, 0.3)',
                    color=DARK_COLORS['light']
                ),
                yaxis=dict(
                    gridcolor='rgba(75, 85, 99, 0.3)',
                    color=DARK_COLORS['light']
                )
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)

else:
    st.error("❌ No data available. Please check your GitHub CSV files.")


        # --- Helper Function ---
def get_image_url(player_name):
    row = players_df[players_df['Player_std'] == player_name]
    if not row.empty:
        return row.iloc[0]['image url']
    # fallback icon if not found
    return "https://cdn-icons-png.flaticon.com/512/1055/1055687.png"

# --- Streamlit UI ---
st.title("Optimal Player Trio Combinations")

selected_row = st.selectbox(
    "Select a trio to visualize",
    combos_df.index,
    format_func=lambda idx: 
        f"{combos_df.loc[idx, 'Player 1']} + {combos_df.loc[idx, 'Player 2']} + {combos_df.loc[idx, 'Player 3']}"
)

row = combos_df.loc[selected_row]
p1, p2, p3 = row['Player 1'], row['Player 2'], row['Player 3']
img1, img2, img3 = get_image_url(p1), get_image_url(p2), get_image_url(p3)

nodes = {
    p1: (0.5, 1.0),
    p2: (0.15, 0.2),
    p3: (0.85, 0.2)
}
images = {p1: img1, p2: img2, p3: img3}
labels = {
    p1: f"<b>{p1} ({row['Position 1']})</b>",
    p2: f"{p2} ({row['Position 2']})",
    p3: f"{p3} ({row['Position 3']})"
}

fig = go.Figure()

# Draw arrows
fig.add_annotation(x=nodes[p1][0], y=nodes[p1][1], ax=nodes[p2][0], ay=nodes[p2][1],
    showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor="#1e293b")
fig.add_annotation(x=nodes[p1][0], y=nodes[p1][1], ax=nodes[p3][0], ay=nodes[p3][1],
    showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor="#1e293b")
fig.add_annotation(x=nodes[p2][0], y=nodes[p2][1], ax=nodes[p3][0], ay=nodes[p3][1],
    showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor="#1e293b")

# Add images and labels
for name, (x, y) in nodes.items():
    fig.add_layout_image(
        dict(
            source=images[name],
            xref="x", yref="y",
            x=x-0.08, y=y+0.06,
            sizex=0.16, sizey=0.16,
            xanchor="center", yanchor="middle",
            layer="above"
        )
    )
    fig.add_trace(go.Scatter(
        x=[x], y=[y-0.13], text=[labels[name]], mode="text",
        textfont=dict(color="#1e293b", size=16)
    ))

fig.update_xaxes(visible=False, range=[0, 1])
fig.update_yaxes(visible=False, range=[0, 1.15])
fig.update_layout(
    width=350, height=400,
    plot_bgcolor="#f8fafc",
    paper_bgcolor="#f8fafc",
    margin=dict(l=0, r=0, t=10, b=0),
    showlegend=False
)

st.plotly_chart(fig)
