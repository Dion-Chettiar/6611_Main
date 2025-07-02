import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Page configuration ---
st.set_page_config(
    page_title="Soccer Analytics Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styling (as before, or customize as needed) ---
st.markdown("""
<style>
    .main { padding-top: 1rem; background-color: #0E1117; }
    h1 { color: #f9fafb !important; font-weight: 700 !important; }
    h2, h3 { color: #e5e7eb !important; }
</style>
""", unsafe_allow_html=True)

# --- Data loading function ---
@st.cache_data
def load_data():
    try:
        sub_optimizer_url = "https://raw.githubusercontent.com/Dion-Chettiar/Streamlit_Dashboard/refs/heads/main/sub_optimizer%202.csv"
        performance_url = "https://raw.githubusercontent.com/Dion-Chettiar/Streamlit_Dashboard/refs/heads/main/Performance_Dropoff_Per_Player.csv"
        players_df_url = "https://raw.githubusercontent.com/Dion-Chettiar/6611_Main/refs/heads/main/Praveen_Dataset.csv"
        combos_df_url = "https://raw.githubusercontent.com/Dion-Chettiar/6611_Main/refs/heads/main/data.parquet"

        # Load data
        sub_data = pd.read_csv(sub_optimizer_url)
        perf_data = pd.read_csv(performance_url)
        players_df = pd.read_csv(players_df_url)
        combos_df = pd.read_parquet(combos_df_url)

        # Clean columns
        sub_data.columns = sub_data.columns.str.strip()
        perf_data.columns = perf_data.columns.str.strip()
        players_df.columns = players_df.columns.str.strip()
        combos_df.columns = combos_df.columns.str.strip()

        # Merge for main dashboard
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

        return merged_data, players_df, combos_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# --- Load all dataframes ---
data, players_df, combos_df = load_data()

# --- Main dashboard analytics (as before) ---
if not data.empty:
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">⚽ Soccer Analytics Dashboard</h1>
        <p style="color: #9ca3af; font-size: 1.2rem;">Advanced performance analysis with modern visualization</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Players", len(data))
    with col2:
        avg_fatigue = data['Fatigue Score'].mean()
        st.metric("Avg Fatigue Score", f"{avg_fatigue:.2f}")
    with col3:
        high_fatigue = len(data[data['Fatigue Score'] > 2])
        st.metric("High Fatigue Players", high_fatigue)
    with col4:
        sub_early_count = len(data[data['Sub Recommendation'] == 'Sub Early'])
        st.metric("Sub Early Recommendations", sub_early_count)
    
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
                color_continuous_scale='RdYlGn_r',
                title=f"Top {top_n} Overperforming Players",
                labels={'Overperformance': 'Overperformance Value', 'Player': 'Player Name'},
                hover_data=['Position', 'Minutes', 'Actual Impact', 'Sub Recommendation']
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
                fatigue_emoji = "🔴"
                fatigue_status = "High Alert"
            elif fatigue_score > 1:
                fatigue_emoji = "🟡"
                fatigue_status = "Monitor"
            else:
                fatigue_emoji = "🟢"
                fatigue_status = "Good Condition"
            st.markdown(f"""
            <div style="padding: 1.5rem; border-radius: 12px; color: #f9fafb; text-align: center; border: 2px solid #3b82f6;">
                <h3 style="margin-bottom: 1rem;">{highest_fatigue_player['Player']}</h3>
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">{fatigue_emoji} {fatigue_score:.2f}</div>
                <p style="font-size: 1.2rem; margin-bottom: 1rem; font-weight: bold;">{fatigue_status}</p>
                <hr>
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

# --- TRIO COMBINATION VISUALIZATION SECTION ---
if not combos_df.empty and not players_df.empty:
    st.markdown("## Optimal Player Trio Combinations")
    def get_image_url(player_name):
        row = players_df[players_df['Player_std'] == player_name]
        if not row.empty:
            return row.iloc[0]['Image_url'] if 'Image_url' in row.columns else row.iloc[0]['image url']
        return "https://cdn-icons-png.flaticon.com/512/1055/1055687.png"

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
    fig.add_annotation(x=nodes[p1][0], y=nodes[p1][1], ax=nodes[p2][0], ay=nodes[p2][1],
        showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor="#1e293b")
    fig.add_annotation(x=nodes[p1][0], y=nodes[p1][1], ax=nodes[p3][0], ay=nodes[p3][1],
        showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor="#1e293b")
    fig.add_annotation(x=nodes[p2][0], y=nodes[p2][1], ax=nodes[p3][0], ay=nodes[p3][1],
        showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor="#1e293b")
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
