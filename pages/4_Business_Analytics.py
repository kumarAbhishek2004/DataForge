import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Business Dashboard", page_icon="", layout="wide")

from theme import apply_premium_sidebar
apply_premium_sidebar()


# Inject Custom CSS for Premium, Colorful KPI Cards
st.markdown("""
    <style>
    .kpi-container {
        display: flex;
        justify-content: space-between;
        gap: 1.5rem;
        margin-bottom: 2.5rem;
    }
    .kpi-card {
        flex: 1;
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border-left: 6px solid #3b82f6; /* Default Blue */
        text-align: left;
        transition: transform 0.2s ease-in-out;
    }
    .kpi-card:hover {
        transform: translateY(-5px);
    }
    .kpi-card.green { border-left-color: #10b981; }
    .kpi-card.purple { border-left-color: #8b5cf6; }
    .kpi-card.red { border-left-color: #ef4444; }
    
    .kpi-title {
        color: #64748b;
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .kpi-value {
        color: #0f172a;
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 0.5rem;
    }
    .kpi-trend {
        font-size: 0.9rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }
    .trend-up { color: #10b981; }
    .trend-down { color: #ef4444; }
    .trend-neutral { color: #64748b; }
    </style>
""", unsafe_allow_html=True)

st.title("Advanced Business Insights")

if 'raw_data' not in st.session_state:
    try:
        df = pd.read_csv("data/raw_dataset.csv")
        st.session_state['raw_data'] = df
    except:
        st.warning("No dataset found. Please upload a dataset in the Upload module first.")
        st.stop()
        
df = st.session_state['raw_data']

try:
    with open("data/target.txt", "r") as f:
        target_col = f.read().strip()
except:
    target_col = None

num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

# --- KPI Section ---
st.markdown("### Top Level Metrics")

kpi_html = '<div class="kpi-container">'

# 1. Volume Card (Blue)
kpi_html += f"""
<div class="kpi-card">
    <div class="kpi-title">Total Records</div>
    <div class="kpi-value">{len(df):,}</div>
    <div class="kpi-trend trend-up">↑ Verified Volume</div>
</div>
"""

# --- Smart Metric Selection ---
# Filter out likely ID or index columns for the KPI cards
smart_num_cols = [c for c in num_cols if not any(x in c.lower() for x in ['id', 'row', 'index', 'key']) and c != target_col]

with st.expander("Customize KPI Metrics"):
    c1, c2 = st.columns(2)
    p_metric = c1.selectbox("Primary Metric (Average)", smart_num_cols, index=0 if len(smart_num_cols) > 0 else None)
    s_metric = c2.selectbox("Secondary Metric (Total)", smart_num_cols, index=1 if len(smart_num_cols) > 1 else 0)

# 2. Primary Metric (Green)
if p_metric:
    val = df[p_metric].mean()
    formatted = f"${val:,.2f}" if any(w in p_metric.lower() for w in ["amount", "price", "balance", "salary"]) else f"{val:,.2f}"
    kpi_html += f"""
<div class="kpi-card green">
    <div class="kpi-title">Average {p_metric[:12]}</div>
    <div class="kpi-value">{formatted}</div>
    <div class="kpi-trend trend-up">↑ Primary Metric</div>
</div>
"""

# 3. Secondary Metric (Purple)
if s_metric:
    val2 = df[s_metric].sum()
    formatted2 = f"${val2:,.0f}" if any(w in s_metric.lower() for w in ["amount", "price", "balance", "salary"]) else f"{val2:,.0f}"
    kpi_html += f"""
<div class="kpi-card purple">
    <div class="kpi-title">Total {s_metric[:15]}</div>
    <div class="kpi-value">{formatted2}</div>
    <div class="kpi-trend trend-neutral">→ Aggregated</div>
</div>
"""

# 4. Target Metric (Red)
target_val = "N/A"
if target_col in df.columns:
    if df[target_col].nunique() == 2:
        rate = (df[target_col].value_counts(normalize=True).iloc[1] * 100).round(2)
        target_val = f"{rate}%"
    else:
        target_val = f"{df[target_col].mean():.2f}" if target_col in num_cols else f"{df[target_col].nunique()} Types"

kpi_html += f"""
<div class="kpi-card red">
    <div class="kpi-title">Target ({str(target_col)[:10]})</div>
    <div class="kpi-value">{target_val}</div>
    <div class="kpi-trend trend-down">Optimization Goal</div>
</div>
</div>
"""

st.markdown(kpi_html, unsafe_allow_html=True)

# --- Advanced Visualizations ---
st.markdown("### Deep Dive Analytics")

plot_type = st.selectbox("Select Interactive Plot Type", [
    "Bar Chart (Category vs Value)",
    "Pie Chart (Proportions)",
    "Histogram (Single Distribution)",
    "LM Plot (Trendline Scatter)",
    "Correlation Heatmap",
    "Violin Plot (Distributions)",
    "3D Scatter Plot",
    "Hierarchical Treemap"
])

if plot_type == "Bar Chart (Category vs Value)":
    st.markdown("#### Categorical Aggregation")
    if cat_cols and num_cols:
        c1, c2 = st.columns(2)
        x_col = c1.selectbox("Category (X-Axis)", cat_cols, key="bar_x")
        y_col = c2.selectbox("Value (Y-Axis)", num_cols, key="bar_y")
        fig = px.histogram(df, x=x_col, y=y_col, histfunc="sum", template="plotly_white")
        fig.update_layout(title=f"Total {y_col} by {x_col}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Need categorical and numerical columns for a Bar Chart.")

elif plot_type == "Pie Chart (Proportions)":
    st.markdown("#### Category Proportions")
    if cat_cols:
        c1, c2 = st.columns(2)
        cat_col = c1.selectbox("Slice Category", cat_cols, key="pie_c")
        val_mode = c2.radio("Slice Size", ["Count", "Sum of a Numeric Column"])
        
        if val_mode == "Count":
            fig = px.pie(df, names=cat_col, title=f"Proportion of {cat_col} (Count)", template="plotly_white")
        else:
            if num_cols:
                num_col = st.selectbox("Select Numeric Column to Sum", num_cols)
                fig = px.pie(df, names=cat_col, values=num_col, title=f"Proportion of Total {num_col} by {cat_col}", template="plotly_white")
            else:
                st.warning("No numeric columns available.")
                fig = None
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Need at least one categorical column for a Pie Chart.")

elif plot_type == "Histogram (Single Distribution)":
    st.markdown("#### Simple Data Distribution")
    if num_cols:
        c1, c2 = st.columns(2)
        x_col = c1.selectbox("Numeric Column", num_cols, key="hist_x")
        color_col = c2.selectbox("Color By Category (Optional)", ["None"] + cat_cols)
        
        c = None if color_col == "None" else color_col
        fig = px.histogram(df, x=x_col, color=c, marginal="box", template="plotly_white")
        fig.update_layout(title=f"Distribution of {x_col}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Need at least one numerical column.")

elif plot_type == "LM Plot (Trendline Scatter)":
    st.markdown("#### Linear Relationship & Scatter Matrix")
    if len(num_cols) >= 2:
        c1, c2 = st.columns(2)
        x_ax = c1.selectbox("X-Axis", num_cols, index=0, key="lm_x")
        y_ax = c2.selectbox("Y-Axis", num_cols, index=1 if len(num_cols)>1 else 0, key="lm_y")
        color_split = target_col if target_col in cat_cols else (cat_cols[0] if cat_cols else None)
        
        try:
            fig = px.scatter(df, x=x_ax, y=y_ax, color=color_split, trendline="ols", template="plotly_white", opacity=0.6)
            fig.update_layout(title=f"Trend between {x_ax} and {y_ax}")
        except ImportError:
            # Fallback if statsmodels is not installed
            fig = px.scatter(df, x=x_ax, y=y_ax, color=color_split, marginal_x="box", marginal_y="histogram", template="plotly_white", opacity=0.6)
            fig.update_layout(title=f"Joint Distribution of {x_ax} and {y_ax} (Install 'statsmodels' for OLS trendlines)")
            
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Need at least 2 numerical columns for a scatter plot.")

elif plot_type == "Correlation Heatmap":
    st.markdown("#### Feature Correlation Matrix")
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
        fig.update_layout(title="Pearson Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Need at least 2 numerical columns for a heatmap.")

elif plot_type == "Violin Plot (Distributions)":
    st.markdown("#### Distribution Densities & Outliers")
    if num_cols and cat_cols:
        c1, c2 = st.columns(2)
        y_col = c1.selectbox("Numerical Value", num_cols, key="vio_y")
        color_col = c2.selectbox("Split By Category", cat_cols, key="vio_c")
        fig = px.violin(df, y=y_col, color=color_col, box=True, points="all", template="plotly_white")
        fig.update_layout(title=f"Violin Plot of {y_col} by {color_col}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Need both numerical and categorical columns for a Violin Plot.")

elif plot_type == "3D Scatter Plot":
    st.markdown("#### Complex Multi-Dimensional Relationships")
    if len(num_cols) >= 3:
        c1, c2, c3, c4 = st.columns(4)
        x_3d = c1.selectbox("X-Axis", num_cols, index=0, key="3dx")
        y_3d = c2.selectbox("Y-Axis", num_cols, index=1, key="3dy")
        z_3d = c3.selectbox("Z-Axis", num_cols, index=2, key="3dz")
        c_3d = c4.selectbox("Color Coding", cat_cols if cat_cols else num_cols, key="3dc")
        fig = px.scatter_3d(df, x=x_3d, y=y_3d, z=z_3d, color=c_3d, opacity=0.7, color_continuous_scale="Turbo")
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Need at least 3 numerical columns for 3D analysis.")

elif plot_type == "Hierarchical Treemap":
    st.markdown("#### Categorical Hierarchy Map")
    if cat_cols and len(cat_cols) >= 2 and num_cols:
        c1, c2, c3 = st.columns(3)
        cat1 = c1.selectbox("Primary Category", cat_cols, index=0)
        cat2 = c2.selectbox("Secondary Category", cat_cols, index=1 if len(cat_cols) > 1 else 0)
        metric = c3.selectbox("Size Metric", num_cols, index=0)
        fig = px.treemap(df.dropna(subset=[cat1, cat2, metric]), path=[cat1, cat2], values=metric, color=metric, color_continuous_scale="Blues")
        fig.update_layout(title=f"Treemap: {metric} by {cat1} & {cat2}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough categorical and numerical columns to generate a Treemap.")
