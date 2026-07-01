import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Advanced EDA", page_icon="", layout="wide")

from theme import apply_premium_sidebar
apply_premium_sidebar()


st.title("Advanced Exploratory Data Analysis")

if 'raw_data' not in st.session_state:
    try:
        df = pd.read_csv("data/raw_dataset.csv")
        st.session_state['raw_data'] = df
    except:
        st.warning("No dataset found. Please upload a dataset in the Upload module first.")
        st.stop()
        
df = st.session_state['raw_data']

num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

tab1, tab2, tab3 = st.tabs(["📋 Data Quality & Profiling", "Univariate Analysis", "🔗 Feature Relationships"])

with tab1:
    st.markdown("### Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{df.shape[0]:,}")
    col2.metric("Total Features", f"{df.shape[1]:,}")
    col3.metric("Missing Values", f"{df.isna().sum().sum():,}")
    col4.metric("Duplicate Rows", f"{df.duplicated().sum():,}")

    st.markdown("---")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("#### Detailed Feature Profiling")
        info_df = pd.DataFrame({
            "Data Type": df.dtypes.astype(str),
            "Non-Null Count": df.count(),
            "Missing Values": df.isna().sum(),
            "Missing %": (df.isna().sum() / len(df) * 100).round(2),
            "Unique Values": df.nunique()
        })
        st.dataframe(info_df, use_container_width=True)
    
    with c2:
        st.markdown("#### Missing Value Distribution")
        missing_df = info_df[info_df["Missing Values"] > 0].reset_index()
        if not missing_df.empty:
            missing_df.columns = ["Feature", "Data Type", "Non-Null", "Missing", "Missing %", "Unique"]
            fig_missing = px.bar(missing_df, x="Feature", y="Missing %", text="Missing %", color="Missing %", color_continuous_scale="Reds")
            fig_missing.update_traces(textposition='outside')
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("No missing values detected in the dataset!")

with tab2:
    st.markdown("### Single Feature Analysis (Univariate)")
    
    analysis_type = st.radio("Select Data Type to Analyze:", ["Numerical", "Categorical"], horizontal=True)
    
    if analysis_type == "Numerical" and num_cols:
        col = st.selectbox("Select Numerical Feature", num_cols)
        
        c1, c2 = st.columns([3, 1])
        with c1:
            plot_style = st.radio("Plot Style", ["Histogram & Box", "Density Contour", "Violin"], horizontal=True)
            use_log = st.checkbox("Log Scale (Y-axis)")
            
            if plot_style == "Histogram & Box":
                fig = px.histogram(df, x=col, marginal="box", color_discrete_sequence=['#3b82f6'], log_y=use_log)
            elif plot_style == "Density Contour":
                fig = px.density_contour(df, x=col, color_discrete_sequence=['#8b5cf6'], marginal_x="histogram")
            else:
                fig = px.violin(df, y=col, box=True, points="all", color_discrete_sequence=['#10b981'])
                if use_log:
                    fig.update_layout(yaxis_type="log")
            
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.markdown("#### Statistical Summary")
            desc = df[col].describe().to_frame()
            desc.loc['skewness'] = df[col].skew()
            desc.loc['kurtosis'] = df[col].kurtosis()
            st.dataframe(desc.round(3), use_container_width=True)
            
    elif analysis_type == "Categorical" and cat_cols:
        col = st.selectbox("Select Categorical Feature", cat_cols)
        
        top_n = st.slider("Top N Categories to Show", min_value=5, max_value=50, value=15)
        val_counts = df[col].value_counts().head(top_n).reset_index()
        val_counts.columns = [col, 'Count']
        
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(val_counts, x=col, y='Count', color='Count', color_continuous_scale="Purples", text='Count')
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig_pie = px.pie(val_counts, names=col, values='Count', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_pie, use_container_width=True)

with tab3:
    st.markdown("### Feature Interactions (Bivariate & Multivariate)")
    
    sub_tab1, sub_tab2 = st.tabs(["Correlation Engine", "Feature vs Feature"])
    
    with sub_tab1:
        if len(num_cols) > 1:
            st.markdown("#### Advanced Correlation Matrix")
            corr_method = st.selectbox("Correlation Method", ["pearson", "spearman", "kendall"])
            corr = df[num_cols].corr(method=corr_method)
            
            fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            fig_corr.update_layout(title=f"{corr_method.capitalize()} Correlation Heatmap")
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Find highly correlated pairs
            st.markdown("#### 🚨 Highly Correlated Feature Pairs")
            corr_pairs = corr.unstack().sort_values(kind="quicksort", ascending=False).drop_duplicates()
            strong_pairs = corr_pairs[(abs(corr_pairs) > 0.7) & (corr_pairs != 1.0)]
            if not strong_pairs.empty:
                st.dataframe(strong_pairs.to_frame(name="Correlation Coefficient").reset_index().rename(columns={"level_0": "Feature 1", "level_1": "Feature 2"}), use_container_width=True)
                st.info("High correlation (>0.7 or <-0.7) may indicate multicollinearity. Consider dropping one of the paired features during training.")
            else:
                st.success("No highly correlated pairs found. Features appear independent.")
        else:
            st.info("Need at least 2 numerical columns for correlation analysis.")
            
    with sub_tab2:
        st.markdown("#### Cross-Feature Comparison")
        if num_cols:
            col1, col2, col3 = st.columns(3)
            x_feat = col1.selectbox("X-Axis Feature", df.columns, index=0)
            y_feat = col2.selectbox("Y-Axis Feature", df.columns, index=1 if len(df.columns) > 1 else 0)
            color_feat = col3.selectbox("Color By (Optional)", ["None"] + df.columns.tolist())
            
            color_var = None if color_feat == "None" else color_feat
            
            # Smart Engine: Determine plot type based on data types
            if x_feat in num_cols and y_feat in num_cols:
                # Both numeric -> Scatter with trendline
                try:
                    fig_bi = px.scatter(df, x=x_feat, y=y_feat, color=color_var, trendline="ols", opacity=0.6, template="plotly_white")
                except:
                    fig_bi = px.scatter(df, x=x_feat, y=y_feat, color=color_var, opacity=0.6, template="plotly_white", marginal_x="box", marginal_y="box")
            elif x_feat in cat_cols and y_feat in num_cols:
                # Cat vs Num -> Box plot
                fig_bi = px.box(df, x=x_feat, y=y_feat, color=color_var, template="plotly_white")
            elif x_feat in num_cols and y_feat in cat_cols:
                # Num vs Cat -> Box plot horizontal
                fig_bi = px.box(df, x=y_feat, y=x_feat, color=color_var, template="plotly_white", orientation='h')
            else:
                # Cat vs Cat -> Heatmap of counts
                cross_tab = pd.crosstab(df[x_feat], df[y_feat]).reset_index().melt(id_vars=x_feat)
                fig_bi = px.density_heatmap(cross_tab, x=x_feat, y=y_feat, z="value", text_auto=True, color_continuous_scale="Viridis")
                
            fig_bi.update_layout(title=f"Relationship between {x_feat} and {y_feat}")
            st.plotly_chart(fig_bi, use_container_width=True)
