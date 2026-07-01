import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os
import shap

st.set_page_config(page_title="Model Explainability", page_icon="", layout="wide")

from theme import apply_premium_sidebar
apply_premium_sidebar()

st.title("Advanced Model Explainability (XAI)")
st.markdown("Understand **how** and **why** the Machine Learning model is making its decisions using SHAP (SHapley Additive exPlanations).")

if not os.path.exists("models/trained/analytics_pipeline.pkl"):
    st.warning("No trained model found. Please train a model in the AutoML module first.")
    st.stop()

from backend.pipeline import AnalyticsPipeline
pipeline_dict = AnalyticsPipeline.load("models/trained/analytics_pipeline.pkl")
best_model_name = pipeline_dict.get('best_model_name', 'Unknown')

st.success(f"Successfully loaded winning model: **{best_model_name}**")

# Get Feature Importance
fi_df = pipeline_dict.get('feature_importance')

tab1, tab2, tab3 = st.tabs(["Global Feature Importance", "SHAP Summary Plot", "SHAP Dependence"])

with tab1:
    st.markdown("### Global Feature Importance")
    st.info("This shows the aggregate impact of each feature on the model's predictions.")
    if fi_df is not None and not isinstance(fi_df, dict) and not fi_df.empty:
        fi_plot_df = fi_df.sort_values(by="Importance", ascending=True).tail(15)
        fig = px.bar(fi_plot_df, x="Importance", y="Feature", orientation='h', color="Importance", color_continuous_scale="Viridis", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Feature importance is not available for this type of model.")

with tab2:
    st.markdown("### SHAP Summary Plot")
    st.write("The SHAP summary plot combines feature importance with feature effects. Each dot represents a row in your dataset.")
    
    if st.button("Generate SHAP Summary Plot"):
        with st.spinner("Calculating SHAP values..."):
            from backend.explainability import ExplainabilityEngine
            df = st.session_state.get('raw_data')
            target = pipeline_dict.get('target_column')
            
            if df is not None and target in df.columns:
                X_train = df.drop(columns=[target])
                # Downsample for speed
                X_sample = X_train.sample(min(200, len(X_train)), random_state=42)
                
                engine = ExplainabilityEngine()
                try:
                    engine.fit_explainer(X_sample)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(engine.shap_values, X_sample, show=False)
                    # Adjust colors for dark theme
                    fig.patch.set_facecolor('#0E1117')
                    ax.set_facecolor('#0E1117')
                    ax.tick_params(colors='white')
                    ax.xaxis.label.set_color('white')
                    ax.yaxis.label.set_color('white')
                    
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating SHAP values for this model architecture: {e}")
            else:
                st.error("Raw data not found in session state.")

with tab3:
    st.markdown("### SHAP Dependence Plot")
    st.write("Understand the interaction effect between a specific feature and the target variable.")
    
    if 'raw_data' in st.session_state and pipeline_dict.get('target_column') in st.session_state['raw_data'].columns:
        features = st.session_state['raw_data'].drop(columns=[pipeline_dict.get('target_column')]).columns.tolist()
        feat_to_plot = st.selectbox("Select Feature to Analyze", features)
        
        if st.button("Generate Dependence Plot"):
            with st.spinner(f"Plotting dependence for {feat_to_plot}..."):
                from backend.explainability import ExplainabilityEngine
                df = st.session_state.get('raw_data')
                target = pipeline_dict.get('target_column')
                X_train = df.drop(columns=[target])
                X_sample = X_train.sample(min(200, len(X_train)), random_state=42)
                
                engine = ExplainabilityEngine()
                try:
                    engine.fit_explainer(X_sample)
                    
                    fig, ax = plt.subplots(figsize=(8, 5))
                    # Handle multi-class vs binary shap values
                    shap_vals = engine.shap_values[1] if isinstance(engine.shap_values, list) else engine.shap_values
                    shap.dependence_plot(feat_to_plot, shap_vals, X_sample, show=False, ax=ax)
                    
                    fig.patch.set_facecolor('#0E1117')
                    ax.set_facecolor('#0E1117')
                    ax.tick_params(colors='white')
                    ax.xaxis.label.set_color('white')
                    ax.yaxis.label.set_color('white')
                    
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Could not generate dependence plot: {e}")
