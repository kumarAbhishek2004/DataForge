import streamlit as st
import pandas as pd
import json

st.set_page_config(page_title="Feature Engineering", page_icon="", layout="wide")

from theme import apply_premium_sidebar
apply_premium_sidebar()

st.title("Feature Engineering Configuration")
st.markdown("Configure how the AutoML pipeline will process your data before training models.")

if 'raw_data' not in st.session_state:
    st.warning("Please upload a dataset first.")
    st.stop()

st.markdown("### 1. Missing Value Treatment")
missing_strategy_num = st.selectbox("Numerical Imputation Strategy", ["Mean", "Median", "Mode", "KNN Imputer"])
missing_strategy_cat = st.selectbox("Categorical Imputation Strategy", ["Most Frequent", "Constant Value (e.g., 'Unknown')"])

st.markdown("### 2. Feature Scaling & Transformation")
scaling_method = st.selectbox("Scaling Method (Numerical Features)", ["StandardScaler (Z-Score)", "MinMaxScaler (0 to 1)", "RobustScaler (Ignores Outliers)"])

st.markdown("### 3. Categorical Encoding")
encoding_method = st.selectbox("Encoding Strategy (Categorical Features)", ["Target Encoding", "One-Hot Encoding", "Ordinal Encoding"])

st.markdown("### 4. Advanced Features")
handle_outliers = st.checkbox("Automatically Clip Outliers (1st & 99th Percentiles)", value=True)
create_polynomial = st.checkbox("Create Polynomial Features (Degree 2) - Warning: Increases memory", value=False)

if st.button("Save Engineering Pipeline Configuration", type="primary"):
    config = {
        "missing_num": missing_strategy_num,
        "missing_cat": missing_strategy_cat,
        "scaling": scaling_method,
        "encoding": encoding_method,
        "outliers": handle_outliers,
        "polynomial": create_polynomial
    }
    with open("data/engineering_config.json", "w") as f:
        json.dump(config, f)
    st.success("Configuration saved! The AutoML engine will utilize these settings.")
