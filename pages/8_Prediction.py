import streamlit as st
import pandas as pd
import numpy as np
import os
from backend.prediction import PredictionEngine

st.set_page_config(page_title="Model Prediction", page_icon="", layout="wide")

from theme import apply_premium_sidebar
apply_premium_sidebar()


st.title("Model Prediction Engine")
st.markdown("Use your trained ensemble model to predict on new, unseen data.")

pipeline_exists = os.path.exists("models/trained/analytics_pipeline.pkl")

if not pipeline_exists:
    st.warning("No trained model found. Please train a model in the AutoML module first.")
    st.stop()

tab1, tab2 = st.tabs(["✍️ Single Prediction (Manual Entry)", "Batch Prediction (File Upload)"])

with tab1:
    st.subheader("Manual Data Entry")
    st.markdown("Enter feature values manually to generate a prediction on the fly.")
    
    if 'raw_data' in st.session_state:
        df = st.session_state['raw_data']
    else:
        try:
            df = pd.read_csv("data/raw_dataset.csv")
            st.session_state['raw_data'] = df
        except:
            st.error("Original dataset missing, cannot construct input form.")
            st.stop()
            
    try:
        with open("data/target.txt", "r") as f:
            target_col = f.read().strip()
    except:
        target_col = None

    features = [c for c in df.columns if c != target_col]
    
    with st.form("single_prediction_form"):
        st.markdown("##### Input Features")
        cols = st.columns(3)
        
        user_inputs = {}
        for i, feat in enumerate(features):
            col_idx = i % 3
            if pd.api.types.is_numeric_dtype(df[feat]):
                min_val = float(df[feat].min())
                max_val = float(df[feat].max())
                mean_val = float(df[feat].mean())
                
                # Check if it's integer-like
                if pd.api.types.is_integer_dtype(df[feat]):
                    user_inputs[feat] = cols[col_idx].number_input(f"{feat}", min_value=int(min_val), max_value=int(max_val), value=int(mean_val), step=1)
                else:
                    user_inputs[feat] = cols[col_idx].number_input(f"{feat}", min_value=min_val, max_value=max_val, value=mean_val)
            else:
                unique_vals = df[feat].dropna().unique().tolist()
                user_inputs[feat] = cols[col_idx].selectbox(f"{feat}", options=unique_vals)
                
        submit = st.form_submit_button("Generate Prediction", type="primary")
        
        if submit:
            with st.spinner("Scoring..."):
                input_df = pd.DataFrame([user_inputs])
                try:
                    engine = PredictionEngine()
                    res = engine.predict(input_df)
                    st.success("Prediction Generated Successfully!")
                    st.dataframe(res, use_container_width=True)
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

with tab2:
    st.subheader("Batch Prediction")
    st.markdown("Upload a CSV or Excel file to score multiple records at once.")
    
    upload_new = st.file_uploader("Upload Data for Scoring", type=["csv", "xlsx"])

    if upload_new is not None:
        try:
            if upload_new.name.endswith('.csv'):
                new_df = pd.read_csv(upload_new)
            else:
                new_df = pd.read_excel(upload_new)
                
            st.success("New dataset loaded successfully!")
            st.dataframe(new_df.head())
            
            if st.button("Generate Batch Predictions", type="primary"):
                with st.spinner("Scoring dataset..."):
                    engine = PredictionEngine()
                    results_df = engine.predict(new_df)
                    
                    # Merge predictions with original data for easy download
                    final_df = pd.concat([new_df, results_df], axis=1)
                    
                    st.subheader("Prediction Results")
                    st.dataframe(final_df.head(15))
                    
                    # Provide download link
                    csv = final_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Full Predictions as CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv",
                    )
                    
        except Exception as e:
            st.error(f"Error processing predictions: {str(e)}")
