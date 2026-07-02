import streamlit as st
import pandas as pd
import os
import json
import threading
import time
import plotly.express as px
from backend.automl import AutoML
from backend.pipeline import AnalyticsPipeline

st.set_page_config(page_title="AutoML Training", page_icon="", layout="wide")

from theme import apply_premium_sidebar
apply_premium_sidebar()


STATUS_FILE = "data/training_status.json"

def get_training_status():
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r") as f:
                return json.load(f)
        except:
            return {"status": "idle"}
    return {"status": "idle"}

def set_training_status(status_dict):
    os.makedirs("data", exist_ok=True)
    with open(STATUS_FILE, "w") as f:
        json.dump(status_dict, f)

def background_train_task(df, target_col, selected_models=None):
    automl = AutoML()
    
    def update_progress(current, total, current_model):
        set_training_status({
            "status": "running",
            "current": current,
            "total": total,
            "model": current_model
        })
        
    try:
        # Train the models asynchronously 
        results = automl.train(df, target=target_col, progress_callback=update_progress, selected_models=selected_models)
        set_training_status({"status": "completed"})
    except Exception as e:
        set_training_status({"status": "error", "error_message": str(e)})


st.title("AutoML Engine")

if 'raw_data' not in st.session_state:
    try:
        df = pd.read_csv("data/raw_dataset.csv")
        st.session_state['raw_data'] = df
    except:
        st.warning("No dataset found. Please upload a dataset first.")
        st.stop()

df = st.session_state['raw_data']

try:
    with open("data/target.txt", "r") as f:
        target_col = f.read().strip()
except:
    st.warning("No target variable selected. Please go to the Upload module to select one.")
    st.stop()

status_data = get_training_status()
current_status = status_data.get("status", "idle")

st.markdown(f"**Target Variable:** `{target_col}`")

if current_status in ["idle", "error", "running"]:
    if current_status in ["error", "running"]:
        st.error(f"Previous training failed: {status_data.get('error_message')}")
        
    st.info("Click the button below to initiate the autonomous machine learning pipeline. The process will run completely in the background, allowing you to freely navigate to the Dashboard or EDA pages while it crunches the data!")
    
    # --- Model Selection Logic ---
    temp_automl = AutoML()
    prob_type = temp_automl.detect_problem_type(df, target_col)
    
    if prob_type == "classification":
        from backend.registry import ClassificationRegistry
        available_models = list(ClassificationRegistry().get_models().keys())
    else:
        from backend.registry import RegressionRegistry
        available_models = list(RegressionRegistry().get_models().keys())
        
    selected_models = st.multiselect(
        "Select Models to Train in the Ensemble", 
        options=available_models, 
        default=available_models,
        help="Remove models to speed up training time. Keeping more models increases the chance of finding a better performing algorithm."
    )
    
    if not selected_models:
        st.warning("Please select at least one model to train.")
    else:
        if st.button("Start Training", type="primary"):
            
            st.info("🚀 Training models... Please do not close or navigate away from this page.")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress_sync(current, total, current_model):
                fraction = min(current / max(total, 1), 1.0)
                progress_bar.progress(fraction)
                status_text.write(f"Training Model {current + 1} of {total}: **{current_model}**")
            
            try:
                with st.spinner("Crunching data..."):
                    automl = AutoML()
                    results = automl.train(df, target=target_col, progress_callback=update_progress_sync, selected_models=selected_models)
                set_training_status({"status": "completed"})
                st.success("Training Complete!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                set_training_status({"status": "error", "error_message": str(e)})
                st.error(f"Training failed: {e}")

elif current_status == "completed":
    st.success("Training Complete!")
    
    if st.button("🔄 Reset & Train Again"):
        set_training_status({"status": "idle"})
        st.rerun()
        
    try:
        # Load the completed pipeline from disk
        pipeline_dict = AnalyticsPipeline.load("models/trained/analytics_pipeline.pkl")
        
        st.subheader("🏆 Model Leaderboard")
        st.dataframe(pipeline_dict["leaderboard"], use_container_width=True)
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🥇 Best Model")
            st.metric("Model Name", pipeline_dict["best_model_name"])
            st.metric("Score", f"{pipeline_dict['best_score']:.4f}")
            
        with col2:
            st.subheader("Feature Importance")
            fi_df = pipeline_dict.get("feature_importance")
            if fi_df is not None and not isinstance(fi_df, dict) and not fi_df.empty:
                fi_df = fi_df.sort_values(by="Importance", ascending=True).tail(10)
                fig = px.bar(fi_df, x="Importance", y="Feature", orientation='h', color="Importance", color_continuous_scale="Viridis")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature importance not available for this model type.")
                
        st.success("Pipeline perfectly packaged and saved at: `models/trained/analytics_pipeline.pkl`")
        
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
