import streamlit as st
import os
import pandas as pd
from backend.pipeline import AnalyticsPipeline
from backend.explainability import ExplainabilityEngine
from backend.report_generator import ReportGenerator
from chatbot.engine import ChatbotEngine

st.set_page_config(page_title="Report Generation", page_icon="", layout="wide")

from theme import apply_premium_sidebar
apply_premium_sidebar()


st.title("Executive Report Generation")
st.markdown("Compile your data, model performance, SHAP explainability, and AI insights into a professional PDF.")

pipeline_exists = os.path.exists("models/trained/analytics_pipeline.pkl")

if not pipeline_exists:
    st.warning("No trained model found. Please train a model in the AutoML module first.")
    st.stop()

if st.button("Generate Final PDF Report", type="primary"):
    with st.spinner("Compiling insights and rendering PDF..."):
        try:
            # 1. Load Pipeline
            pipeline_dict = AnalyticsPipeline.load("models/trained/analytics_pipeline.pkl")
            
            # 2. Generate SHAP Plot
            st.info("Calculating SHAP Explainability (this might take a moment for complex models)...")
            shap_engine = ExplainabilityEngine()
            
            # Use training data for background
            df = st.session_state.get('raw_data')
            target = pipeline_dict.get('target_column')
            if df is not None and target in df.columns:
                X_train = df.drop(columns=[target])
                # Limit size for speed in SHAP
                X_sample = X_train.sample(min(100, len(X_train)), random_state=42)
                shap_engine.fit_explainer(X_sample)
                shap_path = shap_engine.plot_summary()
            else:
                shap_path = None
                
            # 3. Generate AI Insights
            st.info("Consulting AI Copilot for Business Insights...")
            api_key = os.getenv("GEMINI_API_KEY")
            insights = None
            if api_key:
                bot = ChatbotEngine(api_key=api_key)
                insights = bot.generate_insights(
                    pipeline_dict.get('metadata', {}), 
                    pipeline_dict.get('leaderboard', pd.DataFrame()), 
                    pipeline_dict.get('feature_importance', pd.DataFrame())
                )
                
            # 4. Generate PDF
            st.info("Rendering PDF Document...")
            reporter = ReportGenerator()
            pdf_path = reporter.generate(
                insights_text=insights,
                leaderboard_df=pipeline_dict.get('leaderboard'),
                shap_image_path=shap_path
            )
            
            st.success(f"Report successfully generated at: `{pdf_path}`")
            
            # Download Button
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="Download Executive Report (PDF)",
                    data=pdf_file,
                    file_name="Business_Report.pdf",
                    mime="application/pdf"
                )
                
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
