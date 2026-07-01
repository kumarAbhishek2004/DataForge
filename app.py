import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="DataForge",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

from theme import apply_premium_sidebar
apply_premium_sidebar()


# Custom CSS for Premium Design
st.markdown("""
    <style>
    /* Dark sidebar styling to match the reference */
    [data-testid="stSidebar"] {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    [data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }
    
    /* Clean Topbar */
    header {background-color: transparent;}
    
    /* KPI Card styling */
    .kpi-card {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #E2E8F0;
        text-align: center;
        margin-bottom: 20px;
    }
    .kpi-title {
        color: #64748B;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .kpi-value {
        color: #0F172A;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .kpi-trend-up {
        color: #10B981;
        font-size: 14px;
        font-weight: 600;
    }
    .kpi-trend-down {
        color: #EF4444;
        font-size: 14px;
        font-weight: 600;
    }
    
    /* Main Content Background */
    .stApp {
        background-color: transparent;
    }
    
    /* Hide Streamlit footer */
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title("DataForge")
st.markdown("### Automated Insights Platform")

st.info("Welcome to DataForge AI! Please select a module from the sidebar to begin. Start by uploading a dataset in the **Upload** section.")

st.markdown("""
**Available Modules:**
- **Upload**: Ingest datasets (CSV/Excel) and define target variables.
- **EDA**: Automatically visualize distributions, correlations, and data quality.
- **Dashboard**: Interactive Plotly-based business intelligence KPIs.
- **AutoML**: Train and tune multiple Classification or Regression models autonomously.
- **Prediction**: Score new data using your winning pipeline.
- **Chatbot**: Chat with your data and models using Gemini AI.
- **Report**: Export a comprehensive PDF of your analytics journey.
""")
