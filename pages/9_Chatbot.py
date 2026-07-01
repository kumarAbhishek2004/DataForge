import streamlit as st
import os
import pandas as pd
from chatbot.engine import ChatbotEngine
from backend.pipeline import AnalyticsPipeline

st.set_page_config(page_title="AI Copilot Chat", page_icon="", layout="wide")

from theme import apply_premium_sidebar
apply_premium_sidebar()


st.title("AI Copilot")
st.markdown("Ask anything about your dataset, your machine learning models, or request business insights.")

# Setup Gemini API Key
api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Leave blank to use the .env key.")

if not api_key and not os.getenv("GEMINI_API_KEY"):
    st.warning("Please enter your Gemini API Key in the sidebar or set it in your .env file to activate the Copilot.")
    st.stop()

# Initialize Chat Engine
try:
    bot = ChatbotEngine(api_key=api_key if api_key else None)
except Exception as e:
    st.error(f"Failed to initialize Chatbot: {str(e)}")
    st.stop()

# Load context
context_loaded = False
pipeline_dict = {}
dataset_head = None

if os.path.exists("models/trained/analytics_pipeline.pkl"):
    pipeline_dict = AnalyticsPipeline.load("models/trained/analytics_pipeline.pkl")
    context_loaded = True

if 'raw_data' in st.session_state:
    dataset_head = st.session_state['raw_data'].head(5)
    context_loaded = True

if not context_loaded:
    st.info("No data or models found. Upload a dataset and train a model first to give the Copilot context.")

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Generate Insights Button
if st.sidebar.button("Generate Executive Insights"):
    if pipeline_dict:
        with st.spinner("Analyzing model results..."):
            meta = pipeline_dict.get('metadata', {})
            lb = pipeline_dict.get('leaderboard', pd.DataFrame())
            fi = pipeline_dict.get('feature_importance', pd.DataFrame())
            insights = bot.generate_insights(meta, lb, fi)
            st.session_state.messages.append({"role": "assistant", "content": insights})
            st.rerun()
    else:
        st.sidebar.error("Train a model first to generate insights.")

# Chat input
if prompt := st.chat_input("Ask about churn, feature importance, or data quality..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from Gemini
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = bot.ask_question(prompt, pipeline_dict, dataset_head)
            st.markdown(response)
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
