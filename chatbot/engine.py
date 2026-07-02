"""
=========================================================
Analytics Copilot AI
Conversational LLM Chatbot
=========================================================

Author : Kumar Abhishek
"""

import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class ChatbotEngine:

    def __init__(self, api_key=None):
        self.api_key = api_key
        
        if not self.api_key:
            # Try Streamlit Secrets first
            try:
                import streamlit as st
                self.api_key = st.secrets.get("GEMINI_API_KEY")
            except:
                pass
                
        if not self.api_key:
            # Fallback to os.environ
            self.api_key = os.getenv("GEMINI_API_KEY")
            
        if not self.api_key:
            raise ValueError("Gemini API Key is missing. Please set GEMINI_API_KEY in Streamlit secrets or environment variables.")
            
        # Strip accidental quotes from TOML copy-pasting
        self.api_key = self.api_key.strip('"').strip("'").strip()
            
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=self.api_key,
            temperature=0.2
        )

    def generate_insights(self, metadata, leaderboard_df, feature_importance_df):
        """Generates proactive business insights based on the trained pipeline."""
        prompt = PromptTemplate.from_template(
            """
            You are an expert Data Scientist and Business Analyst.
            I have just trained a Machine Learning model. Here is the metadata:
            {metadata}
            
            Here is the Model Leaderboard:
            {leaderboard}
            
            Here is the Feature Importance (SHAP values):
            {features}
            
            Please provide a professional, concise executive summary of the results. 
            Highlight the best model, its performance, and the top 3 features that drive the target variable. 
            Provide 2-3 actionable business recommendations based on these features.
            Format the output in clean Markdown.
            """
        )
        
        chain = prompt | self.llm
        
        response = chain.invoke({
            "metadata": metadata,
            "leaderboard": leaderboard_df.head(5).to_string(),
            "features": feature_importance_df.head(10).to_string() if not feature_importance_df.empty else "Not available."
        })
        return response.content

    def ask_question(self, question, pipeline_dict, dataset_head=None):
        """Answers arbitrary user questions based on the dataset/model context."""
        
        # Build context string
        context = f"Problem Type: {pipeline_dict.get('problem_type')}\n"
        context += f"Target Variable: {pipeline_dict.get('target_column')}\n"
        context += f"Best Model: {pipeline_dict.get('best_model_name')} (Score: {pipeline_dict.get('best_score')})\n\n"
        
        if pipeline_dict.get('leaderboard') is not None:
            context += f"Leaderboard:\n{pipeline_dict['leaderboard'].head(5).to_string()}\n\n"
            
        if pipeline_dict.get('feature_importance') is not None and isinstance(pipeline_dict['feature_importance'], pd.DataFrame):
            context += f"Top Features:\n{pipeline_dict['feature_importance'].head(10).to_string()}\n\n"
            
        if dataset_head is not None:
            context += f"Dataset Sample (first 3 rows):\n{dataset_head.head(3).to_string()}\n"

        prompt = PromptTemplate.from_template(
            """
            You are 'Analytics Copilot', an elite AI Data Scientist assistant.
            Answer the user's question accurately using ONLY the context provided below.
            If the question cannot be answered from the context, politely state that you do not have enough data to answer it.
            Keep your answers concise, insightful, and strictly business-focused.
            
            Context:
            {context}
            
            Question:
            {question}
            
            Answer:
            """
        )
        
        chain = prompt | self.llm
        
        response = chain.invoke({"context": context, "question": question})
        return response.content
