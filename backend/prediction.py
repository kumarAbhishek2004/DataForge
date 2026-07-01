"""
=========================================================
Analytics Copilot AI
Prediction Engine
=========================================================

Author : Kumar Abhishek
"""

import pandas as pd
import numpy as np
from backend.pipeline import AnalyticsPipeline


class PredictionEngine:

    def __init__(self, pipeline_path="models/trained/analytics_pipeline.pkl"):
        self.pipeline_dict = AnalyticsPipeline.load(pipeline_path)
        self.model = self.pipeline_dict["model"]
        self.preprocessor = self.pipeline_dict["preprocessor"]
        self.feature_columns = self.pipeline_dict["feature_columns"]
        self.problem_type = self.pipeline_dict["problem_type"]

    def predict(self, df):
        """Generates predictions and confidence scores for new datasets."""
        X = df.copy()
        
        # Preprocessing requires the original columns
        X_processed = self.preprocessor.transform(X)
        feature_names = self.preprocessor.get_feature_names()
        
        # Convert back to DataFrame to align columns exactly with training
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
        
        # Ensure only the exact features the model was trained on are passed
        # Missing columns in X_processed_df (e.g. unseen categories) will be handled by OneHotEncoder (ignore)
        # We just need to align to self.feature_columns
        X_final = X_processed_df[self.feature_columns]

        predictions = self.model.predict(X_final)
        
        results = pd.DataFrame({
            "Prediction": predictions
        })
        
        # Add Confidence Score for Classification
        if self.problem_type == "classification" and hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_final)
            confidence = np.max(proba, axis=1)
            results["Confidence_Score"] = np.round(confidence, 4)
            
        return results
