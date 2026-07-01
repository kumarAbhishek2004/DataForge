"""
=========================================================
Analytics Copilot AI
Explainability Engine (SHAP)
=========================================================

Author : Kumar Abhishek
"""

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from backend.pipeline import AnalyticsPipeline

class ExplainabilityEngine:

    def __init__(self, pipeline_path="models/trained/analytics_pipeline.pkl"):
        self.pipeline_dict = AnalyticsPipeline.load(pipeline_path)
        self.model = self.pipeline_dict["model"]
        self.preprocessor = self.pipeline_dict["preprocessor"]
        self.feature_columns = self.pipeline_dict["feature_columns"]
        self.problem_type = self.pipeline_dict["problem_type"]
        self.explainer = None
        self.shap_values = None
        self.X_final = None
        
    def fit_explainer(self, X_train_sample):
        """Fits the SHAP explainer on a background dataset."""
        # Preprocess the sample
        X_processed = self.preprocessor.transform(X_train_sample)
        feature_names = self.preprocessor.get_feature_names()
        self.X_final = pd.DataFrame(X_processed, columns=feature_names)[self.feature_columns]
        
        # Use TreeExplainer if possible, otherwise KernelExplainer
        try:
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_values = self.explainer(self.X_final)
        except Exception:
            # Fallback for models not supported by TreeExplainer (like Linear Regression)
            background = shap.kmeans(self.X_final, 50)
            self.explainer = shap.KernelExplainer(self.model.predict, background)
            self.shap_values = self.explainer(self.X_final)
            
        return self.explainer, self.shap_values

    def plot_summary(self, save_path="reports/shap_summary.png"):
        """Generates and saves a SHAP summary plot."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(10, 6))
        
        if hasattr(self.shap_values, "values") and len(self.shap_values.values.shape) == 3:
             # Multiclass or binary with proba
             shap.summary_plot(self.shap_values.values[:, :, 1], self.X_final, show=False)
        else:
             shap.summary_plot(self.shap_values, self.X_final, show=False)
             
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path
        
    def plot_waterfall(self, instance_index=0, save_path="reports/shap_waterfall.png"):
        """Generates a waterfall plot for a single instance."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(10, 6))
        
        if hasattr(self.shap_values, "values"):
            if len(self.shap_values.values.shape) == 3:
                explanation = self.shap_values[instance_index, :, 1]
            else:
                explanation = self.shap_values[instance_index]
            shap.plots.waterfall(explanation, show=False)
            plt.tight_layout()
            plt.savefig(save_path)
        plt.close()
        return save_path

    def get_feature_importance_df(self, save_path="reports/feature_importance.csv"):
        """Returns and saves a DataFrame of mean absolute SHAP values."""
        if not hasattr(self.shap_values, "values"):
            return pd.DataFrame()
            
        vals = np.abs(self.shap_values.values)
        if len(vals.shape) == 3:
            vals = vals[:, :, 1]
            
        mean_shap = np.mean(vals, axis=0)
        
        df = pd.DataFrame({
            "Feature": self.feature_columns,
            "Importance": mean_shap
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        
        return df
