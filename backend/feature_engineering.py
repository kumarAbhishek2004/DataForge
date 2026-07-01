"""
=========================================================
Analytics Copilot AI
Feature Engineering Module
=========================================================
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression


class FeatureEngineer:

    def __init__(self):
        self.variance_selector = None
        self.selected_features = None

    def generate_date_features(self, df):
        """Extracts year, month, day, dayofweek from datetime columns."""
        df_out = df.copy()
        for col in df_out.select_dtypes(include=['datetime64', 'datetimetz']).columns:
            df_out[f"{col}_year"] = df_out[col].dt.year
            df_out[f"{col}_month"] = df_out[col].dt.month
            df_out[f"{col}_day"] = df_out[col].dt.day
            df_out[f"{col}_dayofweek"] = df_out[col].dt.dayofweek
            # Drop original date column to avoid issues with ML models
            df_out = df_out.drop(columns=[col])
        return df_out

    def generate_age_buckets(self, df, age_column='Age'):
        """Creates age buckets if an Age column exists."""
        df_out = df.copy()
        if age_column in df_out.columns:
            bins = [0, 18, 30, 45, 60, 100]
            labels = ['0-18', '19-30', '31-45', '46-60', '60+']
            df_out[f"{age_column}_Bucket"] = pd.cut(df_out[age_column], bins=bins, labels=labels)
        return df_out

    def variance_threshold_filter(self, df, threshold=0.01):
        """Removes low variance numerical features."""
        # Only apply to numeric columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            return df
        
        self.variance_selector = VarianceThreshold(threshold=threshold)
        self.variance_selector.fit(df[num_cols])
        
        retained_cols = num_cols[self.variance_selector.get_support()]
        dropped_cols = list(set(num_cols) - set(retained_cols))
        
        return df.drop(columns=dropped_cols)

    def correlation_filter(self, df, target, threshold=0.9):
        """Removes highly correlated features to prevent multicollinearity."""
        num_df = df.select_dtypes(include=[np.number])
        if target in num_df.columns:
            num_df = num_df.drop(columns=[target])
            
        corr_matrix = num_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return df.drop(columns=to_drop)
    
    def process_all(self, df, target=None):
        """Runs the entire feature engineering pipeline."""
        df = self.generate_date_features(df)
        df = self.generate_age_buckets(df)
        
        if target and target in df.columns:
            # We want to do feature selection
            # Separate features and target
            features = df.drop(columns=[target])
            features = self.variance_threshold_filter(features)
            features = self.correlation_filter(features, target)
            features[target] = df[target]
            return features
        return df
