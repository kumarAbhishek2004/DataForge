"""
=========================================================
Analytics Copilot AI
Pipeline Manager
=========================================================

Stores everything required for prediction
inside one .pkl file.

Author : Kumar Abhishek
"""

import os
import joblib


class AnalyticsPipeline:

    def __init__(self):

        self.pipeline = {

            "model": None,

            "preprocessor": None,

            "feature_columns": None,

            "problem_type": None,

            "target_column": None,

            "best_model_name": None,

            "best_score": None,

            "leaderboard": None,

            "feature_importance": None,

            "metadata": {}

        }

    ############################################################

    def build(

        self,

        model,

        preprocessor,

        feature_columns,

        problem_type,

        target_column,

        best_model_name,

        best_score,

        leaderboard,

        feature_importance,

        metadata=None

    ):

        self.pipeline["model"] = model

        self.pipeline["preprocessor"] = preprocessor

        self.pipeline["feature_columns"] = feature_columns

        self.pipeline["problem_type"] = problem_type

        self.pipeline["target_column"] = target_column

        self.pipeline["best_model_name"] = best_model_name

        self.pipeline["best_score"] = best_score

        self.pipeline["leaderboard"] = leaderboard

        self.pipeline["feature_importance"] = feature_importance

        self.pipeline["metadata"] = metadata or {}

        return self.pipeline

    ############################################################

    def save(

        self,

        save_dir="models/trained",

        filename="analytics_pipeline.pkl"

    ):

        os.makedirs(

            save_dir,

            exist_ok=True

        )

        path = os.path.join(

            save_dir,

            filename

        )

        joblib.dump(

            self.pipeline,

            path

        )

        return path

    ############################################################

    @staticmethod

    def load(path):

        return joblib.load(path)
