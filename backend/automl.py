"""
=========================================================
Analytics Copilot AI
AutoML Engine
=========================================================

Author : Kumar Abhishek

This is the controller of the complete ML pipeline.
"""

import pandas as pd
import numpy as np
from pandas.api.types import is_integer_dtype, is_float_dtype

from backend.classification import ClassificationTrainer
from backend.regression import RegressionTrainer


class AutoML:

    def __init__(self):

        self.problem_type = None

        self.trainer = None

        self.results = None

    ###########################################################

    def detect_problem_type(
        self,
        df,
        target,
        user_choice="auto"
    ):
        """
        Auto Detect ML Problem

        Parameters
        ----------
        user_choice

        auto
        classification
        regression
        """

        if user_choice != "auto":

            self.problem_type = user_choice

            return self.problem_type

        y = df[target]

        unique = y.nunique()

        total = len(y)

        ratio = unique / total

        # Object columns are always classification

        if y.dtype == "object":

            self.problem_type = "classification"

        # Boolean

        elif str(y.dtype) == "bool":

            self.problem_type = "classification"

        # Numeric

        elif is_integer_dtype(y):

            # Binary

            if unique == 2:

                self.problem_type = "classification"

            # Multiclass

            elif ratio < 0.05 and unique <= 30:

                self.problem_type = "classification"

            else:

                self.problem_type = "regression"

        elif is_float_dtype(y):

            self.problem_type = "regression"

        else:

            self.problem_type = "classification"

        return self.problem_type

    ###########################################################

    def build_trainer(self):

        if self.problem_type == "classification":

            self.trainer = ClassificationTrainer()

        else:

            self.trainer = RegressionTrainer()

    ###########################################################

    def train(
        self,
        df,
        target,
        user_choice="auto",
        progress_callback=None,
        selected_models=None
    ):

        self.detect_problem_type(
            df,
            target,
            user_choice
        )

        self.build_trainer()

        self.results = self.trainer.train(
            df,
            target,
            progress_callback=progress_callback,
            selected_models=selected_models
        )

        return self.results

    ###########################################################

    def leaderboard(self):

        return self.results["leaderboard"]

    ###########################################################

    def pipeline_path(self):

        return self.results["pipeline_path"]

    ###########################################################

    def best_model(self):

        return self.results["best_model"]

    ###########################################################

    def best_model_name(self):

        return self.results["best_model_name"]

    ###########################################################

    def feature_importance(self):

        name = self.best_model_name()

        return self.results["results"][name]["feature_importance"]

    ###########################################################

    def metrics(self):

        name = self.best_model_name()

        return self.results["results"][name]["metrics"]

    ###########################################################

    def summary(self):

        print("=" * 80)

        print("Analytics Copilot AI")

        print("=" * 80)

        print(f"Problem Type : {self.problem_type}")

        print(f"Best Model   : {self.best_model_name()}")

        print(f"Score        : {self.results['best_score']}")

        print("=" * 80)

        print(self.leaderboard())

    ###########################################################

    def get_results(self):

        return self.results
