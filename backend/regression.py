"""
=========================================================
Analytics Copilot AI
Regression Trainer
=========================================================

Uses BaseTrainer for regression problems.

Author : Kumar Abhishek
"""

from backend.trainer import BaseTrainer
from backend.registry import RegressionRegistry
from backend.evaluation import EvaluationEngine


class RegressionTrainer(BaseTrainer):

    def __init__(self):

        super().__init__()

        self.problem_type = "regression"

        self.registry = RegressionRegistry()

        self.models = self.registry.get_models()

        self.primary_metric = "R2"

    ########################################################

    def evaluate(

        self,

        model,

        prediction,

        probability=None

    ):

        metrics = EvaluationEngine.regression_metrics(

            self.y_test,

            prediction

        )

        return metrics

    ########################################################

    def train(
        self,
        df,
        target,
        progress_callback=None,
        selected_models=None
    ):

        from backend.preprocessing import DataPreprocessor
        import pandas as pd

        if selected_models:
            self.models = {k: v for k, v in self.models.items() if k in selected_models}

        preprocessor = DataPreprocessor()

        X = df.drop(columns=[target])

        y = df[target]

        X_processed = preprocessor.fit_transform(X)

        feature_names = preprocessor.get_feature_names()

        processed_df = pd.DataFrame(
            X_processed,
            columns=feature_names
        )

        processed_df[target] = y.values

        self.preprocessor = preprocessor

        self.prepare_data(

            processed_df,

            target

        )

        scoring = [

            "r2",

            "neg_mean_absolute_error",

            "neg_mean_squared_error"

        ]

        total_models = len(self.models)
        for i, (model_name, model) in enumerate(self.models.items()):
            if progress_callback:
                progress_callback(i, total_models, model_name)

            metrics = self.train_single_model(

                model_name,

                model,

                scoring

            )

            self.update_best_model(

                model_name,

                model,

                metrics["R2"]

            )
            
        if progress_callback:
            progress_callback(total_models, total_models, "Finalizing")

        self.create_leaderboard(

            self.primary_metric

        )

        self.summary()

        return self.get_results()
