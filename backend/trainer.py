"""
=========================================================
Analytics Copilot AI
Base Trainer
=========================================================

Common training engine for Classification and Regression.

Author : Kumar Abhishek
"""

import time
import warnings
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod

from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    StratifiedKFold,
    KFold
)

warnings.filterwarnings("ignore")


class BaseTrainer(ABC):

    def __init__(self):

        self.model = None

        self.model_name = None

        self.problem_type = None

        self.preprocessor = None

        self.results = {}

        self.best_model = None

        self.best_model_name = None

        self.best_score = -999999

        self.leaderboard = None

        self.X_train = None

        self.X_test = None

        self.y_train = None

        self.y_test = None

        self.target = None

    ########################################################

    def prepare_data(

        self,

        df,

        target,

        test_size=0.2,

        random_state=42

    ):

        self.target = target

        X = df.drop(columns=[target])

        y = df[target]

        stratify = None

        if self.problem_type == "classification":

            stratify = y

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(

            X,

            y,

            test_size=test_size,

            random_state=random_state,

            stratify=stratify

        )

        return (

            self.X_train,

            self.X_test,

            self.y_train,

            self.y_test

        )

    ########################################################

    def get_cv(self):

        if self.problem_type == "classification":

            return StratifiedKFold(

                n_splits=5,

                shuffle=True,

                random_state=42

            )

        return KFold(

            n_splits=5,

            shuffle=True,

            random_state=42

        )

    ########################################################

    def fit_model(

        self,

        model

    ):

        start = time.time()

        model.fit(

            self.X_train,

            self.y_train

        )

        training_time = round(

            time.time()-start,

            3

        )

        return training_time

    ########################################################

    def predict(

        self,

        model

    ):

        start = time.time()

        prediction = model.predict(

            self.X_test

        )

        inference_time = round(

            time.time()-start,

            5

        )

        probability = None

        if hasattr(

            model,

            "predict_proba"

        ):

            probability = model.predict_proba(

                self.X_test

            )

        return (

            prediction,

            probability,

            inference_time

        )

    ########################################################

    def feature_importance(

        self,

        model

    ):

        if hasattr(

            model,

            "feature_importances_"

        ):

            importance = model.feature_importances_

        elif hasattr(

            model,

            "coef_"

        ):

            coef = model.coef_

            if len(np.shape(coef)) > 1:

                coef = coef[0]

            importance = np.abs(coef)

        else:

            return None

        return pd.DataFrame({

            "Feature": self.X_train.columns,

            "Importance": importance

        }).sort_values(

            by="Importance",

            ascending=False

        ).reset_index(drop=True)

    ########################################################

    @abstractmethod
    def evaluate(self, model, prediction, probability):
        pass

    ########################################################

    @abstractmethod
    def train(self, df, target):
        pass

    ########################################################
    # Cross Validation
    ########################################################

    def cross_validation(

        self,

        model,

        scoring

    ):

        cv = self.get_cv()

        scores = cross_validate(

            estimator=model,

            X=self.X_train,

            y=self.y_train,

            cv=cv,

            scoring=scoring,

            n_jobs=-1,

            return_train_score=False

        )

        summary = {}

        for key, value in scores.items():

            if key.startswith("test_"):

                metric = key.replace("test_", "")

                summary[metric] = {

                    "mean": round(np.mean(value), 4),

                    "std": round(np.std(value), 4)

                }

        return summary

    ########################################################
    # Store Result
    ########################################################

    def store_result(

        self,

        model_name,

        model,

        metrics,

        cv_result,

        feature_importance,

        training_time,

        inference_time

    ):

        self.results[model_name] = {

            "model": model,

            "metrics": metrics,

            "cv": cv_result,

            "feature_importance": feature_importance,

            "training_time": training_time,

            "inference_time": inference_time

        }

    ########################################################
    # Select Best Model
    ########################################################

    def update_best_model(

        self,

        model_name,

        model,

        score

    ):

        if score > self.best_score:

            self.best_score = score

            self.best_model = model

            self.best_model_name = model_name

    ########################################################
    # Train One Model
    ########################################################

    def train_single_model(

        self,

        model_name,

        model,

        scoring

    ):

        print(f"Training : {model_name}")

        training_time = self.fit_model(

            model

        )

        prediction, probability, inference_time = self.predict(

            model

        )

        metrics = self.evaluate(

            model,

            prediction,

            probability

        )

        cv_result = self.cross_validation(

            model,

            scoring

        )

        importance = self.feature_importance(

            model

        )

        self.store_result(

            model_name,

            model,

            metrics,

            cv_result,

            importance,

            training_time,

            inference_time

        )

        return metrics

    ########################################################
    # Create Leaderboard
    ########################################################

    def create_leaderboard(

        self,

        primary_metric

    ):

        rows = []

        for model_name, result in self.results.items():

            row = {

                "Model": model_name,

                "Training Time": result["training_time"],

                "Inference Time": result["inference_time"]

            }

            for metric_name, metric_value in result["metrics"].items():

                if metric_name == "Confusion Matrix":

                    continue

                if isinstance(metric_value, np.ndarray):

                    continue

                row[metric_name] = metric_value

            for cv_metric, values in result["cv"].items():

                row[f"CV {cv_metric}"] = values["mean"]

            rows.append(row)

        leaderboard = pd.DataFrame(rows)

        leaderboard = leaderboard.sort_values(

            by=primary_metric,

            ascending=False

        ).reset_index(drop=True)

        self.leaderboard = leaderboard

        return leaderboard

    ########################################################
    # Summary
    ########################################################

    def summary(self):

        print("\n")

        print("=" * 70)

        print("Training Summary")

        print("=" * 70)

        print(f"Problem Type : {self.problem_type}")

        print(f"Best Model   : {self.best_model_name}")

        print(f"Best Score   : {self.best_score}")

        print("=" * 70)

        print(self.leaderboard)

    ########################################################
    # Get Results
    ########################################################

    def get_results(self):

        files = self.finalize_training()

        return {

            "problem_type": self.problem_type,

            "best_model_name": self.best_model_name,

            "best_model": self.best_model,

            "best_score": self.best_score,

            "leaderboard": self.leaderboard,

            "results": self.results,

            "X_train": self.X_train,

            "X_test": self.X_test,

            "y_train": self.y_train,

            "y_test": self.y_test,

            **files

        }

    ########################################################
    # Reset Trainer
    ########################################################

    def reset(self):

        self.results = {}

        self.best_model = None

        self.best_model_name = None

        self.best_score = -999999

        self.leaderboard = None

        self.X_train = None

        self.X_test = None

        self.y_train = None

        self.y_test = None

        self.target = None

    ########################################################
    # Complete Training
    ########################################################

    def finalize_training(self):

        from backend.pipeline import AnalyticsPipeline

        pipeline = AnalyticsPipeline()

        feature_importance = None

        if self.best_model_name in self.results:

            feature_importance = self.results[

                self.best_model_name

            ]["feature_importance"]

        pipeline.build(

            preprocessor=self.preprocessor,

            model=self.best_model,

            feature_columns=list(

                self.X_train.columns

            ),

            problem_type=self.problem_type,

            target_column=self.target,

            best_model_name=self.best_model_name,

            best_score=self.best_score,

            leaderboard=self.leaderboard,

            feature_importance=feature_importance,

            metadata={

                "rows":

                len(self.X_train)+len(self.X_test),

                "features":

                len(self.X_train.columns)

            }

        )

        pipeline_path = pipeline.save()

        return {

            "pipeline_path":pipeline_path

        }
