"""
=========================================================
Analytics Copilot AI
Classification Trainer
=========================================================
"""

from backend.trainer import BaseTrainer
from backend.registry import ClassificationRegistry
from backend.evaluation import EvaluationEngine


class ClassificationTrainer(BaseTrainer):

    def __init__(self):

        super().__init__()

        self.problem_type = "classification"

        self.registry = ClassificationRegistry()

        self.models = self.registry.get_models()

        self.primary_metric = "Accuracy"

    ########################################################

    def evaluate(

        self,

        model,

        prediction,

        probability

    ):

        if probability is not None:

            probability = probability[:, 1]

        metrics = EvaluationEngine.classification_metrics(

            self.y_test,

            prediction,

            probability

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
        
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        X_processed = preprocessor.fit_transform(X)

        feature_names = preprocessor.get_feature_names()

        processed_df = pd.DataFrame(
            X_processed,
            columns=feature_names
        )

        processed_df[target] = y_encoded

        self.preprocessor = preprocessor

        self.prepare_data(

            processed_df,

            target

        )

        scoring = [

            "accuracy",

            "precision",

            "recall",

            "f1",

            "roc_auc"

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

                metrics["Accuracy"]

            )
        
        if progress_callback:
            progress_callback(total_models, total_models, "Finalizing")

        self.create_leaderboard(

            self.primary_metric

        )

        self.summary()

        return self.get_results()
