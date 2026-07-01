"""
=========================================================
Analytics Copilot AI
Evaluation Engine
=========================================================

Author : Kumar Abhishek
"""

import numpy as np

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    roc_auc_score,

    confusion_matrix,

    mean_squared_error,

    mean_absolute_error,

    r2_score

)


class EvaluationEngine:

    #####################################################
    # Classification Metrics
    #####################################################

    @staticmethod
    def classification_metrics(

        y_true,

        y_pred,

        probability=None

    ):

        metrics = {

            "Accuracy": round(

                accuracy_score(

                    y_true,

                    y_pred

                ),

                4

            ),

            "Precision": round(

                precision_score(

                    y_true,

                    y_pred,

                    average="weighted",

                    zero_division=0

                ),

                4

            ),

            "Recall": round(

                recall_score(

                    y_true,

                    y_pred,

                    average="weighted",

                    zero_division=0

                ),

                4

            ),

            "F1 Score": round(

                f1_score(

                    y_true,

                    y_pred,

                    average="weighted",

                    zero_division=0

                ),

                4

            ),

            "Confusion Matrix":

                confusion_matrix(

                    y_true,

                    y_pred

                )

        }

        if probability is not None:

            try:

                metrics["ROC AUC"] = round(

                    roc_auc_score(

                        y_true,

                        probability

                    ),

                    4

                )

            except:

                metrics["ROC AUC"] = None

        return metrics

    #####################################################
    # Regression Metrics
    #####################################################

    @staticmethod
    def regression_metrics(

        y_true,

        y_pred

    ):

        rmse = np.sqrt(

            mean_squared_error(

                y_true,

                y_pred

            )

        )

        return {

            "RMSE": round(

                rmse,

                4

            ),

            "MAE": round(

                mean_absolute_error(

                    y_true,

                    y_pred

                ),

                4

            ),

            "R2": round(

                r2_score(

                    y_true,

                    y_pred

                ),

                4

            )

        }
