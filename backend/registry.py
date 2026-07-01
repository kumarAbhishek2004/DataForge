"""
=========================================================
Model Registry
=========================================================
"""

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import AdaBoostClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier


class ClassificationRegistry:

    def __init__(self):

        self.models = {

            "Logistic Regression":

                LogisticRegression(

                    max_iter=1000,

                    random_state=42

                ),

            "Decision Tree":

                DecisionTreeClassifier(

                    random_state=42

                ),

            "Random Forest":

                RandomForestClassifier(

                    n_estimators=300,

                    random_state=42

                ),

            "Extra Trees":

                ExtraTreesClassifier(

                    n_estimators=300,

                    random_state=42

                ),

            "AdaBoost":

                AdaBoostClassifier(

                    random_state=42

                ),

            "XGBoost":

                XGBClassifier(

                    n_estimators=300,

                    learning_rate=0.05,

                    max_depth=6,

                    eval_metric="logloss",

                    random_state=42

                ),

            "LightGBM":

                LGBMClassifier(

                    random_state=42

                )

        }

    #######################################################

    def get_models(self):

        return self.models

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.tree import (
    DecisionTreeRegressor
)

from sklearn.ensemble import (

    RandomForestRegressor,

    ExtraTreesRegressor,

    AdaBoostRegressor

)

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor


class RegressionRegistry:

    def __init__(self):

        self.models = {

            "Linear Regression":

                LinearRegression(),

            "Ridge":

                Ridge(

                    alpha=1.0

                ),

            "Lasso":

                Lasso(

                    alpha=0.01

                ),

            "Decision Tree":

                DecisionTreeRegressor(

                    random_state=42

                ),

            "Random Forest":

                RandomForestRegressor(

                    n_estimators=300,

                    random_state=42

                ),

            "Extra Trees":

                ExtraTreesRegressor(

                    n_estimators=300,

                    random_state=42

                ),

            "AdaBoost":

                AdaBoostRegressor(

                    random_state=42

                ),

            "XGBoost":

                XGBRegressor(

                    n_estimators=300,

                    learning_rate=0.05,

                    max_depth=6,

                    random_state=42

                ),

            "LightGBM":

                LGBMRegressor(

                    random_state=42

                )

        }

    ########################################################

    def get_models(self):

        return self.models
