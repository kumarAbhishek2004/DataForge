"""
=========================================================
Analytics Copilot AI
Universal Preprocessor
=========================================================

Author : Kumar Abhishek
"""

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder
)
from sklearn.impute import SimpleImputer


class DataPreprocessor:

    def __init__(self):

        self.numeric_columns = []

        self.categorical_columns = []

        self.preprocessor = None

    #########################################################

    def detect_columns(self, df):

        self.numeric_columns = list(

            df.select_dtypes(

                include=["int64", "float64"]

            ).columns

        )

        self.categorical_columns = list(

            df.select_dtypes(

                include=["object", "category", "bool"]

            ).columns

        )

    #########################################################

    def build_pipeline(self):

        numeric_pipeline = Pipeline(

            steps=[

                (

                    "imputer",

                    SimpleImputer(

                        strategy="median"

                    )

                ),

                (

                    "scaler",

                    StandardScaler()

                )

            ]

        )

        categorical_pipeline = Pipeline(

            steps=[

                (

                    "imputer",

                    SimpleImputer(

                        strategy="most_frequent"

                    )

                ),

                (

                    "encoder",

                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1
                    )

                )

            ]

        )

        self.preprocessor = ColumnTransformer(

            transformers=[

                (

                    "num",

                    numeric_pipeline,

                    self.numeric_columns

                ),

                (

                    "cat",

                    categorical_pipeline,

                    self.categorical_columns

                )

            ],

            remainder="drop"

        )

    #########################################################

    def fit_transform(self, X):

        self.detect_columns(X)

        self.build_pipeline()

        transformed = self.preprocessor.fit_transform(X)

        return transformed

    #########################################################

    def transform(self, X):

        return self.preprocessor.transform(X)

    #########################################################

    def get_feature_names(self):

        names = []

        names.extend(

            self.numeric_columns

        )

        if len(self.categorical_columns):

            cat = self.preprocessor.named_transformers_[

                "cat"

            ]

            encoder = cat.named_steps["encoder"]

            names.extend(

                encoder.get_feature_names_out(

                    self.categorical_columns

                )

            )

        return names

    #########################################################

    def save(

        self,

        path="models/trained/preprocessor.pkl"

    ):

        joblib.dump(

            self.preprocessor,

            path

        )

    #########################################################

    def load(

        self,

        path="models/trained/preprocessor.pkl"

    ):

        self.preprocessor = joblib.load(path)
