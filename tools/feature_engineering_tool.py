import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)


class FeatureEngineeringTool:

    def __init__(self, dataframe):

        self.df = dataframe

        self.logs = []

        # Store fitted encoders/scalers
        self.fitted_objects = {}

    # ===================================================
    # One Hot Encoding
    # ===================================================

    def one_hot_encode(self, column):

        if column not in self.df.columns:
            return False

        self.df = pd.get_dummies(
            self.df,
            columns=[column],
            drop_first=False,
            dtype=int
        )

        self.logs.append({

            "operation": "one_hot_encode",

            "column": column

        })

        return True

    # ===================================================
    # Label Encoding
    # ===================================================

    def label_encode(self, column):

        if column not in self.df.columns:
            return False

        encoder = LabelEncoder()

        self.df[column] = encoder.fit_transform(
            self.df[column].astype(str)
        )

        self.fitted_objects[column] = encoder

        self.logs.append({

            "operation": "label_encode",

            "column": column

        })

        return True

    # ===================================================
    # Standard Scaling
    # ===================================================

    def standard_scale(self, column):

        if column not in self.df.columns:
            return False

        scaler = StandardScaler()

        self.df[[column]] = scaler.fit_transform(
            self.df[[column]]
        )

        self.fitted_objects[column] = scaler

        self.logs.append({

            "operation": "standard_scale",

            "column": column

        })

        return True

    # ===================================================
    # MinMax Scaling
    # ===================================================

    def minmax_scale(self, column):

        if column not in self.df.columns:
            return False

        scaler = MinMaxScaler()

        self.df[[column]] = scaler.fit_transform(
            self.df[[column]]
        )
        self.fitted_objects[column] = scaler

        self.logs.append({

            "operation": "minmax_scale",

            "column": column

        })

        return True

    # ===================================================
    # Robust Scaling
    # ===================================================

    def robust_scale(self, column):

        if column not in self.df.columns:
            return False

        scaler = RobustScaler()

        self.df[[column]] = scaler.fit_transform(
            self.df[[column]]
        )
        self.fitted_objects[column] = scaler

        self.logs.append({

            "operation": "robust_scale",

            "column": column

        })

        return True

    # ===================================================
    # Log Transform
    # ===================================================

    def log_transform(self, column):

        import numpy as np

        if (self.df[column] < 0).any():

            return False

        self.df[column] = np.log1p(
            self.df[column]
        )

        self.logs.append({

            "operation": "log_transform",

            "column": column

        })

        return True
    
    # ===================================================
    # Save Dataset
    # ===================================================

    def save_dataset(self, path):
        """
        Saves the feature engineered dataset.
        """

        self.df.to_csv(path, index=False)

    # ===================================================

    def get_dataframe(self):

        return self.df

    # ===================================================

    def get_logs(self):

        return self.logs
    
    def get_fitted_objects(self):

        return self.fitted_objects