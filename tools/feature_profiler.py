import pandas as pd


class FeatureProfiler:
    """
    Analyzes the cleaned dataset and extracts feature-level
    information for the Feature Engineering Agent.
    """

    def __init__(
        self,
        dataframe,
        target_column
    ):

        self.df = dataframe

        self.target = target_column

    # =======================================================
    # Categorical Columns
    # =======================================================

    def get_categorical_columns(self):

        categorical = {}

        feature_df = self.df.drop(
            columns=[self.target]
        )

        cat_df = feature_df.select_dtypes(
            include=["object", "category"]
        )

        for column in cat_df.columns:

            categorical[column] = {

                "unique_values": int(
                    feature_df[column].nunique()
                ),

                "missing_values": int(
                    feature_df[column].isnull().sum()
                )

            }

        return categorical

    # =======================================================
    # Numerical Columns
    # =======================================================

    def get_numeric_columns(self):

        numeric = {}

        feature_df = self.df.drop(
            columns=[self.target]
        )

        num_df = feature_df.select_dtypes(
            include=["number"]
        )

        for column in num_df.columns:

            numeric[column] = {

                "mean": round(float(num_df[column].mean()), 2),

                "std": round(float(num_df[column].std()), 2),

                "min": round(float(num_df[column].min()), 2),

                "max": round(float(num_df[column].max()), 2)

            }

        return numeric

    # =======================================================
    # Skewness
    # =======================================================

    def get_skewness(self):

        skew_report = {}

        feature_df = self.df.drop(
            columns=[self.target]
        )

        num_df = feature_df.select_dtypes(
            include=["number"]
        )

        for column in num_df.columns:

            skew = float(num_df[column].skew())

            skew_report[column] = round(skew, 2)

        return skew_report

    # =======================================================
    # Low Variance Features
    # =======================================================

    def get_low_variance_columns(self, threshold=0.01):

        low_variance = []

        feature_df = self.df.drop(
            columns=[self.target]
        )

        num_df = feature_df.select_dtypes(
            include=["number"]
        )

        for column in num_df.columns:

            if num_df[column].var() < threshold:

                low_variance.append(column)

        return low_variance

    # =======================================================
    # Generate Report
    # =======================================================

    def generate_report(self):

        return {

            "target_information":
                self.get_target_information(),

            "categorical_columns":
                self.get_categorical_columns(),

            "numeric_columns":
                self.get_numeric_columns(),

            "skewness":
                self.get_skewness(),

            "low_variance_columns":
                self.get_low_variance_columns()

        }
    
    def get_target_information(self):

        return {

            "target_column": self.target,

            "unique_values":
                int(
                    self.df[self.target].nunique()
                ),

            "dtype":
                str(
                    self.df[self.target].dtype
                )

        }