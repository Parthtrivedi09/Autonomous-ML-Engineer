import pandas as pd


class ModelProfiler:
    """
    Generates information required by the
    Model Selection Agent.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        problem_type: str
    ):

        self.df = dataframe

        self.target = target_column

        self.problem_type = problem_type

    # ======================================================
    # Dataset Shape
    # ======================================================

    def get_dataset_size(self):

        return {

            "rows": int(self.df.shape[0]),

            "columns": int(self.df.shape[1]),

            "feature_count": int(self.df.shape[1] - 1)

        }

    # ======================================================
    # Feature Types
    # ======================================================

    def get_feature_types(self):

        feature_df = self.df.drop(
            columns=[self.target]
        )

        numeric = feature_df.select_dtypes(
            include=["number"]
        ).shape[1]

        categorical = feature_df.select_dtypes(
            include=["object", "category"]
        ).shape[1]

        return {

            "numeric_features": int(numeric),

            "categorical_features": int(categorical)

        }

    # ======================================================
    # Missing Values
    # ======================================================

    def get_missing_values(self):

        return int(
            self.df.isnull().sum().sum()
        )

    # ======================================================
    # Classification Details
    # ======================================================

    def get_class_information(self):

        if self.problem_type != "Classification":

            return None

        target = self.df[self.target]

        return {

            "number_of_classes":
                int(target.nunique()),

            "class_distribution":
                target.value_counts().to_dict()

        }

    # ======================================================
    # Dataset Category
    # ======================================================

    def get_dataset_category(self):

        rows = len(self.df)

        if rows < 1000:

            return "Small"

        elif rows < 100000:

            return "Medium"

        return "Large"

    # ======================================================
    # Generate Report
    # ======================================================

    def generate_report(self):

        return {

            "problem_type":
                self.problem_type,

            "target_column":
                self.target,

            "dataset_size":
                self.get_dataset_size(),

            "dataset_category":
                self.get_dataset_category(),

            "feature_types":
                self.get_feature_types(),

            "missing_values":
                self.get_missing_values(),

            "class_information":
                self.get_class_information()

        }