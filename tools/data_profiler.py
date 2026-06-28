import pandas as pd


class DataProfiler:
    """
    This class is responsible for extracting factual information
    from a pandas DataFrame.

    IMPORTANT:
    This class DOES NOT make recommendations.
    It only computes statistics and returns them.

    The LLM will later use this information to generate insights
    and recommendations.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Constructor

        Parameters:
            dataframe (pd.DataFrame): The dataset loaded by DataLoader.

        We store the dataframe inside the object so every method
        can access it using self.df.
        """

        self.df = dataframe

    def get_shape(self):
        """
        Returns the number of rows and columns in the dataset.

        Example:
        {
            "rows": 891,
            "columns": 12
        }
        """

        return {
            "rows": self.df.shape[0],      # Number of rows
            "columns": self.df.shape[1]    # Number of columns
        }

    def get_column_types(self):
        """
        Categorizes columns into Numeric, Categorical, and Datetime.

        Why?

        Different preprocessing techniques are applied to different
        types of columns.

        Numeric:
            Age, Salary, Fare

        Categorical:
            Gender, City

        Datetime:
            JoiningDate
        """

        numeric_columns = self.df.select_dtypes(
            include=["number"]
        ).columns.tolist()

        categorical_columns = self.df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        datetime_columns = self.df.select_dtypes(
            include=["datetime"]
        ).columns.tolist()

        return {
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "datetime_columns": datetime_columns
        }

    def generate_report(self):
        """
        This function combines all profiling information into
        one dictionary.

        Think of this as the final report that will later be
        passed to the LLM.

        As we build more methods, we'll keep adding them here.
        """

        report = {
            "shape": self.get_shape(),
            "column_types": self.get_column_types()
        }

        return report