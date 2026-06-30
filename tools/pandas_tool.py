import pandas as pd


class PandasTool:
    """
    Executes preprocessing operations on a pandas DataFrame.

    NOTE:
    The LLM never edits the dataframe directly.
    It only decides WHAT should happen.

    This class is responsible for actually applying
    those changes using pandas.
    """

    def __init__(self, dataframe: pd.DataFrame):

        # Store dataframe
        self.df = dataframe

        # Store every preprocessing operation
        self.operations_log = []

    # ==========================================================
    # Drop Column
    # ==========================================================

    def drop_column(self, column: str):

        """
        Drops a column from the dataframe.
        """

        if column not in self.df.columns:

            return False

        self.df.drop(columns=[column], inplace=True)

        self.operations_log.append({

            "operation": "drop_column",

            "column": column

        })

        return True

    # ==========================================================
    # Remove Duplicate Rows
    # ==========================================================

    def remove_duplicates(self):

        """
        Removes duplicate rows from the dataset.
        """

        before = len(self.df)

        self.df.drop_duplicates(inplace=True)

        after = len(self.df)

        removed = before - after

        self.operations_log.append({

            "operation": "remove_duplicates",

            "rows_removed": int(removed)

        })

        return True

    # ==========================================================
    # Median Imputation
    # ==========================================================

    def median_imputation(self, column):

        """
        Replaces missing values using the median.
        """

        if column not in self.df.columns:
            return False

        if not pd.api.types.is_numeric_dtype(
            self.df[column]
        ):

            print(
                f"Skipping median imputation for '{column}' "
                "because it is not numeric."
            )

            return False

        median = self.df[column].median()

        self.df[column] = self.df[column].fillna(
            median
        )

        self.operations_log.append({

            "operation": "median_imputation",

            "column": column,

            "median": float(median)

        })

        return True
    # ==========================================================
    # Mode Imputation
    # ==========================================================
    def mode_imputation(self, column):

        """
        Replaces missing values using the mode.
        """

        if column not in self.df.columns:
            return False

        mode = self.df[column].mode()

        if mode.empty:
            return False

        self.df[column] = self.df[column].fillna(
            mode.iloc[0]
        )

        self.operations_log.append({

            "operation": "mode_imputation",

            "column": column,

            "mode": mode.iloc[0]

        })

        return True

    # ==========================================================
    # Save Dataset
    # ==========================================================

    def save_dataset(self, path: str):

        """
        Saves the cleaned dataframe.
        """

        self.df.to_csv(path, index=False)

    # ==========================================================
    # Get Updated DataFrame
    # ==========================================================

    def get_dataframe(self):

        """
        Returns the updated dataframe.
        """

        return self.df

    # ==========================================================
    # Get Operation Log
    # ==========================================================

    def get_logs(self):

        """
        Returns every preprocessing operation.
        """

        return self.operations_log