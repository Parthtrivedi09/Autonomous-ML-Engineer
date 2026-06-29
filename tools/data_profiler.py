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

    
    
    def get_missing_values(self):
        """
        Returns the missing value statistics for every column.

        Instead of returning only the count, we also calculate the
        percentage of missing values because percentages are much
        easier for the LLM (and the user) to reason about.

        Example Output

        {
            "Age": {
                "count": 177,
                "percentage": 19.87
            },
            "Cabin": {
                "count": 687,
                "percentage": 77.10
            }
        }
        """

        missing_report = {}

        # Total number of rows in the dataset
        total_rows = len(self.df)

        # Loop through every column
        for column in self.df.columns:

            # Count missing values in this column
            missing_count = self.df[column].isnull().sum()

            # Only include columns that actually have missing values
            if missing_count > 0:

                missing_percentage = round(
                    (missing_count / total_rows) * 100,
                    2
                )

                missing_report[column] = {
                    "count": int(missing_count),
                    "percentage": missing_percentage
                }

        return missing_report
    

    def get_duplicates(self):
        """
        Returns the number of duplicate rows.

        Duplicate rows can bias the training process because the
        model may see the same information multiple times.
        """

        duplicate_count = self.df.duplicated().sum()

        return {
            "duplicate_rows": int(duplicate_count)
        }
    


    def get_unique_values(self):
        """
        Returns the number of unique values in every column.

        Why is this useful?

        - Columns where every value is unique may be identifiers
        (e.g., PassengerId, EmployeeID).

        - Columns with only one unique value provide no useful
        information for machine learning.

        Example Output

        {
            "PassengerId": 891,
            "Sex": 2,
            "Embarked": 3,
            "Cabin": 147
        }
        """

        unique_report = {}

        for column in self.df.columns:

            unique_report[column] = {
                "unique_values": int(self.df[column].nunique())
            }

        return unique_report



    def get_constant_columns(self):
        """
        Finds columns that contain only one unique value.

        Such columns have zero variance and do not help a
        machine learning model.

        Example

        Country = India
        Country = India
        Country = India

        This column carries no useful information.
        """

        constant_columns = []

        for column in self.df.columns:

            if self.df[column].nunique() == 1:
                constant_columns.append(column)

        return constant_columns



    def get_statistics(self):
        """
        Generates descriptive statistics for all numeric columns.

        Why?

        Instead of only knowing that a column is numeric,
        we also want to understand its distribution.

        These statistics will later help the LLM decide:

        - Whether outliers exist
        - Whether scaling is needed
        - Whether log transformation may help
        """

        statistics = {}

        # Select only numeric columns
        numeric_df = self.df.select_dtypes(include=["number"])

        # Loop through every numeric column
        for column in numeric_df.columns:

            statistics[column] = {

                "mean": round(float(numeric_df[column].mean()), 2),

                "median": round(float(numeric_df[column].median()), 2),

                "std": round(float(numeric_df[column].std()), 2),

                "min": round(float(numeric_df[column].min()), 2),

                "max": round(float(numeric_df[column].max()), 2),

                "q1": round(float(numeric_df[column].quantile(0.25)), 2),

                "q3": round(float(numeric_df[column].quantile(0.75)), 2)

            }

        return statistics



    def get_outliers(self):
        """
        Detects outliers using the IQR (Interquartile Range) method.

        We return only the number of outliers instead of the actual rows
        because the LLM only needs summary information to make
        recommendations.
        """

        outlier_report = {}

        numeric_df = self.df.select_dtypes(include=["number"])

        for column in numeric_df.columns:

            q1 = numeric_df[column].quantile(0.25)

            q3 = numeric_df[column].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr

            upper = q3 + 1.5 * iqr

            outliers = numeric_df[
                (numeric_df[column] < lower) |
                (numeric_df[column] > upper)
            ]

            outlier_report[column] = {

                "outlier_count": int(len(outliers))

            }

        return outlier_report
    


    def get_correlations(self, threshold=0.8):
        """
        Finds highly correlated pairs of numeric features.

        Why?

        Highly correlated features often contain redundant information,
        which may lead to multicollinearity in certain machine learning models.

        Instead of returning the complete correlation matrix,
        we only return feature pairs whose absolute correlation
        is greater than or equal to the specified threshold.
        """

        correlation_report = []

        # Select only numeric columns
        numeric_df = self.df.select_dtypes(include=["number"])

        # If there is only one numeric column,
        # correlation analysis is not possible.
        if numeric_df.shape[1] < 2:
            return correlation_report

        # Compute correlation matrix
        correlation_matrix = numeric_df.corr()

        columns = correlation_matrix.columns

        # Iterate over only the upper triangle
        # to avoid duplicate pairs.
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):

                corr_value = correlation_matrix.iloc[i, j]

                if abs(corr_value) >= threshold:

                    # Classify correlation strength
                    if abs(corr_value) >= 0.95:
                        strength = "Very Strong"
                    elif abs(corr_value) >= 0.90:
                        strength = "Strong"
                    else:
                        strength = "Moderate"

                    correlation_report.append({

                        "feature_1": columns[i],

                        "feature_2": columns[j],

                        "correlation": round(float(corr_value), 2),

                        "strength": strength

                    })

        return correlation_report
    



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
            "column_types": self.get_column_types(),
            "missing_values": self.get_missing_values(),
            "duplicates": self.get_duplicates(),
            "unique_values": self.get_unique_values(),
            "constant_columns": self.get_constant_columns(),
            "statistics": self.get_statistics(),
            "outliers": self.get_outliers(),
            "correlations": self.get_correlations()
        }

        return report