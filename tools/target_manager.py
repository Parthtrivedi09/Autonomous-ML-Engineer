class TargetManager:
    """
    Responsible for validating and storing
    the target column selected by the user.
    """

    def __init__(self, dataframe):

        self.df = dataframe

    def validate_target(self, target_column):

        """
        Checks whether the target column exists.
        """

        return target_column in self.df.columns

    def get_problem_type(self, target_column):

        """
        Determines whether the ML problem
        is Classification or Regression.
        """

        target = self.df[target_column]

        # Numeric target
        if target.dtype.kind in "if":

            unique = target.nunique()

            # Few unique values -> Classification
            

            ratio = unique / len(target)

            # Classification if few distinct values
            if unique <= 20 and ratio < 0.1:

                return "Classification"

            return "Regression"


        # String / category target
        return "Classification"