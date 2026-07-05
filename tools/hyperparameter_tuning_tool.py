from sklearn.model_selection import GridSearchCV


class HyperparameterTuningTool:
    """
    Performs Hyperparameter Optimization using GridSearchCV.

    Responsibilities:

    1. Store controlled parameter grids.
    2. Select the correct grid for a model.
    3. Run GridSearchCV.
    4. Return the best tuned estimator.

    The LLM does NOT generate parameter grids.
    """

    def __init__(
        self,
        problem_type
    ):

        self.problem_type = problem_type

    # =====================================================
    # Parameter Grids
    # =====================================================

    def get_parameter_grid(
        self,
        model_name
    ):

        parameter_grids = {

            # =============================================
            # Classification / Regression compatible
            # grids are selected according to model names
            # supported by the current project.
            # =============================================

            "LogisticRegression": {

                "C": [
                    0.01,
                    0.1,
                    1,
                    10,
                    100
                ],

                "solver": [
                    "liblinear",
                    "lbfgs"
                ],

                "max_iter": [
                    500,
                    1000
                ]

            },

            "DecisionTree": {

                "criterion": [
                    "gini",
                    "entropy"
                ],

                "max_depth": [
                    None,
                    3,
                    5,
                    10,
                    20
                ],

                "min_samples_split": [
                    2,
                    5,
                    10
                ],

                "min_samples_leaf": [
                    1,
                    2,
                    4
                ]

            },

            "RandomForest": {

                "n_estimators": [
                    100,
                    200,
                    300
                ],

                "max_depth": [
                    None,
                    5,
                    10,
                    20
                ],

                "min_samples_split": [
                    2,
                    5,
                    10
                ],

                "min_samples_leaf": [
                    1,
                    2,
                    4
                ]

            },

            "XGBoost": {

                "n_estimators": [
                    100,
                    200,
                    300
                ],

                "max_depth": [
                    3,
                    5,
                    7
                ],

                "learning_rate": [
                    0.01,
                    0.1,
                    0.2
                ],

                "subsample": [
                    0.8,
                    1.0
                ]

            },

            "SVM": {

                "C": [
                    0.1,
                    1,
                    10,
                    100
                ],

                "kernel": [
                    "linear",
                    "rbf"
                ],

                "gamma": [
                    "scale",
                    "auto"
                ]

            },

            "KNN": {

                "n_neighbors": [
                    3,
                    5,
                    7,
                    9,
                    11
                ],

                "weights": [
                    "uniform",
                    "distance"
                ],

                "metric": [
                    "euclidean",
                    "manhattan"
                ]

            }

        }

        return parameter_grids.get(
            model_name
        )

    # =====================================================
    # Scoring Strategy
    # =====================================================

    def get_scoring_metric(self):

        if self.problem_type == "Classification":

            return "f1"

        return "neg_mean_squared_error"

    # =====================================================
    # Grid Search
    # =====================================================

    def tune(
        self,
        model_name,
        model,
        X_train,
        y_train
    ):

        # ---------------------------------------------
        # Get model-specific parameter grid
        # ---------------------------------------------

        parameter_grid = self.get_parameter_grid(
            model_name
        )

        if parameter_grid is None:

            print(
                f"\nNo parameter grid available "
                f"for {model_name}."
            )

            return None

        # ---------------------------------------------
        # Select scoring strategy
        # ---------------------------------------------

        scoring = self.get_scoring_metric()

        # ---------------------------------------------
        # Create GridSearchCV
        # ---------------------------------------------

        grid_search = GridSearchCV(

            estimator=model,

            param_grid=parameter_grid,

            scoring=scoring,

            cv=5,

            n_jobs=-1,

            refit=True

        )

        # ---------------------------------------------
        # Run optimization
        # ---------------------------------------------

        grid_search.fit(
            X_train,
            y_train
        )

        # ---------------------------------------------
        # Return optimization result
        # ---------------------------------------------

        return {

            "best_model":
                grid_search.best_estimator_,

            "best_parameters":
                grid_search.best_params_,

            "best_cv_score":
                grid_search.best_score_,

            "scoring_metric":
                scoring

        }