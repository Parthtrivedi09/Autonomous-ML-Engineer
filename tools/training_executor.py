import os
import joblib

from tools.model_training_tool import ModelTrainingTool
from tools.evaluation_tool import EvaluationTool


class TrainingExecutor:
    """
    Responsible for:

    1. Training all selected baseline models.
    2. Evaluating every baseline model.
    3. Collecting experiment results.
    4. Preserving train/test data for later phases.
    5. Saving trained model artifacts.

    IMPORTANT:

    It does NOT decide which model is best.

    The Evaluation Agent compares the Python-computed
    metrics and selects the model for Phase 6.
    """

    def __init__(
        self,
        dataframe,
        target_column,
        problem_type
    ):

        self.problem_type = problem_type

        # ---------------------------------------------
        # Model training component
        # ---------------------------------------------

        self.training_tool = ModelTrainingTool(
            dataframe,
            target_column,
            problem_type
        )

        # ---------------------------------------------
        # Metric computation component
        # ---------------------------------------------

        self.evaluation_tool = EvaluationTool(
            problem_type
        )

        # Stores complete experiment results
        self.results = {}

    # =====================================================
    # Train All Selected Models
    # =====================================================

    def train_models(
        self,
        model_list
    ):

        # Reset results if method is called again
        self.results = {}

        for model_name in model_list:

            print(
                f"\nTraining {model_name}..."
            )

            # -----------------------------------------
            # Train baseline model
            # -----------------------------------------

            output = self.training_tool.train_model(
                model_name
            )

            # Skip unsupported model
            if output is None:

                print(
                    f"Skipping unsupported model: "
                    f"{model_name}"
                )

                continue

            # -----------------------------------------
            # Compute baseline metrics
            # -----------------------------------------

            metrics = self.evaluation_tool.evaluate(
                output["y_test"],
                output["y_pred"],
                output["y_prob"]
            )

            # -----------------------------------------
            # Store complete experiment result
            # -----------------------------------------

            self.results[model_name] = {

                # Trained baseline estimator
                "model": output["model"],

                # Python-computed metrics
                "metrics": metrics,

                # Training partition
                # Used later by GridSearchCV
                "X_train": output["X_train"],
                "y_train": output["y_train"],

                # Untouched test partition
                # Used for final tuned evaluation
                "X_test": output["X_test"],
                "y_test": output["y_test"],

                # Baseline predictions
                "y_pred": output["y_pred"],

                # Baseline probabilities
                "y_prob": output["y_prob"]

            }

        return self.results

    # =====================================================
    # Get One Trained Model
    # =====================================================

    def get_model(
        self,
        model_name
    ):

        if model_name not in self.results:

            return None

        return self.results[
            model_name
        ]["model"]

    # =====================================================
    # Get Complete Result For One Model
    # =====================================================

    def get_model_result(
        self,
        model_name
    ):

        if model_name not in self.results:

            return None

        return self.results[
            model_name
        ]

    # =====================================================
    # Save All Baseline Models
    # =====================================================

    def save_models(
        self,
        folder="artifacts/models"
    ):

        os.makedirs(
            folder,
            exist_ok=True
        )

        for model_name, result in self.results.items():

            model_path = os.path.join(
                folder,
                f"{model_name}.pkl"
            )

            joblib.dump(
                result["model"],
                model_path
            )

        return True

    # =====================================================
    # Get All Results
    # =====================================================

    def get_results(self):

        return self.results