import os
import joblib

from tools.model_training_tool import ModelTrainingTool
from tools.evaluation_tool import EvaluationTool


class TrainingExecutor:
    """
    Responsible for:

    1. Training all selected models.
    2. Evaluating every model.
    3. Collecting all results.
    4. Saving trained model artifacts.

    IMPORTANT:
    It does NOT decide which model is best.

    The Evaluation Agent compares the computed metrics
    and recommends the model for the next phase.
    """

    def __init__(
        self,
        dataframe,
        target_column,
        problem_type
    ):

        self.problem_type = problem_type

        # Tool responsible only for model training
        self.training_tool = ModelTrainingTool(
            dataframe,
            target_column,
            problem_type
        )

        # Tool responsible only for metric computation
        self.evaluation_tool = EvaluationTool(
            problem_type
        )

        # Stores training outputs for every model
        self.results = {}

    # =====================================================
    # Train All Models
    # =====================================================

    def train_models(
        self,
        model_list
    ):

        # Reset previous results in case this method
        # is called more than once
        self.results = {}

        for model_name in model_list:

            print(f"\nTraining {model_name}...")

            # ---------------------------------------------
            # Train model and generate predictions
            # ---------------------------------------------

            output = self.training_tool.train_model(
                model_name
            )

            # Unsupported or failed model
            if output is None:

                print(
                    f"Skipping unsupported model: "
                    f"{model_name}"
                )

                continue

            # ---------------------------------------------
            # Compute evaluation metrics
            # ---------------------------------------------

            metrics = self.evaluation_tool.evaluate(
                output["y_test"],
                output["y_pred"],
                output["y_prob"]
            )

            # ---------------------------------------------
            # Store complete experiment result
            # ---------------------------------------------

            self.results[model_name] = {

                "model": output["model"],

                "metrics": metrics,

                "X_test": output["X_test"],

                "y_test": output["y_test"],

                "y_pred": output["y_pred"],

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

        return self.results[model_name]["model"]

    # =====================================================
    # Save All Trained Models
    # =====================================================

    def save_models(
        self,
        folder="artifacts/models"
    ):

        # Create folder if it does not exist
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
    # Get Results
    # =====================================================

    def get_results(self):

        return self.results