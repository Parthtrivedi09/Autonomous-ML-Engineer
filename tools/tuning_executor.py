import os
import joblib

from tools.hyperparameter_tuning_tool import (
    HyperparameterTuningTool
)

from tools.evaluation_tool import (
    EvaluationTool
)


class TuningExecutor:
    """
    Executes Phase 6 Hyperparameter Optimization.

    Responsibilities:

    1. Receive the model selected by the Evaluation Agent.
    2. Run GridSearchCV on training data only.
    3. Generate predictions on untouched test data.
    4. Evaluate the tuned model.
    5. Store baseline and tuned results.

    IMPORTANT:

    GridSearchCV NEVER receives X_test or y_test.

    The test set is used only after tuning is complete.
    """

    def __init__(
        self,
        problem_type
    ):

        self.problem_type = problem_type

        # ---------------------------------------------
        # GridSearchCV component
        # ---------------------------------------------

        self.tuning_tool = (
            HyperparameterTuningTool(
                problem_type
            )
        )

        # ---------------------------------------------
        # Metric computation component
        # ---------------------------------------------

        self.evaluation_tool = (
            EvaluationTool(
                problem_type
            )
        )

        self.result = None

    # =====================================================
    # Tune Selected Model
    # =====================================================

    def tune_model(
        self,
        model_name,
        baseline_result
    ):

        # ---------------------------------------------
        # Validate required experiment data
        # ---------------------------------------------

        required_keys = [
            "model",
            "metrics",
            "X_train",
            "y_train",
            "X_test",
            "y_test"
        ]

        missing_keys = [

            key

            for key in required_keys

            if key not in baseline_result

        ]

        if missing_keys:

            print(
                "\nTuning Error: "
                "Baseline result is incomplete."
            )

            print(
                "Missing:",
                missing_keys
            )

            return None

        # ---------------------------------------------
        # Extract baseline experiment data
        # ---------------------------------------------

        baseline_model = baseline_result[
            "model"
        ]

        baseline_metrics = baseline_result[
            "metrics"
        ]

        X_train = baseline_result[
            "X_train"
        ]

        y_train = baseline_result[
            "y_train"
        ]

        X_test = baseline_result[
            "X_test"
        ]

        y_test = baseline_result[
            "y_test"
        ]

        print(
            f"\nTuning {model_name} "
            f"using GridSearchCV..."
        )

        # ---------------------------------------------
        # Run GridSearchCV
        # ---------------------------------------------

        tuning_output = self.tuning_tool.tune(

            model_name=model_name,

            model=baseline_model,

            X_train=X_train,

            y_train=y_train

        )

        if tuning_output is None:

            return None

        # ---------------------------------------------
        # Get best tuned estimator
        # ---------------------------------------------

        tuned_model = tuning_output[
            "best_model"
        ]

        # ---------------------------------------------
        # Predict on untouched test data
        # ---------------------------------------------

        y_pred = tuned_model.predict(
            X_test
        )

        # ---------------------------------------------
        # Generate probability / decision scores
        # ---------------------------------------------

        y_prob = None

        if self.problem_type == "Classification":

            if hasattr(
                tuned_model,
                "predict_proba"
            ):

                probabilities = (
                    tuned_model.predict_proba(
                        X_test
                    )
                )

                # Binary classification
                if probabilities.shape[1] == 2:

                    y_prob = probabilities[:, 1]

                # Multiclass classification
                else:

                    y_prob = probabilities

            elif hasattr(
                tuned_model,
                "decision_function"
            ):

                y_prob = (
                    tuned_model.decision_function(
                        X_test
                    )
                )

        # ---------------------------------------------
        # Evaluate tuned model
        # ---------------------------------------------

        tuned_metrics = (
            self.evaluation_tool.evaluate(

                y_test,

                y_pred,

                y_prob

            )
        )

        # ---------------------------------------------
        # Store complete tuning result
        # ---------------------------------------------

        self.result = {

            "model_name":
                model_name,

            "baseline_model":
                baseline_model,

            "tuned_model":
                tuned_model,

            "baseline_metrics":
                baseline_metrics,

            "tuned_metrics":
                tuned_metrics,

            "best_parameters":
                tuning_output[
                    "best_parameters"
                ],

            "best_cv_score":
                tuning_output[
                    "best_cv_score"
                ],

            "scoring_metric":
                tuning_output[
                    "scoring_metric"
                ],

            "y_test":
                y_test,

            "y_pred":
                y_pred,

            "y_prob":
                y_prob

        }

        return self.result

    # =====================================================
    # Save Tuned Model
    # =====================================================

    def save_tuned_model(
        self,
        folder="artifacts/tuned_models"
    ):

        if self.result is None:

            print(
                "\nNo tuned model available to save."
            )

            return False

        os.makedirs(
            folder,
            exist_ok=True
        )

        model_name = self.result[
            "model_name"
        ]

        model_path = os.path.join(
            folder,
            f"{model_name}_tuned.pkl"
        )

        joblib.dump(
            self.result["tuned_model"],
            model_path
        )

        return True

    # =====================================================
    # Get Tuning Result
    # =====================================================

    def get_result(self):

        return self.result
    

    # =====================================================
    # Save Final Selected Model
    # =====================================================

    def save_final_model(
        self,
        final_decision,
        folder="artifacts/final_model"
    ):

        if self.result is None:

            print(
                "\nNo tuning result available."
            )

            return False

        os.makedirs(
            folder,
            exist_ok=True
        )

        model_name = self.result[
            "model_name"
        ]

        # ---------------------------------------------
        # Select estimator according to final decision
        # ---------------------------------------------

        if final_decision == "TunedModel":

            selected_model = self.result[
                "tuned_model"
            ]

        elif final_decision == "BaselineModel":

            selected_model = self.result[
                "baseline_model"
            ]

        else:

            print(
                "\nInvalid final model decision."
            )

            return False

        # ---------------------------------------------
        # Save selected final estimator
        # ---------------------------------------------

        model_path = os.path.join(
            folder,
            f"{model_name}_final.pkl"
        )

        joblib.dump(
            selected_model,
            model_path
        )

        return True