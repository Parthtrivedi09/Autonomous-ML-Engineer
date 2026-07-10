import json
import re

from models.llm import llm

from prompts.tuning_evaluation_prompt import (
    TUNING_EVALUATION_PROMPT
)

from utils.report_formatter import (
    ReportFormatter
)


class TuningEvaluationAgent:
    """
    Reviews the completed hyperparameter optimization
    experiment.

    The LLM compares:

    - Baseline test metrics
    - Tuned test metrics
    - Best parameters
    - Cross-validation evidence

    It then recommends whether to retain the
    baseline or tuned estimator.
    """

    def analyze(
        self,
        problem_type,
        tuning_result
    ):

        # ---------------------------------------------
        # Validate required tuning evidence
        # ---------------------------------------------

        required_keys = [
            "model_name",
            "baseline_metrics",
            "tuned_metrics",
            "best_parameters",
            "best_cv_score",
            "scoring_metric"
        ]

        missing_keys = [
            key
            for key in required_keys
            if key not in tuning_result
        ]

        if missing_keys:

            print(
                "\nTuning Evaluation Error: "
                "Incomplete tuning result."
            )

            print(
                "Missing:",
                missing_keys
            )

            return None

        # ---------------------------------------------
        # Build evidence-only report
        # ---------------------------------------------

        evaluation_context = {

            "problem_type":
                problem_type,

            "model_name":
                tuning_result["model_name"],

            "scoring_metric":
                tuning_result["scoring_metric"],

            "best_cv_score":
                tuning_result["best_cv_score"],

            "best_parameters":
                tuning_result["best_parameters"],

            "baseline_metrics":
                tuning_result["baseline_metrics"],

            "tuned_metrics":
                tuning_result["tuned_metrics"]

        }

        formatted_context = (
            ReportFormatter.format_for_llm(
                evaluation_context
            )
        )

        # ---------------------------------------------
        # Build prompt
        # ---------------------------------------------

        prompt = f"""
{TUNING_EVALUATION_PROMPT}

==================================================

PYTHON-COMPUTED OPTIMIZATION RESULTS

{formatted_context}

==================================================
"""

        # ---------------------------------------------
        # Ask LLM to review tuning experiment
        # ---------------------------------------------

        response = llm.invoke(
            prompt
        )

        clean_response = response.content.strip()

        # ---------------------------------------------
        # Remove accidental Markdown fences
        # ---------------------------------------------

        clean_response = re.sub(
            r"^```json\s*",
            "",
            clean_response
        )

        clean_response = re.sub(
            r"^```\s*",
            "",
            clean_response
        )

        clean_response = re.sub(
            r"\s*```$",
            "",
            clean_response
        )

        # ---------------------------------------------
        # Parse JSON
        # ---------------------------------------------

        try:

            result = json.loads(
                clean_response
            )

        except json.JSONDecodeError as error:

            print(
                "\nTuning Evaluation JSON Parsing Error"
            )

            print(error)

            print("\nLLM Response:\n")

            print(response.content)

            return None

        # ---------------------------------------------
        # Validate final decision
        # ---------------------------------------------

        valid_decisions = [
            "TunedModel",
            "BaselineModel"
        ]

        final_decision = result.get(
            "final_decision"
        )

        if final_decision not in valid_decisions:

            print(
                "\nTuning Evaluation Error: "
                "Invalid final decision."
            )

            print(
                "Decision:",
                final_decision
            )

            return None

        # ---------------------------------------------
        # Validate model identity
        # ---------------------------------------------

        if result.get("model_name") != (
            tuning_result["model_name"]
        ):

            print(
                "\nTuning Evaluation Error: "
                "Model name mismatch."
            )

            return None

        return result