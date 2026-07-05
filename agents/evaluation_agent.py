import json
import re

from models.llm import llm

from prompts.evaluation_prompt import (
    EVALUATION_PROMPT
)

from utils.report_formatter import (
    ReportFormatter
)


class EvaluationAgent:
    """
    Reviews baseline model evaluation results.

    The LLM:
    1. Compares supplied metrics.
    2. Selects one model for tuning.
    3. Returns a machine-readable decision.

    Python remains responsible for all metric computation.
    """

    def analyze(
        self,
        problem_type,
        evaluation_results
    ):

        # ---------------------------------------------
        # Format Python-computed evaluation results
        # ---------------------------------------------

        formatted_results = (
            ReportFormatter.format_for_llm(
                evaluation_results
            )
        )

        # ---------------------------------------------
        # Build evaluation prompt
        # ---------------------------------------------

        prompt = f"""
{EVALUATION_PROMPT}

==================================================

PROBLEM TYPE

{problem_type}

==================================================

PYTHON-COMPUTED EVALUATION RESULTS

{formatted_results}

==================================================
"""

        # ---------------------------------------------
        # Ask LLM to compare models
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
        # Parse strict JSON
        # ---------------------------------------------

        try:

            result = json.loads(
                clean_response
            )

        except json.JSONDecodeError as error:

            print("\nEvaluation JSON Parsing Error")
            print(error)

            print("\nLLM Response:\n")
            print(response.content)

            return None

        # ---------------------------------------------
        # Basic structural validation
        # ---------------------------------------------

        if "selected_model" not in result:

            print(
                "\nEvaluation Error: "
                "'selected_model' missing."
            )

            return None

        # ---------------------------------------------
        # Critical model-name validation
        # ---------------------------------------------

        selected_model = result["selected_model"]

        if selected_model not in evaluation_results:

            print(
                "\nEvaluation Error: "
                "LLM selected an invalid model."
            )

            print(
                "Selected:",
                selected_model
            )

            print(
                "Available:",
                list(evaluation_results.keys())
            )

            return None

        return result