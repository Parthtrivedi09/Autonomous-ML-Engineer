from models.llm import llm

from prompts.model_selection_prompt import (
    MODEL_SELECTION_PROMPT
)

from utils.report_formatter import ReportFormatter


class ModelSelectionAgent:

    """
    Recommends baseline ML models
    based on the dataset characteristics.
    """

    def analyze(self, report):

        formatted_report = ReportFormatter.format_for_llm(
            report
        )

        prompt = f"""
{MODEL_SELECTION_PROMPT}

==================================================

MODEL PROFILE

{formatted_report}

==================================================
"""

        response = llm.invoke(prompt)

        return response.content