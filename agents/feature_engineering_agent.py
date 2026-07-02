from models.llm import llm

from prompts.feature_engineering_prompt import (
    FEATURE_ENGINEERING_PROMPT
)

from utils.report_formatter import ReportFormatter


class FeatureEngineeringAgent:

    """
    Generates feature engineering recommendations
    for the cleaned dataset.
    """

    def analyze(
        self,
        report,
        problem_type
    ):

        formatted_report = ReportFormatter.format_for_llm(
            report
        )

        prompt = f"""
{FEATURE_ENGINEERING_PROMPT}

====================================

Problem Type

{problem_type}

====================================

FEATURE REPORT

{formatted_report}

====================================
"""

        response = llm.invoke(prompt)

        return response.content