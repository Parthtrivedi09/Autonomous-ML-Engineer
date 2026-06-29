from models.llm import llm
from prompts.analysis_prompt import ANALYSIS_PROMPT
from utils.report_formatter import ReportFormatter


class AnalysisAgent:

    """
    Sends the profiler report to the LLM
    and receives a professional dataset audit.
    """

    def analyze(self, report: dict):

        formatted_report = ReportFormatter.format_for_llm(report)

        prompt = f"""
        {ANALYSIS_PROMPT}

        ======================================================

        DATASET SUMMARY (JSON)

        {formatted_report}

        ======================================================
        """

        response = llm.invoke(prompt)

        return response.content