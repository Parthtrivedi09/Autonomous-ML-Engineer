import json


class ReportFormatter:
    """
    Converts the raw profiler report into a clean JSON string
    that is easy for the LLM to understand.
    """

    @staticmethod
    def format_for_llm(report: dict) -> str:
        """
        Converts the Python dictionary into formatted JSON.
        """

        return json.dumps(
            report,
            indent=4,
            ensure_ascii=False
        )