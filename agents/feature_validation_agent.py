import json
import re

from models.llm import llm
from prompts.feature_validation_prompt import (
    FEATURE_VALIDATION_PROMPT
)


class FeatureValidationAgent:

    def generate_operations(

        self,

        report,

        problem_type,

        recommendations,

        user_response

):

        prompt = f"""
{FEATURE_VALIDATION_PROMPT}

====================================

FEATURE SUMMARY

{report}

====================================

RECOMMENDATIONS

{recommendations}

====================================

USER RESPONSE

{user_response}
"""

        response = llm.invoke(prompt)

        clean = response.content.strip()

        # Remove markdown blocks
        clean = re.sub(
            r"^```json\s*",
            "",
            clean,
            flags=re.MULTILINE
        )

        clean = re.sub(
            r"^```\s*",
            "",
            clean,
            flags=re.MULTILINE
        )

        clean = re.sub(
            r"\s*```$",
            "",
            clean
        )

        # Extract JSON if extra text exists
        start = clean.find("{")
        end = clean.rfind("}")

        if start != -1 and end != -1:

            clean = clean[start:end + 1]

        try:

            return json.loads(clean)

        except json.JSONDecodeError as e:

            print("\n========== JSON Parsing Error ==========\n")

            print(e)

            print("\n========== Raw LLM Response ==========\n")

            print(response.content)

            return None