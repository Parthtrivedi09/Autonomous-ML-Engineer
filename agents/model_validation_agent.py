import json
import re

from models.llm import llm

from prompts.model_validation_prompt import (
    MODEL_VALIDATION_PROMPT
)


class ModelValidationAgent:

    """
    Converts user decisions into
    executable model selections.
    """

    def generate_models(

        self,

        problem_type,

        recommendations,

        user_response

    ):

        prompt = f"""
{MODEL_VALIDATION_PROMPT}

====================================

Problem Type

{problem_type}

====================================

Model Recommendations

{recommendations}

====================================

User Decisions

{user_response}
"""

        response = llm.invoke(prompt)

        clean = response.content.strip()

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

        start = clean.find("{")
        end = clean.rfind("}")

        if start != -1 and end != -1:

            clean = clean[start:end+1]

        try:

            return json.loads(clean)

        except json.JSONDecodeError:

            print("\nInvalid JSON Returned\n")

            print(response.content)

            return None