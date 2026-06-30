import json
import re

from models.llm import llm
from prompts.validation_prompt import VALIDATION_PROMPT


class ValidationAgent:

    def generate_operations(
        self,
        report,
        recommendations,
        user_response
    ):

        prompt = f"""
            {VALIDATION_PROMPT}

            AI Recommendations:
            {recommendations}

            User Response:
            {user_response}
        """

        response = llm.invoke(prompt)

        clean_response = response.content.strip()

        # Remove markdown code blocks if present
        clean_response = re.sub(r"^```json\s*", "", clean_response, flags=re.MULTILINE)
        clean_response = re.sub(r"^```\s*", "", clean_response, flags=re.MULTILINE)
        clean_response = re.sub(r"\s*```$", "", clean_response)

        # Extract JSON if the model added extra text
        start = clean_response.find("{")
        end = clean_response.rfind("}")

        if start != -1 and end != -1:
            clean_response = clean_response[start:end + 1]

        try:
            return json.loads(clean_response)

        except json.JSONDecodeError as e:

            print("\n========== JSON Parsing Error ==========\n")

            print(e)

            print("\n========== Raw LLM Response ==========\n")

            print(response.content)

            return None