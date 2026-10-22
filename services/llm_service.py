
import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def analyze_statement(statement):

    prompt = f"""
You are a factuality and hallucination detection system.

Analyze the following statement:

"{statement}"

Determine whether the statement is factually correct.

Return ONLY valid JSON in exactly this format:

{{
    "score": 0.0,
    "label": "TRUE",
    "explanation": "Short explanation"
}}

Rules:
- score must be between 0.0 and 1.0
- 1.0 means definitely factual
- 0.0 means definitely false
- label must be TRUE, FALSE, or UNCERTAIN
- Do not include markdown or extra text.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    try:
        result = json.loads(response.text)
        return result

    except Exception:
        return {
            "score": 0.5,
            "label": "UNCERTAIN",
            "explanation": response.text
        }
