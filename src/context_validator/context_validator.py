import json
import re
from llm_interface.interface import LLMInterface

def validate_context(sentence, pun_word, sense_a, sense_b, provider: LLMInterface):
    prompt = f"""
You are evaluating whether a pun works.

Sentence: "{sentence}"
Pun word: "{pun_word}"

Sense A: {sense_a}
Sense B: {sense_b}

Return STRICT JSON:

{{
    "sense_a_valid": true or false,
    "sense_b_valid": true or false,
    "pun_works": true or false,
    "reason": "short explanation"
}}
"""

    try:
        text = provider.generate(prompt)
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = text.replace("```", "").strip()
        return json.loads(text)
    except Exception:
        return {
            "sense_a_valid": False,
            "sense_b_valid": False,
            "pun_works": False,
            "reason": "Could not parse model response."
        }
