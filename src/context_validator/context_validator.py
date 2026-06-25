import json
import re
from llm_interface.interface import LLMInterface

def validate_context(sentence, candidates: list[dict], provider: LLMInterface) -> dict:
    candidates_text = "\n".join([
        f"- \"{c['word']}\": Sense A: {c['sense_a']} | Sense B: {c['sense_b']}"
        for c in candidates
    ])

    prompt = f"""
You are evaluating whether a pun works.

Sentence: "{sentence}"
Candidate pun words:
{candidates_text}

Return STRICT JSON only, no markdown:
{{
    "pun_word": "the actual pun word from the candidates above",
    "sense_a": "the first meaning that applies in context",
    "sense_b": "the second meaning that applies in context",
    "sense_a_valid": true or false,
    "sense_b_valid": true or false,
    "pun_works": true or false,
    "reason": "short explanation of why the pun works or doesn't"
}}
"""
    for attempt in range(3):
        try:
            text = provider.generate(prompt)
            text = re.sub(r"```(?:json)?\s*", "", text)
            text = text.replace("```", "").strip()
            result = json.loads(text)
            # verify all required keys are present
            required = {"pun_word", "sense_a", "sense_b", "sense_a_valid", "sense_b_valid", "pun_works", "reason"}
            if not required.issubset(result.keys()):
                raise ValueError("Missing required keys in response")
            return result
        except Exception:
            if attempt == 2:
                # fall back to top candidate
                return {
                    "pun_word": candidates[0]["word"],
                    "sense_a": candidates[0]["sense_a"],
                    "sense_b": candidates[0]["sense_b"],
                    "sense_a_valid": False,
                    "sense_b_valid": False,
                    "pun_works": False,
                    "reason": "Could not parse model response."
                }
