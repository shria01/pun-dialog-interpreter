import json
import re
import time
from llm_interface.interface import LLMInterface


def validate_context(sentence, candidates: list[dict], provider: LLMInterface) -> dict:
    selected = candidates[0]

    prompt = f"""
You are validating and explaining a pun detected by an NLP pipeline.

Sentence: "{sentence}"
Detected pun word: "{selected['word']}"
First WordNet sense: "{selected['sense_a']}"
Second WordNet sense: "{selected['sense_b']}"

The detected word and senses are fixed. Do not replace them, rewrite them,
or propose a different pun word. Judge whether each supplied sense applies
and briefly explain how the fixed senses create the wordplay.

Return STRICT JSON only, no markdown:
{{
    "sense_a_valid": true or false,
    "sense_b_valid": true or false,
    "pun_works": true or false,
    "reason": "short explanation of why the pun works or doesn't"
}}
"""
    started = time.perf_counter()
    try:
        text = provider.generate(prompt)
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = text.replace("```", "").strip()
        result = json.loads(text)
        required = {"sense_a_valid", "sense_b_valid", "pun_works", "reason"}
        if not required.issubset(result.keys()):
            raise ValueError("Missing required keys in response")
        boolean_fields = ("sense_a_valid", "sense_b_valid", "pun_works")
        if any(not isinstance(result[field], bool) for field in boolean_fields):
            raise ValueError("Validation flags must be JSON booleans")
        if not isinstance(result["reason"], str):
            raise ValueError("Validation reason must be a string")
        return {
            **selected,
            "pun_word": selected["word"],
            "sense_a_valid": result["sense_a_valid"],
            "sense_b_valid": result["sense_b_valid"],
            "pun_works": result["pun_works"],
            "reason": result["reason"],
        }
    except Exception as error:
        print(f"validation request failed: {type(error).__name__}: {error}", flush=True)
        return {
            **selected,
            "pun_word": selected["word"],
            "sense_a_valid": False,
            "sense_b_valid": False,
            "pun_works": False,
            "reason": "Could not validate the candidate meanings with the selected model."
        }
    finally:
        print(f"timing validate_context={time.perf_counter() - started:.3f}s", flush=True)
