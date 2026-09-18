import json
import re
import time
from llm_interface.interface import LLMInterface


def explain_ranking_failure(selected, validation, candidates):
    """Describe the detector/validator disagreement using detector evidence."""
    invalid_senses = []
    if not validation["sense_a_valid"]:
        invalid_senses.append("Interpretation A")
    if not validation["sense_b_valid"]:
        invalid_senses.append("Interpretation B")
    rejected = " and ".join(invalid_senses) or "the sense pair"

    score = selected.get("pun_score")
    context_fit = selected.get("context_fit")
    sense_distance = selected.get("sense_distance")
    metrics = []
    if isinstance(score, (int, float)):
        metrics.append(f"combined score {score:.3f}")
    if isinstance(context_fit, (int, float)):
        metrics.append(f"context fit {context_fit:.3f}")
    if isinstance(sense_distance, (int, float)):
        metrics.append(f"sense distance {sense_distance:.3f}")
    metric_text = f" ({', '.join(metrics)})" if metrics else ""

    candidate_word_valid = validation["candidate_word_valid"]
    pos_ambiguities = [
        ambiguity
        for candidate in candidates
        for ambiguity in candidate.get("sentence_pos_ambiguities", [])
    ]
    if pos_ambiguities:
        unique = {
            ambiguity["lemma"]: ambiguity
            for ambiguity in pos_ambiguities
        }
        descriptions = [
            f'“{ambiguity["lemma"]}” as '
            + " and ".join(ambiguity["parts_of_speech"])
            for ambiguity in unique.values()
        ]
        return (
            "The detector failed because this wordplay depends on syntactic "
            f"ambiguity: {', '.join(descriptions)}. The current ranker evaluates "
            "WordNet senses inside one spaCy part-of-speech analysis at a time, so "
            "it cannot compare the sentence's competing grammatical parses."
        )

    coverage_gaps = [
        gap
        for candidate in candidates
        for gap in candidate.get(
            "sentence_coverage_gaps",
            candidate.get("coverage_gaps", []),
        )
    ]
    if coverage_gaps:
        unique_gaps = []
        seen_keys = set()
        for gap in coverage_gaps:
            if gap["wordnet_key"] not in seen_keys:
                unique_gaps.append(gap)
                seen_keys.add(gap["wordnet_key"])
        phrases = [
            f'“{gap["surface"]}” (`{gap["wordnet_key"]}`)'
            for gap in unique_gaps
        ]
        joined_phrases = ", ".join(phrases)
        entries_exist = any(
            gap.get("wordnet_entry_found", False) for gap in unique_gaps
        )
        if entries_exist:
            coverage_detail = (
                "WordNet contains entries for at least one of these phrases, but "
                "the current ranker does not retrieve or score multiword entries."
            )
        else:
            coverage_detail = (
                "WordNet has no entry for the complete phrase, so its phrase-level "
                "meaning is unavailable to the ranker."
            )
        consequence = (
            "It found the pun word, but selected the wrong WordNet sense pair."
            if candidate_word_valid
            else f'It therefore ranked “{selected["word"]}” instead of the pun word.'
        )
        return (
            "The detector failed because it analyzes individual words, not phrasal "
            f"senses. This wordplay depends on {joined_phrases}. {coverage_detail} "
            "The detector could not evaluate that phrase-level meaning. "
            f"{consequence}"
        )

    if candidate_word_valid:
        definition_score = selected.get("definition_pair_score")
        definition_context = selected.get("definition_context_fit")
        definition_distance = selected.get("definition_sense_distance")
        definition_metrics = []
        if isinstance(definition_score, (int, float)):
            definition_metrics.append(f"definition score {definition_score:.3f}")
        if isinstance(definition_context, (int, float)):
            definition_metrics.append(
                f"weaker context fit {definition_context:.3f}"
            )
        if isinstance(definition_distance, (int, float)):
            definition_metrics.append(
                f"sense distance {definition_distance:.3f}"
            )
        definition_metric_text = (
            f" ({', '.join(definition_metrics)})"
            if definition_metrics
            else ""
        )
        return (
            f'The detector identified “{selected["word"]}” as the pun word, but the '
            f"validation step could not fully verify {rejected}"
            f"{definition_metric_text}. The selected WordNet gloss may be narrower "
            "than the intended figurative reading, or the definition reranker may "
            "have selected a nearby sense. The pun word itself was detected correctly."
        )

    return (
        f'The ranker selected “{selected["word"]}” because its strongest '
        f"WordNet sense pair received the highest SBERT-based score{metric_text}. "
        f"The validation step rejected {rejected}, which means the ranking signal "
        "rewarded semantic similarity or sense contrast even though both meanings "
        "did not actually support the wordplay. This is a ranking error, not a "
        "no-pun result."
    )


def validate_context(sentence, candidates: list[dict], provider: LLMInterface) -> dict:
    selected = candidates[0]

    prompt = f"""
You are validating and explaining a pun detected by an NLP pipeline.

Sentence: "{sentence}"
Detected pun word: "{selected['word']}"
First WordNet sense: "{selected['sense_a']}"
Second WordNet sense: "{selected['sense_b']}"

The detected word and senses are fixed. Do not replace them, rewrite them,
or propose/name a different pun word. Judge whether each supplied sense applies,
and separately judge whether the detected word itself is the locus of the
sentence's apparent wordplay. A word can be correct even when its supplied
WordNet senses are wrong.

The detected word is valid only if that exact word carries the two readings.
A nearby clue word is not valid merely because it is related to the joke. For
example, in "Broken pencils are pointless," "pencil" is a clue while
"pointless" is the word whose two readings create the pun.

Validate the supplied definitions rather than silently substituting an unrelated
meaning of the same word. A definition may still be valid when the joke uses a
normal figurative extension or a broader everyday use of that definition. Do not
require every detail of a narrow dictionary gloss to be stated explicitly in the
sentence. For example, WordNet's "addicted to a drug" sense of "hooked" supports
the conventional broader reading "strongly interested or addicted." In contrast,
a sound-or-color definition of "clean" does not support "free from addiction."
Confirm the former kind of connection and reject the latter.

Do not confirm a pun merely because the detected word has two dictionary
meanings. The sentence must provide a recognizable contextual cue, setup, or
incongruity for each supplied reading. If only one reading is supported and the
other can be introduced only by arbitrarily substituting another dictionary
definition, classify the sentence as "not_a_pun". For example, mentioning a
musician supports the musical meaning of "bass," but without any fish-related
cue, "trouble with the bass" does not activate the fish meaning.

The reason must explicitly connect Interpretation A and Interpretation B as
written above. If it needs a genuinely different, unrelated dictionary meaning,
mark that supplied sense false and classify the result as "wrong_candidate".

then classify the result as exactly one of:
- "confirmed": the fixed word and both senses capture the pun.
- "wrong_candidate": the sentence appears to contain wordplay, but the fixed
  word or senses do not capture it.
- "not_a_pun": the sentence does not appear to contain meaningful wordplay.

Write a concise, user-facing reason for the classification:
- For "confirmed", explain how both fixed senses create the wordplay.
- For "wrong_candidate", identify which supplied sense or contextual connection
  fails and explain why this fixed pair does not capture the apparent wordplay.
- For "not_a_pun", explain why the sentence has only an ordinary reading.

Do not supply, name, or hint at a replacement candidate.

Return STRICT JSON only, no markdown:
{{
    "outcome": "confirmed" or "wrong_candidate" or "not_a_pun",
    "candidate_word_valid": true or false,
    "sense_a_valid": true or false,
    "sense_b_valid": true or false,
    "pun_works": true or false,
    "reason": "specific explanation of why this classification applies"
}}
"""
    started = time.perf_counter()
    try:
        text = provider.generate(prompt)
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = text.replace("```", "").strip()
        result = json.loads(text)
        required = {
            "outcome", "candidate_word_valid", "sense_a_valid", "sense_b_valid",
            "pun_works", "reason"
        }
        if not required.issubset(result.keys()):
            raise ValueError("Missing required keys in response")
        boolean_fields = (
            "candidate_word_valid", "sense_a_valid", "sense_b_valid", "pun_works"
        )
        if any(not isinstance(result[field], bool) for field in boolean_fields):
            raise ValueError("Validation flags must be JSON booleans")
        if not isinstance(result["reason"], str):
            raise ValueError("Validation reason must be a string")
        allowed_outcomes = {"confirmed", "wrong_candidate", "not_a_pun"}
        if result["outcome"] not in allowed_outcomes:
            raise ValueError("Unknown validation outcome")
        if result["outcome"] == "confirmed" and not all(
            result[field]
            for field in ("candidate_word_valid", "sense_a_valid", "sense_b_valid")
        ):
            # The model can call a result "confirmed" while honestly flagging one
            # supplied sense as a poor fit. That's really a wrong-candidate case
            # (right word, wrong sense), not a failure — reclassify instead of
            # discarding a usable answer.
            result["outcome"] = "wrong_candidate"
        elif result["outcome"] == "wrong_candidate" and all(
            result[field]
            for field in ("candidate_word_valid", "sense_a_valid", "sense_b_valid")
        ):
            # If the word and both senses are valid, this is a confirmed pun even
            # when the model returned the wrong outcome label.
            result["outcome"] = "confirmed"
        elif result["outcome"] == "not_a_pun":
            # A model may mark an ordinary dictionary sense as valid even though
            # it correctly concludes that the sentence contains no wordplay.
            # The outcome is what controls the UI, so normalize these subordinate
            # flags instead of turning a valid no-pun result into a validation error.
            result["candidate_word_valid"] = False
            result["sense_a_valid"] = False
            result["sense_b_valid"] = False
        result["pun_works"] = result["outcome"] == "confirmed"
        validated = {
            **selected,
            "pun_word": selected["word"],
            "outcome": result["outcome"],
            "candidate_word_valid": result["candidate_word_valid"],
            "sense_a_valid": result["sense_a_valid"],
            "sense_b_valid": result["sense_b_valid"],
            "pun_works": result["pun_works"],
            "reason": result["reason"],
        }
        if result["outcome"] == "wrong_candidate":
            validated["detector_reason"] = explain_ranking_failure(
                selected,
                result,
                candidates,
            )
        return validated
    except Exception as error:
        print(f"validation request failed: {type(error).__name__}: {error}", flush=True)
        return {
            **selected,
            "pun_word": selected["word"],
            "outcome": "validation_error",
            "candidate_word_valid": False,
            "sense_a_valid": False,
            "sense_b_valid": False,
            "pun_works": False,
            "reason": "Could not validate the candidate meanings with the selected model."
        }
    finally:
        print(f"timing validate_context={time.perf_counter() - started:.3f}s", flush=True)
