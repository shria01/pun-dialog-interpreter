"""Run deterministic detector regressions without calling Gemini or OpenAI.

This checks candidate ranking and WordNet sense selection only. It does not
measure end-to-end pun classification, which also depends on LLM validation.
"""

import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from sense_finder.sense_finder import (  # noqa: E402
    parse_sentence,
    rank_sense_candidates,
    retrieve_wordnet_candidates,
)


REGRESSION_CASES = [
    (
        "I used to be a banker but I lost interest",
        "interest",
        {"interest.n.01", "interest.n.04"},
    ),
    (
        "Broken pencils are pointless",
        "pointless",
        {"pointless.a.01", "otiose.s.01"},
    ),
    (
        "The math teacher was a good ruler",
        "ruler",
        {"ruler.n.01", "rule.n.12"},
    ),
    (
        "A boiled egg in the morning is hard to beat",
        "beat",
        {"beat.v.01", "beat.v.10"},
    ),
    (
        "I used to be addicted to soap, but I'm clean now.",
        "clean",
        {"clean.a.01", "clean.s.14"},
    ),
    (
        "The fisherman was hooked.",
        "hooked",
        {"hook.v.08", "dependent.s.04"},
    ),
    (
        "Time flies like an arrow, but fruit flies like a banana.",
        "flies",
        {"fly.v.08", "fly.n.01"},
    ),
]

# Formatting variants protect normalization behavior without inflating the main
# regression score by counting the same example twice.
VARIANT_CASES = [
    (
        "I used to be addicted to soap, but I’m clean now.",
        "clean",
        {"clean.a.01", "clean.s.14"},
    ),
]


def evaluate_cases(cases, label):
    """Evaluate one named group and return its pass counts."""
    top1_correct = 0
    top3_correct = 0
    sense_pair_correct = 0
    sense_pair_total = 0

    print(f"\n{label}")
    print("expected\tpredicted\ttop3\tword result\tsense result")
    for sentence, expected, expected_senses in cases:
        doc = parse_sentence(sentence)
        candidates = retrieve_wordnet_candidates(doc)
        ranked = rank_sense_candidates(sentence, candidates)
        predicted = ranked[0]["word"]
        top3 = [candidate["word"] for candidate in ranked]
        top1_match = predicted == expected
        top3_match = expected in top3
        top1_correct += int(top1_match)
        top3_correct += int(top3_match)
        result = "PASS" if top1_match else "MISS"
        sense_result = "N/A"
        expected_candidate = next(
            (candidate for candidate in ranked if candidate["word"] == expected),
            None,
        )
        if expected_senses is not None:
            sense_pair_total += 1
            sense_match = False
            if expected_candidate is not None:
                selected_senses = {
                    expected_candidate["sense_a_id"],
                    expected_candidate["sense_b_id"],
                }
                acceptable_pairs = (
                    expected_senses
                    if isinstance(expected_senses, list)
                    else [expected_senses]
                )
                sense_match = selected_senses in acceptable_pairs
            sense_pair_correct += int(sense_match)
            sense_result = "PASS" if sense_match else "MISS"
        print(
            f"{expected}\t{predicted}\t{', '.join(top3)}\t"
            f"{result}\t{sense_result}"
        )

    total = len(cases)
    print(f"Top-1 regression score: {top1_correct}/{total}")
    print(f"Top-3 regression score: {top3_correct}/{total}")
    if sense_pair_total:
        print(
            "Sense regression score: "
            f"{sense_pair_correct}/{sense_pair_total}"
        )
    passed = (
        top1_correct == total
        and top3_correct == total
        and sense_pair_correct == sense_pair_total
    )
    return passed


def evaluate():
    regression_passed = evaluate_cases(
        REGRESSION_CASES,
        "Core single-word regression cases",
    )
    variants_passed = evaluate_cases(
        VARIANT_CASES,
        "Normalization variants (reported separately)",
    )
    if not (regression_passed and variants_passed):
        raise AssertionError("Detector regression evaluation failed")



if __name__ == "__main__":
    evaluate()
