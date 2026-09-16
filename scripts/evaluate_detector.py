"""Evaluate pun-word ranking without calling Gemini or OpenAI."""

import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from sense_finder.sense_finder import (  # noqa: E402
    parse_sentence,
    rank_sense_candidates,
    retrieve_wordnet_candidates,
)


TEST_CASES = [
    ("I used to be a banker but I lost interest", "interest"),
    ("Broken pencils are pointless", "pointless"),
    ("The math teacher was a good ruler", "ruler"),
    ("A boiled egg in the morning is hard to beat", "beat"),
    ("I used to hate facial hair but then it grew on me", "grow"),
]


def evaluate():
    top1_correct = 0
    top3_correct = 0

    print("expected\tpredicted\ttop3\tresult")
    for sentence, expected in TEST_CASES:
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
        print(f"{expected}\t{predicted}\t{', '.join(top3)}\t{result}")

    total = len(TEST_CASES)
    print(f"\nTop-1 accuracy: {top1_correct}/{total} ({top1_correct / total:.0%})")
    print(f"Top-3 recall:   {top3_correct}/{total} ({top3_correct / total:.0%})")


if __name__ == "__main__":
    evaluate()
