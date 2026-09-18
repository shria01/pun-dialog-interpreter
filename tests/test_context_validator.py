import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from context_validator.context_validator import validate_context  # noqa: E402


class FakeProvider:
    def __init__(self, response):
        self.response = response

    def generate(self, _prompt):
        return self.response


class ContextValidatorTests(unittest.TestCase):
    def setUp(self):
        self.selected = {
            "word": "interest",
            "sense_a": "a fixed charge for borrowing money",
            "sense_b": "a feeling of concern and curiosity",
            "pun_score": 0.42,
        }

    def test_model_cannot_replace_detector_selection(self):
        provider = FakeProvider(
            """{
                "pun_word": "banker",
                "sense_a": "replacement A",
                "sense_b": "replacement B",
                "outcome": "confirmed",
                "candidate_word_valid": true,
                "sense_a_valid": true,
                "sense_b_valid": true,
                "pun_works": true,
                "reason": "Both fixed senses fit."
            }"""
        )

        result = validate_context("I lost interest", [self.selected], provider)

        self.assertEqual(result["pun_word"], "interest")
        self.assertEqual(result["sense_a"], self.selected["sense_a"])
        self.assertEqual(result["sense_b"], self.selected["sense_b"])
        self.assertTrue(result["pun_works"])

    def test_string_booleans_are_rejected(self):
        provider = FakeProvider(
            """{
                "sense_a_valid": "false",
                "sense_b_valid": "false",
                "pun_works": "false",
                "reason": "Malformed flags."
            }"""
        )

        result = validate_context("I lost interest", [self.selected], provider)

        self.assertFalse(result["sense_a_valid"])
        self.assertFalse(result["sense_b_valid"])
        self.assertFalse(result["pun_works"])
        self.assertEqual(result["outcome"], "validation_error")

    def test_wrong_candidate_is_distinct_from_not_a_pun(self):
        provider = FakeProvider(
            """{
                "outcome": "wrong_candidate",
                "candidate_word_valid": false,
                "sense_a_valid": true,
                "sense_b_valid": false,
                "pun_works": false,
                "reason": "The sentence has wordplay, but these senses do not capture it."
            }"""
        )

        result = validate_context("I lost interest", [self.selected], provider)

        self.assertEqual(result["outcome"], "wrong_candidate")
        self.assertFalse(result["pun_works"])
        self.assertIn("do not capture", result["reason"])
        self.assertIn("ranking error", result["detector_reason"])
        self.assertIn("combined score 0.420", result["detector_reason"])

    def test_fully_valid_wrong_candidate_is_normalized_to_confirmed(self):
        provider = FakeProvider(
            """{
                "outcome": "wrong_candidate",
                "candidate_word_valid": true,
                "sense_a_valid": true,
                "sense_b_valid": true,
                "pun_works": false,
                "reason": "Both meanings fit despite the incorrect outcome label."
            }"""
        )

        result = validate_context("I lost interest", [self.selected], provider)

        self.assertEqual(result["outcome"], "confirmed")
        self.assertTrue(result["pun_works"])
        self.assertNotIn("detector_reason", result)

    def test_not_a_pun_with_valid_sense_flags_is_normalized(self):
        provider = FakeProvider(
            """{
                "outcome": "not_a_pun",
                "candidate_word_valid": true,
                "sense_a_valid": true,
                "sense_b_valid": false,
                "pun_works": false,
                "reason": "Contradictory validation response."
            }"""
        )

        result = validate_context("I lost interest", [self.selected], provider)

        self.assertEqual(result["outcome"], "not_a_pun")
        self.assertFalse(result["candidate_word_valid"])
        self.assertFalse(result["sense_a_valid"])
        self.assertFalse(result["sense_b_valid"])
        self.assertFalse(result["pun_works"])

    def test_wordnet_phrase_gap_is_reported_when_observed(self):
        provider = FakeProvider(
            """{
                "outcome": "wrong_candidate",
                "candidate_word_valid": false,
                "sense_a_valid": true,
                "sense_b_valid": false,
                "pun_works": false,
                "reason": "The selected pair misses the phrase-level wordplay."
            }"""
        )
        phrase_candidate = {
            **self.selected,
            "coverage_gaps": [{
                "surface": "grew on",
                "wordnet_key": "grow_on",
            }],
        }

        result = validate_context(
            "I hated facial hair, but it grew on me",
            [phrase_candidate],
            provider,
        )

        self.assertIn("detector failed", result["detector_reason"])
        self.assertIn("not phrasal senses", result["detector_reason"])
        self.assertIn("WordNet has no entry", result["detector_reason"])
        self.assertIn("grow_on", result["detector_reason"])
        self.assertIn("phrase-level meaning", result["detector_reason"])

    def test_existing_phrasal_entries_are_identified_as_retrieval_limitation(self):
        provider = FakeProvider(
            """{
                "outcome": "wrong_candidate",
                "candidate_word_valid": false,
                "sense_a_valid": true,
                "sense_b_valid": false,
                "pun_works": false,
                "reason": "The selected pair misses the phrase-level wordplay."
            }"""
        )
        selected = {
            **self.selected,
            "word": "gravity",
            "sentence_coverage_gaps": [{
                "surface": "put down",
                "wordnet_key": "put_down",
                "wordnet_entry_found": True,
            }],
        }

        result = validate_context(
            "I read a book on anti-gravity. I couldn't put it down.",
            [selected],
            provider,
        )

        self.assertIn("put_down", result["detector_reason"])
        self.assertIn(
            "does not retrieve or score multiword entries",
            result["detector_reason"],
        )

    def test_correct_word_with_wrong_senses_is_reported_separately(self):
        provider = FakeProvider(
            """{
                "outcome": "wrong_candidate",
                "candidate_word_valid": true,
                "sense_a_valid": false,
                "sense_b_valid": false,
                "pun_works": false,
                "reason": "The word is correct, but these meanings are not."
            }"""
        )
        selected = {
            **self.selected,
            "word": "beat",
            "sense_a": "give a beating to",
            "sense_b": "be bewildering to",
        }

        result = validate_context(
            "A boiled egg in the morning is hard to beat",
            [selected],
            provider,
        )

        self.assertTrue(result["candidate_word_valid"])
        self.assertIn("identified “beat” as the pun word", result["detector_reason"])
        self.assertIn("could not fully verify", result["detector_reason"])
        self.assertIn("gloss may be narrower", result["detector_reason"])

    def test_pos_ambiguity_is_reported_as_parser_limitation(self):
        provider = FakeProvider(
            """{
                "outcome": "wrong_candidate",
                "candidate_word_valid": false,
                "sense_a_valid": false,
                "sense_b_valid": false,
                "pun_works": false,
                "reason": "The fixed candidate misses the structural ambiguity."
            }"""
        )
        selected = {
            **self.selected,
            "sentence_pos_ambiguities": [{
                "lemma": "fly",
                "forms": ["flies"],
                "parts_of_speech": ["NOUN", "VERB"],
            }],
            # The real spaCy parse also proposes "flies like" as a phrase.
            # POS ambiguity is the more specific diagnosis and must win.
            "sentence_coverage_gaps": [{
                "surface": "flies like",
                "normalized": "fly_like",
                "wordnet_entry_found": False,
            }],
        }

        result = validate_context(
            "Time flies like an arrow, but fruit flies like a banana",
            [selected],
            provider,
        )

        self.assertIn("syntactic ambiguity", result["detector_reason"])
        self.assertIn("NOUN and VERB", result["detector_reason"])
        self.assertNotIn("phrasal senses", result["detector_reason"])

    def test_plain_sentence_can_be_classified_as_not_a_pun(self):
        provider = FakeProvider(
            """{
                "outcome": "not_a_pun",
                "candidate_word_valid": false,
                "sense_a_valid": false,
                "sense_b_valid": false,
                "pun_works": false,
                "reason": "The sentence has no meaningful double interpretation."
            }"""
        )

        result = validate_context("The meeting starts at noon", [self.selected], provider)

        self.assertEqual(result["outcome"], "not_a_pun")
        self.assertFalse(result["pun_works"])


if __name__ == "__main__":
    unittest.main()
