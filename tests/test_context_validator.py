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


if __name__ == "__main__":
    unittest.main()
